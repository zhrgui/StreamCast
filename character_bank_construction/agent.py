from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import json as _json
from metaphone import doublemetaphone as _doublemetaphone
import numpy as np
import requests as _requests

from actions import flush_unresolved, read_face, read_name, write_face, write_name
from clustering import cluster_tracks
from memory import CharacterBank
from schemas import FaceTrack, NameMention
from trajectory import ShotStep, ShotTrajectory, log_step
from vlm_linker import DEFAULT_MODEL_ID, DEFAULT_SAMPLE_FPS, _COLOR_NAMES, _FACE_QUALITY_PROMPT_LOOSE, _FACE_QUALITY_PROMPT_STRICT
from vlm_linker import filter_clear_face_tracks, run_vlm_name_linking
from tqdm import tqdm

CONFIDENCE_CAP = 1.00


class CharacterBankAgent:
    """Streams through shots of a movie, building a persistent CharacterBank."""
    def __init__(
        self,
        tau_face: float = 0.5,
        tau_tight_face: float = 0.45,
        tau_voice: float = 0.25,
        lang: str = "en",
        flush_min_shots: int = 3,
        video_file: str = "",
        base_vlm_url: Optional[str] = None,
        base_vlm_model_id: str = DEFAULT_MODEL_ID,
        vlm_linker_url: Optional[str] = None,
        vlm_linker_model_id: Optional[str] = None,
        vlm_sample_fps: float = DEFAULT_SAMPLE_FPS,
        vlm_max_tokens: int = 256,
        llm_url: Optional[str] = None,
        llm_model_id: str = "Qwen/Qwen3-8B",
        diar_model: str = "BUT-FIT/diarizen-wavlm-large-s80-md-v2",
        score_dir: str = "",
        imdbid: str = "",
        sync_threshold: float = -0.075,
        overwrite_threshold: float = 0.1,
        diar_threshold: float = 0.05,
        vis_every_k: int = 50,
        vis_output_dir: str = "",
        vlm_frames_vis_dir: str = "",
        vlm_face_filter_vis_dir: str = "",
        timeline_dir: str = "",
        verbose: bool = False,
    ):
        self.tau_face = tau_face
        self.tau_tight_face = tau_tight_face
        self.tau_voice = tau_voice
        self.lang = lang
        self.flush_min_shots = flush_min_shots
        self.video_file = video_file
        self.vlm_url = base_vlm_url
        self.vlm_model_id = base_vlm_model_id
        self.vlm_linker_url = vlm_linker_url or base_vlm_url
        self.vlm_linker_model_id = vlm_linker_model_id or base_vlm_model_id
        self.vlm_sample_fps = vlm_sample_fps
        self.vlm_max_tokens = vlm_max_tokens
        self.vlm_face_filter = True
        self.llm_url = llm_url
        self.llm_model_id = llm_model_id
        self.use_diarization = True
        self.diar_model = diar_model
        self.score_dir = score_dir
        self.imdbid = imdbid
        self.sync_threshold = sync_threshold
        self.overwrite_threshold = overwrite_threshold
        self.diar_threshold = diar_threshold
        self.vis_every_k = vis_every_k
        self.vis_output_dir = vis_output_dir
        self.vlm_frames_vis_dir = vlm_frames_vis_dir
        self.vlm_face_filter_vis_dir = vlm_face_filter_vis_dir
        self.timeline_dir = timeline_dir
        self.verbose = verbose
        self._cached_video_fps: Optional[float] = None

        # Runtime state (initialised by reset)
        self.bank: Optional[CharacterBank] = None
        self.raw_feats_store: Dict[int, List[np.ndarray]] = {}
        self.voice_feats_store: Dict[int, List[np.ndarray]] = {}
        self.voice_sample_log: Dict[int, List[Tuple[float, float]]] = {}
        # Name mention frequency across all shots (lowercased name → count)
        self._name_mention_counts: Dict[str, int] = defaultdict(int)
        # Secondary bank: named but not bank-ready tracks with no primary match
        self.secondary_bank: List[Dict] = []

    def reset(self, bank_path: str) -> None:
        """Load or create bank and clear per-run accumulators."""
        self.bank = CharacterBank.load(bank_path, tau_face=self.tau_face, tau_voice=self.tau_voice)
        self.raw_feats_store = defaultdict(list)
        self.voice_feats_store = defaultdict(list)
        self.voice_sample_log = defaultdict(list)
        self._processed_shots: set = set()
        self.secondary_bank = []
        self.name_filter_log: List[dict] = []
        self._name_mention_counts = defaultdict(int)

    def _get_video_fps(self) -> float:
        """Return video FPS (cached after first call)."""
        if self._cached_video_fps is not None:
            return self._cached_video_fps
        fps = 25.0
        if self.video_file:
            import cv2; cap = cv2.VideoCapture(self.video_file)
            v = cap.get(cv2.CAP_PROP_FPS); cap.release()
            if v > 0:
                fps = v
        self._cached_video_fps = fps
        return fps

    def _get_extended_window(
        self,
        shot_id: str,
        shot_seg: dict,
        n_before: int = 1,
        n_after: int = 1,
        seconds_after: Optional[float] = None,
        max_after_gap: Optional[float] = None,
        sid_to_subtitles: Optional[dict] = None,
    ) -> Tuple[Tuple[float, float], List[str]]:
        """Return ((start, end), [shot_ids]) covering shot_id ± neighbours.

        If ``seconds_after`` is given, the trailing side includes every shot
        whose time range overlaps ``[cur_end, cur_end + seconds_after]``
        instead of using ``n_after``.

        If ``max_after_gap`` is given (and ``seconds_after`` is not), the
        trailing extension stops once the subtitle gap exceeds this value:
        the first subtitle of the candidate shot must start within
        ``max_after_gap`` seconds of the last subtitle end of the previously
        included shot. Falls back to shot time_range when a shot has no
        subtitles.
        """
        shot_order = sorted(shot_seg.keys(), key=lambda x: int(x) if str(x).isdigit() else x)

        # Resolve key (may be int or str)
        sid_key: object = shot_id
        if shot_id not in shot_seg and shot_id.isdigit():
            sid_key = int(shot_id)

        idx = shot_order.index(sid_key)

        def _get_range(k):
            info = shot_seg.get(k, {})
            tr = info.get("time_range", [0.0, 0.0])
            return float(tr[0]), float(tr[1])

        def _sub_bounds(k):
            """Return (first_sub_start, last_sub_end) for shot k, falling back
            to its time_range when no subtitles are available."""
            tr_start, tr_end = _get_range(k)
            if sid_to_subtitles is None:
                return tr_start, tr_end
            subs = sid_to_subtitles.get(str(k)) or sid_to_subtitles.get(k) or []
            if not subs:
                return tr_start, tr_end
            first = float(subs[0].get("start", tr_start))
            last = float(subs[-1].get("end", tr_end))
            return first, last

        start_idx = max(0, idx - n_before)
        if seconds_after is not None:
            cur_end = _get_range(shot_order[idx])[1]
            threshold = cur_end + float(seconds_after)
            end_idx = idx + 1
            for j in range(idx + 1, len(shot_order)):
                t_start, _ = _get_range(shot_order[j])
                if t_start < threshold:
                    end_idx = j + 1
                else:
                    break
        elif max_after_gap is not None:
            end_idx = idx + 1
            _, last_end = _sub_bounds(shot_order[idx])
            for k in range(1, n_after + 1):
                j = idx + k
                if j >= len(shot_order):
                    break
                cand_first, cand_last = _sub_bounds(shot_order[j])
                if (cand_first - last_end) > max_after_gap:
                    break
                end_idx = j + 1
                last_end = cand_last
        else:
            end_idx = min(len(shot_order), idx + 1 + n_after)
        window_keys = shot_order[start_idx:end_idx]

        start = _get_range(window_keys[0])[0]
        end = _get_range(window_keys[-1])[1]
        return (start, end), [str(k) for k in window_keys]

    @staticmethod
    def _tighten_subs_with_diar(
        subs: List[dict],
        diar_turns: List[Tuple[float, float, str]],
        max_drift: float = 0.5,
    ) -> int:
        """Snap subtitle start/end to their speaker's diar-turn boundaries.

        For each sub with a ``diar_speaker``, find the speaker's turn with
        maximum overlap and snap each boundary if it's within ``max_drift``
        seconds. Boundaries beyond the drift tolerance are left untouched.
        Returns the number of subtitles whose timestamps changed.
        """
        if not diar_turns:
            return 0
        n_tightened = 0
        for sub in subs:
            spk = sub.get("diar_speaker")
            if not spk or spk == "none":
                continue
            s = float(sub.get("start", 0.0))
            e = float(sub.get("end", 0.0))
            best = None
            best_ov = 0.0
            for t0, t1, tspk in diar_turns:
                if tspk != spk:
                    continue
                ov = max(0.0, min(t1, e) - max(t0, s))
                if ov > best_ov:
                    best_ov = ov
                    best = (t0, t1)
            if best is None:
                continue
            t0, t1 = best
            new_s = t0 if abs(t0 - s) <= max_drift else s
            new_e = t1 if abs(t1 - e) <= max_drift else e
            if new_e > new_s and (new_s != s or new_e != e):
                sub["start"] = new_s
                sub["end"] = new_e
                n_tightened += 1
        return n_tightened

    def _llm_filter_onscreen_names(
        self,
        candidate_names: List[str],
        subtitles: List[dict],
    ) -> List[str]:
        """Filter names to only those likely on-screen via text LLM. Falls back to full list on failure."""
        if not self.llm_url or not candidate_names:
            return candidate_names

        sub_lines = []
        for sub in subtitles:
            text = sub.get("text", "").strip()
            if not text:
                continue
            speaker = sub.get("speaker", "unknown")
            sub_lines.append(f"[{speaker}]: {text}")
        if not sub_lines:
            return candidate_names

        sub_text = "\n".join(sub_lines)

        prompt = (
            "You are filtering character names for a movie scene.\n\n"
            "## Subtitle lines with speaker color labels\n"
            f"{sub_text}\n\n"
            "## Candidate names\n"
            f"{candidate_names}\n\n"
            "Keep only **reasonable human first names, last names or full names** (e.g. \"Nora\", \"Hammersmith\", \"John Smith\"). "
            "Discard job titles, roles, or professions (e.g. \"Doctor\", \"Captain\", \"Officer\").\n\n"
            "For each remaining candidate, classify how it appears in the subtitles:\n"
            "- **1st-person** (speaker identifies themselves — \"I'm Nora\", \"My name is Nora\"): KEEP\n"
            "- **2nd-person / direct address** (spoken to them — \"It's good seeing you, Nora\", \"Listen, Nora\", \"Couldn't do it without you, Aliah\", \"You did well, Nora\", \"I need you, Aliah\"): KEEP\n"
            "- **3rd-person introduction** (identifying someone present — \"This is Nora\", \"That's Nora\", \"Meet Nora\"): KEEP\n"
            "- **3rd-person narration** (voice-over / narrator describes the named character's actions as they happen — \"Nora walked into the room\", \"Juno stared at the test\", typically past-tense storytelling from an off-screen narrator, often with no interlocutor): KEEP\n"
            "- **3rd-person mention** (talked about in dialogue, not narration, and not present — \"Nora gave me a letter.\", \"I saw Nora\", \"Nora's shoe\"): REMOVE\n\n"
            "For **3rd-person** cases, don't classify the line in isolation — reason about the **flow of the dialogue**. "
            "Ask: does the conversation logic suggest this person is present and being referred to, "
            "or are they absent and merely being discussed? "
            "For example, if one speaker says \"Where's Nora?\" and another replies \"She's right here\", "
            "Nora is likely present despite neither line being a direct address or introduction. "
            "Narration is distinguished from ordinary mention by its descriptive/storytelling register and the absence of a back-and-forth exchange — a narrator describing \"Nora\" is almost always showing Nora on-screen.\n\n"
            "Goal: infer whether the character is likely **visible on-screen**. When in doubt, REMOVE.\n\n"
            "Reply with ONLY a JSON array of objects, one per candidate. Each object must have:\n"
            "- \"name\": the candidate name\n"
            "- \"classification\": one of \"1st-person\", \"2nd-person / direct address\", \"3rd-person introduction\", \"3rd-person narration\", \"3rd-person mention\", \"not a human name\"\n"
            "- \"reasoning\": one sentence explaining why\n"
            "- \"decision\": \"keep\" or \"remove\"\n\n"
            "Example:\n"
            "[{\"name\": \"Nora\", \"classification\": \"2nd-person / direct address\", "
            "\"reasoning\": \"The line 'Listen, Nora' is spoken directly to her.\", \"decision\": \"keep\"}]\n"
            "If none qualify, reply: []"
        )

        payload = {
            "model": self.llm_model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        endpoint = self.llm_url.rstrip("/") + "/chat/completions"
        try:
            resp = _requests.post(endpoint, json=payload, timeout=60)
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]
            raw = raw.split("</think>")[-1].strip()
            kept = _json.loads(raw)

            if not isinstance(kept, list):
                return candidate_names

            # Parse structured output
            kept_lower = set()
            for item in kept:
                if isinstance(item, dict):
                    name = item.get("name", "")
                    decision = item.get("decision", "").lower()
                    if isinstance(name, str) and decision == "keep":
                        kept_lower.add(name.lower())
                elif isinstance(item, str):
                    # Fallback: plain string
                    kept_lower.add(item.lower())

            filtered = [
                n for n in candidate_names
                if n.lower() in kept_lower
            ]
            if self.verbose:
                removed = [
                    n for n in candidate_names
                    if n.lower() not in kept_lower
                ]
                if removed:
                    print(f"  [onscreen-filter] removed off-screen: {removed}")

            self.name_filter_log.append({
                "candidates": candidate_names,
                "kept": filtered,
                "removed": [n for n in candidate_names if n not in filtered],
                "llm_response": kept,
                "subtitles": [
                    {"speaker": s.get("speaker", "unknown"),
                     "text": s.get("text", "").strip()}
                    for s in subtitles if s.get("text", "").strip()
                ],
            })
            return filtered
        except Exception as exc:
            if self.verbose:
                print(f"  [onscreen-filter] failed ({exc}), keeping all")
            return candidate_names

    def _calibrate_names_metaphone(self) -> None:
        """Prune noisy name variants per cluster using Double Metaphone grouping.

        Within each phonetic group the winner is the name that was mentioned
        most often across all shots (tracked in ``_name_mention_counts``),
        with vote score as a secondary tiebreaker.
        """
        for entry in self.bank.all_entries():
            if len(entry.name_votes) <= 1:
                continue

            # Group name variants by Double Metaphone code of their first token
            code_to_names: Dict[str, List[str]] = {}
            for name in entry.name_votes:
                first_token = name.split()[0]
                primary, secondary = _doublemetaphone(first_token)
                key = primary or name.lower()
                code_to_names.setdefault(key, []).append(name)
                if secondary and secondary != primary:
                    code_to_names.setdefault(secondary, []).append(name)

            # Deduplicate: a name may appear in multiple code buckets.
            # Process only the largest bucket that contains each name.
            seen_names: set = set()
            for names in sorted(code_to_names.values(), key=len, reverse=True):
                unique = [n for n in names if n not in seen_names]
                if len(unique) <= 1:
                    seen_names.update(unique)
                    continue
                seen_names.update(unique)

                # Pick winner: highest vote score, then global mention count,
                # then recency (order in name_votes dict).
                vote_keys = list(entry.name_votes.keys())
                winner = max(unique, key=lambda n: (
                    entry.name_votes.get(n, 0.0),
                    self._name_mention_counts.get(n.lower(), 0),
                    vote_keys.index(n) if n in vote_keys else -1,
                ))
                for loser in unique:
                    if loser == winner:
                        continue
                    loser_vote = entry.name_votes.pop(loser, 0.0)
                    entry.name_votes[winner] = entry.name_votes.get(winner, 0.0) + loser_vote
                    if self.verbose:
                        print(f"  [name-merge] cluster {entry.cluster_id}: "
                              f"\"{loser}\" ({loser_vote:.2f}) → \"{winner}\" "
                              f"(now {entry.name_votes[winner]:.2f}, "
                              f"mentions={self._name_mention_counts.get(winner.lower(), 0)})")

            # Recalculate canonical from remaining votes
            if entry.name_votes:
                vote_keys = list(entry.name_votes.keys())
                best_name = max(vote_keys, key=lambda n: (
                    entry.name_votes[n],
                    self._name_mention_counts.get(n.lower(), 0),
                    vote_keys.index(n),
                ))
                entry.name = best_name
                entry.confidence = entry.name_votes[best_name]

    @staticmethod
    def _resolve_collisions(
        cluster_state: Dict[int, Tuple[Dict[str, float], int]],
    ) -> Dict[int, Tuple[Optional[str], float]]:
        """Assign a unique canonical name per cluster given their name votes.

        Input: {cluster_id: (name_votes, evidence_count)}. When two clusters'
        top-voted names collide, the one with the higher vote on that name
        keeps it and the others advance to their next-highest vote. Iterates
        until stable. Returns {cluster_id: (name, confidence)} where
        confidence is the normalised score (score/total * support).
        """
        ranked: Dict[int, List[Tuple[str, float]]] = {
            cid: sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
            for cid, (votes, _) in cluster_state.items()
        }
        ptr: Dict[int, int] = {cid: 0 for cid in cluster_state}

        def current(cid: int) -> Tuple[Optional[str], float]:
            if not ranked[cid]:
                return None, 0.0
            p = min(ptr[cid], len(ranked[cid]) - 1)
            name, score = ranked[cid][p]
            return name, score

        max_iter = max(5, sum(len(r) for r in ranked.values()) + 1)
        for _ in range(max_iter):
            by_name: Dict[str, List[int]] = defaultdict(list)
            for cid in cluster_state:
                name, _ = current(cid)
                if name is not None:
                    by_name[name].append(cid)

            collisions = [cids for cids in by_name.values() if len(cids) > 1]
            if not collisions:
                break

            changed = False
            for cids in collisions:
                # Winner = highest vote on the shared name; tiebreak by cid
                cids_sorted = sorted(cids, key=lambda c: (-current(c)[1], c))
                for loser in cids_sorted[1:]:
                    if ptr[loser] + 1 < len(ranked[loser]):
                        ptr[loser] += 1
                        changed = True
            if not changed:
                break

        result: Dict[int, Tuple[Optional[str], float]] = {}
        for cid, (votes, ec) in cluster_state.items():
            name, score = current(cid)
            if name is None:
                result[cid] = (None, 0.0)
                continue
            total = sum(votes.values()) or 1.0
            support = min(ec / 3, 1.0)
            result[cid] = (name, min(score / total, 1.0) * support)
        return result

    def _resolve_name_collisions(self) -> None:
        """Apply cross-cluster name collision resolution to the live bank.

        name_votes is never modified — only entry.name, entry.confidence,
        and entry.aliases change.
        """
        entries = list(self.bank.all_entries())
        state = {
            e.cluster_id: (dict(e.name_votes), len(e.evidence_log))
            for e in entries
        }
        resolved = self._resolve_collisions(state)
        for e in entries:
            name, conf = resolved[e.cluster_id]
            if self.verbose and e.name and e.name != name:
                print(f"  [name-collision] cluster {e.cluster_id}: "
                      f"'{e.name}' → '{name}' (conf={conf:.3f})")
            e.name = name
            e.confidence = conf
            e.aliases = [n for n in e.name_votes if n != name] if e.name_votes else []

    def _filter_unusable_tracks(
        self,
        face_tracks: List[FaceTrack],
        track_detections: Dict[str, list],
        shot_id: str,
    ) -> List[FaceTrack]:
        """Loose VLM filter: removes back-of-head, non-human, cartoon, tiny faces."""
        before = len(face_tracks)
        filtered = filter_clear_face_tracks(
            face_tracks=face_tracks,
            track_detections=track_detections,
            video_path=self.video_file,
            vlm_url=self.vlm_url,
            model_id=self.vlm_model_id,
            face_filter_vis_dir=self.vlm_face_filter_vis_dir or None,
            face_filter_vis_tag=f"{shot_id}_pre",
            quality_prompt=_FACE_QUALITY_PROMPT_LOOSE,
        )
        if self.verbose:
            dropped = before - len(filtered)
            print(f"  [face-filter-loose] {before} tracks → {len(filtered)} kept "
                  f"({dropped} dropped as unusable)")
        return filtered

    def _filter_bank_ready(
        self,
        face_tracks: List[FaceTrack],
        track_detections: Dict[str, list],
        shot_id: str,
    ) -> List[FaceTrack]:
        """Strict VLM filter: keeps only clear, identifiable faces for the bank."""
        before = len(face_tracks)
        filtered = filter_clear_face_tracks(
            face_tracks=face_tracks,
            track_detections=track_detections,
            video_path=self.video_file,
            vlm_url=self.vlm_url,
            model_id=self.vlm_model_id,
            face_filter_vis_dir=self.vlm_face_filter_vis_dir or None,
            face_filter_vis_tag=f"{shot_id}_strict",
            quality_prompt=_FACE_QUALITY_PROMPT_STRICT,
        )
        if self.verbose:
            dropped = before - len(filtered)
            print(f"  [face-filter-strict] {before} tracks → {len(filtered)} kept "
                  f"({dropped} dropped as not bank-ready)")
        return filtered

    def _cluster_shot_faces(self, face_tracks: List[FaceTrack]) -> List[FaceTrack]:
        """Cluster face tracks within a single shot; return one rep per cluster."""
        face_only = [ft for ft in face_tracks if ft.has_face and ft.embedding is not None]
        if not face_only:
            return []

        track_inputs = [
            {
                "track_idx": ft.track_id,
                "track_occurence_indices": ft.frame_indices,
                "track_face_feats": ft.embedding,
                "track_body_feats": None,
                "track_back_feats": None,
            }
            for ft in face_only
        ]
        raw_clusters = cluster_tracks(
            track_inputs,
            tau_tight_face=self.tau_tight_face,
        )

        idx_to_track = {i: ft for i, ft in enumerate(face_only)}
        rep_tracks: List[FaceTrack] = []

        for cl in raw_clusters:
            members = cl.get("members", [])
            member_tracks = [idx_to_track[i] for i in members if i in idx_to_track]
            if not member_tracks:
                continue

            embeddings = [
                np.asarray(ft.embedding, dtype=np.float32).ravel()
                for ft in member_tracks
            ]
            mean_emb = np.stack(embeddings, axis=0).mean(axis=0)
            norm = float(np.linalg.norm(mean_emb))
            mean_emb = mean_emb / max(norm, 1e-12)

            first = member_tracks[0]
            frame_set: set = set()
            for ft in member_tracks:
                frame_set.update(ft.frame_indices)

            rep_tracks.append(FaceTrack(
                track_id=first.track_id,
                shot_id=first.shot_id,
                embedding=mean_emb,
                bbox=first.bbox,
                frame_indices=sorted(frame_set),
                has_face=True,
                member_track_ids=[ft.track_id for ft in member_tracks],
            ))

        return rep_tracks

    def _read_faces(
        self,
        shot_id: str,
        person_tracks: dict,
        shot_seg: dict,
        steps: List[ShotStep],
    ) -> Tuple[List[FaceTrack], List[str], List[str]]:
        """Read and filter face tracks for a shot. Returns (tracks, all_ids, discarded_ids)."""
        sid = str(shot_id)

        read_params = {"shot_id": sid, "person_tracks": person_tracks, "shot_seg": shot_seg}
        all_face_tracks: List[FaceTrack] = read_face(**read_params)
        log_step(steps, "read_face", read_params,
                 f"{len(all_face_tracks)} face track(s) read")

        if not all_face_tracks:
            return [], [], []

        all_track_ids = [ft.track_id for ft in all_face_tracks]
        discarded_loose: List[str] = []

        # Loose filter before clustering
        if self.vlm_face_filter and self.vlm_url and self.video_file:
            before_ids = set(ft.track_id for ft in all_face_tracks)
            all_face_tracks = self._filter_unusable_tracks(all_face_tracks, person_tracks.get("face", {}), sid)
            after_ids = set(ft.track_id for ft in all_face_tracks)
            discarded_loose = sorted(before_ids - after_ids)
        return all_face_tracks, all_track_ids, discarded_loose

    def _write_cluster_timelines(self, shot_seg: dict) -> None:
        """Write one JSON timeline file per cluster to self.timeline_dir.

        Each file records when the cluster first appears and how its name
        evolves across the movie (replaying evidence_log chronologically).
        """
        import os
        os.makedirs(self.timeline_dir, exist_ok=True)

        def _shot_start_time(sid: str) -> float:
            info = shot_seg.get(sid, shot_seg.get(int(sid) if sid.isdigit() else sid, {}))
            tr = info.get("time_range", [0.0, 0.0])
            return float(tr[0])

        def _shot_time_range(sid: str):
            info = shot_seg.get(sid, shot_seg.get(int(sid) if sid.isdigit() else sid, {}))
            return info.get("time_range", [None, None])

        entries = [e for e in self.bank.all_entries() if e.shots_seen]

        # Per-cluster timeline shot list + evidence grouped by shot
        entry_shots: Dict[int, List[str]] = {}
        entry_evidence: Dict[int, Dict[str, list]] = {}
        all_shots: set = set()
        for entry in entries:
            all_shot_ids = list(dict.fromkeys(entry.shots_seen))
            for ev in entry.evidence_log:
                if ev.shot_id not in all_shot_ids:
                    all_shot_ids.append(ev.shot_id)
            entry_shots[entry.cluster_id] = sorted(all_shot_ids, key=_shot_start_time)
            ev_by_shot: Dict[str, list] = defaultdict(list)
            for ev in entry.evidence_log:
                ev_by_shot[ev.shot_id].append(ev)
            entry_evidence[entry.cluster_id] = ev_by_shot
            all_shots.update(all_shot_ids)

        sorted_all_shots = sorted(all_shots, key=_shot_start_time)

        # Replay chronologically with per-shot collision resolution
        cum_votes: Dict[int, Dict[str, float]] = {e.cluster_id: {} for e in entries}
        ev_count: Dict[int, int] = {e.cluster_id: 0 for e in entries}
        per_shot_names: Dict[int, Dict[str, Optional[str]]] = {e.cluster_id: {} for e in entries}
        cluster_own_shots: Dict[int, set] = {e.cluster_id: set(entry_shots[e.cluster_id]) for e in entries}

        for shot_id in sorted_all_shots:
            for entry in entries:
                cid = entry.cluster_id
                for ev in entry_evidence[cid].get(shot_id, []):
                    cum_votes[cid][ev.name] = cum_votes[cid].get(ev.name, 0.0) + ev.confidence
                    ev_count[cid] += 1
            state = {
                e.cluster_id: (cum_votes[e.cluster_id], ev_count[e.cluster_id])
                for e in entries
            }
            resolved = self._resolve_collisions(state)
            for entry in entries:
                cid = entry.cluster_id
                if shot_id in cluster_own_shots[cid]:
                    name, _ = resolved[cid]
                    per_shot_names[cid][shot_id] = name

        for entry in entries:
            cid = entry.cluster_id
            sorted_shots = entry_shots[cid]
            timeline: list = []
            prev_name: Optional[str] = None
            for shot_id in sorted_shots:
                time_range = _shot_time_range(shot_id)
                name = per_shot_names[cid].get(shot_id)
                timeline.append({
                    "shot_id": shot_id,
                    "time_start": time_range[0],
                    "time_end": time_range[1],
                    "name": name,
                    "name_changed": name != prev_name,
                })
                prev_name = name

            # First visual occurrence: first shot where the face was detected
            visual_shots = sorted(
                entry.shots_seen, key=_shot_start_time
            ) if entry.shots_seen else []
            if visual_shots:
                vs = visual_shots[0]
                vr = _shot_time_range(vs)
                first_visual = {"shot_id": vs, "time_start": vr[0], "time_end": vr[1]}
            else:
                first_visual = None

            record = {
                "cluster_id": cid,
                "final_name": entry.name,
                "first_appearance": {
                    "shot_id": sorted_shots[0],
                    "time_start": timeline[0]["time_start"],
                    "time_end": timeline[0]["time_end"],
                },
                "first_visual_appearance": first_visual,
                "timeline": timeline,
            }

            fname = f"cluster_{cid}.json"
            with open(
                os.path.join(self.timeline_dir, fname), "w", encoding="utf-8"
            ) as fh:
                _json.dump(record, fh, indent=2, ensure_ascii=False)

        # Save cluster face features for downstream evaluation
        import pickle as _pickle
        feat_map = {}
        for entry in self.bank.all_entries():
            if entry.face_feat is not None:
                feat_map[entry.cluster_id] = np.asarray(
                    entry.face_feat, dtype=np.float32
                )
        feat_path = os.path.join(self.timeline_dir, "cluster_features.pkl")
        with open(feat_path, "wb") as fh:
            _pickle.dump(feat_map, fh, protocol=_pickle.HIGHEST_PROTOCOL)

        if self.verbose:
            n = len(list(self.bank.all_entries()))
            print(f"[Timeline] Wrote {n} cluster timeline(s) + "
                  f"features ({len(feat_map)}) → {self.timeline_dir}")

    def _bank_snapshot(self) -> dict:
        """Return a lightweight snapshot of the current bank state."""
        entries = []
        for e in self.bank.all_entries():
            entries.append({
                "cluster_id": e.cluster_id,
                "name": e.name,
                "confidence": round(e.confidence, 4),
                "track_count": len(e.track_ids),
                "shots_seen": e.shots_seen,
                "aliases": e.aliases,
            })
        secondary = [
            {
                "names": sb["names"],
                "track_ids": sb["track_ids"],
                "rep_track_id": sb["rep_track_id"],
                "shot_id": sb["shot_id"],
            }
            for sb in self.secondary_bank
        ]
        return {
            "total_clusters": len(entries),
            "named_clusters": sum(1 for e in entries if e["name"] is not None),
            "entries": entries,
            "secondary_bank": secondary,
        }

    def run_shot(
        self,
        shot_id: str,
        person_tracks: dict,
        shot_seg: dict,
        sid_to_subtitles: dict,
        bank_path: str,
    ) -> ShotTrajectory:
        """Execute one shot (possibly extended window). Returns ShotTrajectory."""
        sid = str(shot_id)
        steps: List[ShotStep] = []

        # Step 1: extract names via spaCy NER
        unique_mentions: Dict[str, NameMention] = {}
        query_to_canonical: Dict[str, str] = {}

        read_params = {"shot_id": sid, "sid_to_subtitles": sid_to_subtitles, "lang": self.lang}
        name_mentions: List[NameMention] = read_name(**read_params)
        log_step(steps, "read_name", read_params, f"{len(name_mentions)} name mention(s)")

        for mention in (name_mentions or []):
            key = mention.name.lower()
            self._name_mention_counts[key] += 1
            if key not in unique_mentions:
                unique_mentions[key] = mention
                query_to_canonical[key] = mention.name

        # Extend window when names detected and VLM available
        if self.vlm_linker_url and self.video_file and unique_mentions:
            (ext_start, ext_end), window_shot_ids = self._get_extended_window(
                sid, shot_seg, n_before=1, n_after=2,
                max_after_gap=2.0, sid_to_subtitles=sid_to_subtitles,
            )

            # Detect names in extra shots
            extra_sids = [str(s) for s in window_shot_ids if str(s) != sid]
            for w_sid in extra_sids:
                read_params = {"shot_id": w_sid, "sid_to_subtitles": sid_to_subtitles, "lang": self.lang}
                name_mentions = read_name(**read_params)
                log_step(steps, "read_name", read_params, f"{len(name_mentions)} name mention(s)")

                for mention in (name_mentions or []):
                    key = mention.name.lower()
                    self._name_mention_counts[key] += 1
                    if key not in unique_mentions:
                        unique_mentions[key] = mention
                        query_to_canonical[key] = mention.name
        else:
            window_shot_ids = [sid]

        if unique_mentions:
            log_step(steps, "extract_names", {},
                     f"{len(unique_mentions)} name(s) across {len(window_shot_ids)} shot(s)")

        use_vlm = bool(self.vlm_linker_url and self.video_file and unique_mentions)

        # Step 2: read faces from window, cluster (bank writes deferred)
        all_face_tracks: List[FaceTrack] = []
        traj_all_track_ids: List[str] = []
        traj_discarded_loose: List[str] = []
        for w_sid in window_shot_ids:
            w_sid = str(w_sid)
            tracks, read_ids, disc_ids = self._read_faces(
                w_sid, person_tracks, shot_seg, steps,
            )
            all_face_tracks.extend(tracks)
            traj_all_track_ids.extend(read_ids)
            traj_discarded_loose.extend(disc_ids)
            # Mark neighbours as processed
            if w_sid != sid:
                self._processed_shots.add(w_sid)

        # Cluster across window
        cluster_reps = self._cluster_shot_faces(all_face_tracks)

        if not cluster_reps:
            self.bank.save(bank_path)
            return ShotTrajectory(
                sid, steps,
                {
                    "shot_id": sid, "cluster_ids": [], "names_written": [],
                    "trajectory": {
                        "face_tracks_read": traj_all_track_ids,
                        "face_tracks_discarded_loose": traj_discarded_loose,
                        "face_tracks_discarded_strict": [],
                        "clusters": [],
                        "names_detected": sorted(unique_mentions.keys()),
                        "name_linkings": [],
                        "bank_snapshot": self._bank_snapshot(),
                    },
                },
                "no_faces",
            )

        # Assign colors to cluster reps for VLM
        rep_idx_to_color: Dict[int, str] = {}
        track_color_map: Dict[str, str] = {}
        for idx, rep in enumerate(cluster_reps):
            color = _COLOR_NAMES[idx % len(_COLOR_NAMES)]
            rep_idx_to_color[idx] = color
            for tid in (rep.member_track_ids or [rep.track_id]):
                track_color_map[tid] = color

        if self.verbose:
            print(f"[Shot {sid}] {len(cluster_reps)} face cluster(s) across "
                  f"{len(window_shot_ids)} shot(s)")

        # Step 3: VLM name-face linking
        name_to_rep_idx: Dict[str, int] = {}
        names_written: List[str] = []
        # Subtitles with speaker labels (used for voice bank)
        labeled_subs: List[dict] = []

        if use_vlm:
            ext_subs: List[dict] = []
            for eid in window_shot_ids:
                ext_subs.extend(sid_to_subtitles.get(str(eid), []))

            # Filter to on-screen names
            onscreen_names = self._llm_filter_onscreen_names(list(unique_mentions.keys()), ext_subs)

            # Prune to on-screen names only
            onscreen_lower = {n.lower() for n in onscreen_names}
            unique_mentions = {k: v for k, v in unique_mentions.items() if k in onscreen_lower}

            if not onscreen_names:
                use_vlm = False

            if use_vlm and self.use_diarization and self.video_file and ext_subs:
                from speaker_labeler import assign_diar_to_subtitles, run_diarization
                local_turns = run_diarization(self.video_file, self.diar_model, start=ext_start, end=ext_end)
                if local_turns:
                    assign_diar_to_subtitles(ext_subs, local_turns)
                    n_tightened = self._tighten_subs_with_diar(ext_subs, local_turns)
                    if self.verbose:
                        print(f"  [diarization] {len(local_turns)} turns for window "
                              f"[{ext_start:.1f}s, {ext_end:.1f}s] "
                              f"({len({t[2] for t in local_turns})} speakers); "
                              f"tightened {n_tightened} sub(s)")
                elif self.verbose:
                    print(f"  [diarization] no turns for window "
                          f"[{ext_start:.1f}s, {ext_end:.1f}s]")

            if self.score_dir and self.imdbid and ext_subs:
                from sync_scorer import assign_speaker_labels, load_track_scores
                tid_score_dict = load_track_scores(
                    self.score_dir, self.imdbid,
                    [str(s) for s in window_shot_ids],
                )
                if tid_score_dict:
                    assign_speaker_labels(
                        subtitles=ext_subs,
                        track_color_map=track_color_map,
                        tid_score_dict=tid_score_dict,
                        sync_threshold=self.sync_threshold,
                        overwrite_threshold=self.overwrite_threshold,
                        diar_threshold=self.diar_threshold,
                    )
                    if self.verbose:
                        labeled = sum(
                            1 for s in ext_subs
                            if s.get("speaker") and s["speaker"] != "none"
                        )
                        print(f"  [sync-score] {labeled}/{len(ext_subs)} subs labeled "
                              f"with speaker color")

            # Capture labeled subs for voice bank
            labeled_subs = ext_subs

        if use_vlm:
            name_to_color = run_vlm_name_linking(
                video_path=self.video_file,
                candidate_names=onscreen_names,
                track_color_map=track_color_map,
                subtitles=ext_subs,
                start_offset=ext_start,
                end_offset=ext_end,
                vlm_url=self.vlm_linker_url,
                model_id=self.vlm_linker_model_id,
                sample_fps=self.vlm_sample_fps,
                max_new_tokens=self.vlm_max_tokens,
                track_detections=person_tracks.get("face", {}),
                vlm_frames_vis_dir=self.vlm_frames_vis_dir or None,
                vlm_frames_vis_tag=str(sid),
            )

            if name_to_color:
                color_to_rep_idx = {c: i for i, c in rep_idx_to_color.items()}
                for name, color in name_to_color.items():
                    idx = color_to_rep_idx.get(color)
                    if idx is not None:
                        name_to_rep_idx[name] = idx

        # Step 4: strict quality filter + bank write
        named_rep_idxs = set(name_to_rep_idx.values())
        face_dets = person_tracks.get("face", {})

        # track_id → original FaceTrack
        tid_to_face_track = {ft.track_id: ft for ft in all_face_tracks}

        # Strict filter
        can_strict_filter = (
            self.vlm_face_filter and self.vlm_url and self.video_file
        )
        if can_strict_filter:
            strict_result = self._filter_bank_ready(cluster_reps, face_dets, sid)
            strict_passing_tids = {ft.track_id for ft in strict_result}
            strict_passing = {idx for idx, rep in enumerate(cluster_reps) if rep.track_id in strict_passing_tids}
        else:
            # No VLM filter — treat all as passing
            strict_passing = set(range(len(cluster_reps)))

        all_assigned_cluster_ids: List[int] = []
        cluster_to_color: Dict[int, str] = {}
        traj_discarded_strict: List[str] = [cluster_reps[i].track_id for i in range(len(cluster_reps)) if i not in strict_passing]

        # Pre-extract voice embeddings for secondary matching
        rep_idx_voice_feat: Dict[int, np.ndarray] = {}
        if self.video_file and labeled_subs and rep_idx_to_color:
            color_to_rep_idx = {c: i for i, c in rep_idx_to_color.items()}
            has_labeled = any(
                s.get("speaker") and s["speaker"] != "none"
                for s in labeled_subs
            )
            if has_labeled:
                from voice_bank import extract_voice_embeddings, average_normalized_embedding
                color_to_rep = {c: i for i, c in rep_idx_to_color.items()}
                pre_voice_embs = extract_voice_embeddings(video_path=self.video_file, subtitles=labeled_subs, color_to_cluster=color_to_rep)
                for ridx, embs in pre_voice_embs.items():
                    avg = average_normalized_embedding(embs)
                    if avg is not None:
                        rep_idx_voice_feat[ridx] = avg

        for idx, rep in enumerate(cluster_reps):
            is_named = idx in named_rep_idxs
            bank_rep = None

            if idx in strict_passing:
                bank_rep = rep
            elif is_named and can_strict_filter:
                # Named but not bank-ready — try promoting a member track
                member_tids = rep.member_track_ids or [rep.track_id]
                member_tracks = [
                    tid_to_face_track[tid]
                    for tid in member_tids
                    if tid in tid_to_face_track
                ]
                if member_tracks:
                    promoted = self._filter_bank_ready(member_tracks, face_dets, sid)
                    if promoted:
                        bank_rep = promoted[0]
                        if self.verbose:
                            print(f"  [promote] named cluster rep "
                                  f"{rep.track_id} not bank-ready, "
                                  f"promoted member {bank_rep.track_id}")

            if bank_rep is None:
                if is_named:
                    # Collect the linked names for this cluster rep
                    linked_names = [
                        query_to_canonical.get(n.lower(), n)
                        for n, ridx in name_to_rep_idx.items()
                        if ridx == idx
                    ]

                    # Check face similarity against existing bank clusters
                    incoming = np.asarray(
                        rep.embedding, dtype=np.float32
                    ).ravel()
                    from actions import _cos_dist
                    matches: List[Tuple[int, float]] = []
                    for entry in self.bank.all_entries():
                        if entry.face_feat is None:
                            continue
                        d = _cos_dist(incoming, np.asarray(
                            entry.face_feat, dtype=np.float32
                        ))
                        if d <= self.bank.tau_face:
                            matches.append((entry.cluster_id, d))

                    if len(matches) == 1:
                        # Exactly one face match — merge into that cluster
                        best_bank_cid, best_bank_dist = matches[0]

                        # Update face embedding, track IDs, and shots_seen
                        # (mirroring what write_face does on an existing cluster)
                        best_entry = self.bank.get(best_bank_cid)
                        self.raw_feats_store.setdefault(best_bank_cid, [])
                        self.bank.update_face_feat(
                            best_bank_cid, incoming,
                            self.raw_feats_store[best_bank_cid],
                        )
                        self.raw_feats_store[best_bank_cid].append(incoming)
                        member_tids = rep.member_track_ids or [rep.track_id]
                        for tid in member_tids:
                            if tid not in best_entry.track_ids:
                                best_entry.track_ids.append(tid)
                        if sid not in best_entry.shots_seen:
                            best_entry.shots_seen.append(sid)

                        for name, linked_idx in name_to_rep_idx.items():
                            if linked_idx != idx:
                                continue
                            canonical_name = query_to_canonical.get(
                                name.lower(), name
                            )
                            conf = min(0.80, CONFIDENCE_CAP)
                            write_params = {
                                "name": canonical_name,
                                "face_cluster_id": best_bank_cid,
                                "shot_id": sid,
                                "confidence": conf,
                                "bank": self.bank,
                                "reason": "vlm_bank_match",
                            }
                            write_name(**write_params)
                            log_step(steps, "write_name", write_params,
                                     f"wrote '{canonical_name}' → existing bank "
                                     f"cluster {best_bank_cid} (dist={best_bank_dist:.3f})")
                            names_written.append(canonical_name)
                            if self.verbose:
                                print(
                                    f"  [bank-match] named track "
                                    f"{rep.track_id} not bank-ready, "
                                    f"wrote '{canonical_name}' → existing "
                                    f"cluster {best_bank_cid} "
                                    f"(dist={best_bank_dist:.3f})"
                                )
                    elif len(matches) == 0 and any(
                        not self.bank.query(name=n) for n in linked_names
                    ):
                        # No face match and at least one linked name has never
                        # been saved — save the track as a new cluster despite
                        # failing strict filtering so the name is recorded.
                        bank_rep = rep
                        if self.verbose:
                            unseen = [
                                n for n in linked_names
                                if not self.bank.query(name=n)
                            ]
                            print(f"  [force-save] named cluster rep "
                                  f"{rep.track_id} not bank-ready, but "
                                  f"unseen name(s) {unseen} → saving")
                    else:
                        # Multiple ambiguous face matches, or all linked names
                        # already in bank — secondary bank
                        member_tids = rep.member_track_ids or [rep.track_id]
                        self.secondary_bank.append({
                            "names": linked_names,
                            "track_ids": list(member_tids),
                            "rep_track_id": rep.track_id,
                            "embedding": rep.embedding,
                            "shot_id": sid,
                        })
                        if self.verbose:
                            reason = (
                                f"{len(matches)} bank matches (ambiguous)"
                                if len(matches) > 1 else "no bank match"
                            )
                            print(f"  [secondary] named cluster rep "
                                  f"{rep.track_id} not bank-ready, "
                                  f"{reason} → secondary bank "
                                  f"(names={linked_names})")
                # Unnamed and not bank-ready — skip
                if bank_rep is None:
                    continue

            write_face_params = {
                "face_track": bank_rep,
                "shot_id": sid,
                "bank": self.bank,
                "raw_feats_store": self.raw_feats_store,
                "voice_feat": rep_idx_voice_feat.get(idx),
            }
            cid: int = write_face(**write_face_params)
            log_step(steps, "write_face", write_face_params, f"cluster_id={cid}")
            all_assigned_cluster_ids.append(cid)
            cluster_to_color[cid] = rep_idx_to_color[idx]

            # Write linked names
            for name, linked_idx in name_to_rep_idx.items():
                if linked_idx != idx:
                    continue
                canonical_name = query_to_canonical.get(name.lower(), name)
                conf = min(0.80, CONFIDENCE_CAP)

                write_params = {
                    "name": canonical_name,
                    "face_cluster_id": cid,
                    "shot_id": sid,
                    "confidence": conf,
                    "bank": self.bank,
                    "reason": "vlm",
                }
                write_name(**write_params)
                log_step(steps, "write_name", write_params,
                         f"wrote '{canonical_name}' → cluster {cid} (conf={conf:.3f})")

                names_written.append(canonical_name)
                if self.verbose:
                    canonical_note = (
                        f" (canonical of '{name}')"
                        if canonical_name != name else ""
                    )
                    print(
                        f"[Shot {sid}] VLM write_name '{canonical_name}' "
                        f"→ cluster {cid} (conf={conf:.3f}){canonical_note}"
                    )

        if self.verbose and all_assigned_cluster_ids:
            print(f"[Shot {sid}] {len(all_assigned_cluster_ids)} cluster(s) "
                  f"written to bank → IDs {all_assigned_cluster_ids}")

        # Voice bank: update features
        if rep_idx_voice_feat and cluster_to_color:
            from voice_bank import extract_voice_embeddings
            # Re-extract per-subtitle embeddings by bank cluster_id
            voice_color_to_cid = {col: cid for cid, col in cluster_to_color.items()}
            if self.video_file and labeled_subs and voice_color_to_cid:
                cluster_voice_embs = extract_voice_embeddings(
                    video_path=self.video_file,
                    subtitles=labeled_subs,
                    color_to_cluster=voice_color_to_cid,
                )
                for cid, embs in cluster_voice_embs.items():
                    if not embs:
                        continue
                    self.bank.update_voice_feat(
                        cluster_id=cid,
                        new_embeddings=embs,
                        raw_voice_feats=self.voice_feats_store[cid],
                    )
                    self.voice_feats_store[cid].extend(embs)
                # Log subtitle time ranges for voice samples
                for sub in labeled_subs:
                    color = sub.get("speaker")
                    if not color or color == "none":
                        continue
                    cid = voice_color_to_cid.get(color)
                    if cid is None:
                        continue
                    s = float(sub.get("start", 0.0))
                    e = float(sub.get("end", 0.0))
                    if e - s >= 0.5:
                        self.voice_sample_log[cid].append((s, e))

                if self.verbose:
                    n_embs = sum(len(v) for v in cluster_voice_embs.values())
                    print(f"[Shot {sid}] voice bank: {n_embs} embedding(s) "
                          f"for {len(cluster_voice_embs)} cluster(s)")

        # Merge phonetic variants + resolve cross-cluster name collisions
        self._calibrate_names_metaphone()
        self._resolve_name_collisions()
        self.bank.save(bank_path)

        # Build trajectory metadata
        color_to_bank_cid = {col: cid for cid, col in cluster_to_color.items()}
        traj_clusters = [
            {
                "rep_track_id": rep.track_id,
                "member_track_ids": rep.member_track_ids or [rep.track_id],
                "color": rep_idx_to_color.get(idx, ""),
                "bank_cluster_id": color_to_bank_cid.get(rep_idx_to_color.get(idx)),
                "written_to_bank": idx in strict_passing or idx in named_rep_idxs,
            }
            for idx, rep in enumerate(cluster_reps)
        ]
        traj_name_linkings = [
            {
                "name": name,
                "canonical": query_to_canonical.get(name.lower(), name),
                "color": rep_idx_to_color.get(rep_idx, ""),
                "bank_cluster_id": color_to_bank_cid.get(rep_idx_to_color.get(rep_idx)),
                "confidence": 0.80,
            }
            for name, rep_idx in name_to_rep_idx.items()
        ]

        return ShotTrajectory(
            sid,
            steps,
            {
                "shot_id": sid,
                "cluster_ids": list(dict.fromkeys(all_assigned_cluster_ids)),
                "names_written": names_written,
                "trajectory": {
                    "face_tracks_read": traj_all_track_ids,
                    "face_tracks_discarded_loose": traj_discarded_loose,
                    "face_tracks_discarded_strict": traj_discarded_strict,
                    "clusters": traj_clusters,
                    "names_detected": sorted(unique_mentions.keys()),
                    "name_linkings": traj_name_linkings,
                    "bank_snapshot": self._bank_snapshot(),
                },
            },
            "stop",
        )

    def run(
        self,
        shot_seg: dict,
        person_tracks: dict,
        sid_to_subtitles: dict,
        bank_path: str,
        trajectory_log_path: str = "",
    ) -> dict:
        """Process all shots and build the character bank. Returns summary dict."""
        self.reset(bank_path)

        shot_order = sorted(
            shot_seg.keys(),
            key=lambda x: (int(x) if str(x).isdigit() else x),
        )
        shot_summaries = []
        trajectory_log: List[dict] = []

        for shot_count, sid in tqdm(enumerate(shot_order, 1), total=len(shot_order), desc="Processing shots"):
            sid_str = str(sid)
            already_processed = sid_str in self._processed_shots
            if already_processed:
                if self.verbose:
                    print(f"[Shot {sid_str}] already processed in extended window — skipping")
            else:
                trajectory = self.run_shot(
                    shot_id=sid_str,
                    person_tracks=person_tracks,
                    shot_seg=shot_seg,
                    sid_to_subtitles=sid_to_subtitles,
                    bank_path=bank_path,
                )
                shot_summaries.append(trajectory.result)

                # Collect trajectory
                if trajectory_log_path:
                    shot_info = shot_seg.get(sid, shot_seg.get(sid_str, {}))
                    time_range = shot_info.get("time_range", [0.0, 0.0])
                    subs = sid_to_subtitles.get(sid_str, [])
                    traj_entry = {
                        "shot_id": sid_str,
                        "shot_index": shot_count,
                        "time_range": time_range,
                        "subtitles": [
                            {"start": s["start"], "end": s["end"],
                             "text": s.get("text", "")}
                            for s in subs
                        ],
                        "steps": [s.to_log_dict() for s in trajectory.steps],
                        "finish_reason": trajectory.finish_reason,
                        **(trajectory.result.get("trajectory", {})),
                        "cluster_ids": trajectory.result.get("cluster_ids", []),
                        "names_written": trajectory.result.get("names_written", []),
                    }
                    trajectory_log.append(traj_entry)

                # Periodic trajectory save
                if trajectory_log_path and shot_count % 50 == 0:
                    import os as _os
                    import pickle as _pickle
                    _os.makedirs(_os.path.dirname(_os.path.abspath(trajectory_log_path)),
                                 exist_ok=True)
                    with open(trajectory_log_path, "wb") as fh:
                        _pickle.dump(trajectory_log, fh, protocol=_pickle.HIGHEST_PROTOCOL)
                    print(f"[Traj] Saved trajectory log ({len(trajectory_log)} shots) "
                          f"at shot {shot_count}")

            if (self.vis_every_k > 0
                    and self.vis_output_dir
                    and shot_count % self.vis_every_k == 0):
                from bank_visualizer import visualize_bank
                out = visualize_bank(
                    bank=self.bank,
                    person_tracks=person_tracks,
                    video_path=self.video_file,
                    shot_idx=shot_count,
                    output_dir=self.vis_output_dir,
                    secondary_bank=self.secondary_bank,
                    voice_sample_log=self.voice_sample_log or None,
                )
                if out and self.verbose:
                    print(f"[Vis] Bank snapshot → {out}")

        # Prune noisy name variants + resolve cross-cluster collisions
        self._calibrate_names_metaphone()
        self._resolve_name_collisions()

        # Flush unresolved clusters
        unresolved: List[int] = flush_unresolved(
            bank=self.bank, min_shots_seen=self.flush_min_shots,
        )

        if self.verbose:
            total = len(list(self.bank.all_entries()))
            print(f"\n[Done] {total} character clusters total.")
            print(f"[Done] {len(unresolved)} unresolved cluster(s) "
                  f"with >={self.flush_min_shots} shots: {unresolved}")

        self.bank.save(bank_path)

        # Final bank snapshot
        if self.vis_output_dir:
            from bank_visualizer import visualize_bank
            out = visualize_bank(
                bank=self.bank,
                person_tracks=person_tracks,
                video_path=self.video_file,
                shot_idx=len(shot_order),
                output_dir=self.vis_output_dir,
                secondary_bank=self.secondary_bank,
                voice_sample_log=self.voice_sample_log or None,
            )
            if out and self.verbose:
                print(f"[Vis] Final bank snapshot → {out}")

        # Write per-cluster timeline files
        if self.timeline_dir:
            self._write_cluster_timelines(shot_seg)

        # Save trajectory log
        if trajectory_log_path:
            import os as _os
            import pickle as _pickle
            _os.makedirs(_os.path.dirname(_os.path.abspath(trajectory_log_path)),
                         exist_ok=True)
            with open(trajectory_log_path, "wb") as fh:
                _pickle.dump(trajectory_log, fh, protocol=_pickle.HIGHEST_PROTOCOL)
            print(f"[DONE] Trajectory log ({len(trajectory_log)} shots) → "
                  f"{trajectory_log_path}")

        # Save name filter log
        if self.name_filter_log:
            import os as _os
            filter_log_path = bank_path.rsplit(".", 1)[0] + "_name_filter_log.json"
            _os.makedirs(_os.path.dirname(_os.path.abspath(filter_log_path)),
                         exist_ok=True)
            with open(filter_log_path, "w", encoding="utf-8") as fh:
                _json.dump(self.name_filter_log, fh, indent=2, ensure_ascii=False)
            if self.verbose:
                print(f"[DONE] Name filter log ({len(self.name_filter_log)} entries) → "
                      f"{filter_log_path}")

        return {
            "shot_summaries": shot_summaries,
            "unresolved_cluster_ids": unresolved,
            "total_clusters": len(list(self.bank.all_entries())),
            "named_clusters": sum(1 for e in self.bank.all_entries() if e.name is not None),
        }
