"""
Agent actions for the character bank pipeline.

Read actions are side-effect-free; write/merge actions mutate the bank.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np

from schemas import CharacterEntry, EvidenceEntry, FaceTrack, NameMention
from memory import CharacterBank


def _l2(x: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(x))
    return x / max(n, 1e-12)


def _cos_dist(a: np.ndarray, b: np.ndarray) -> float:
    a = _l2(np.asarray(a, dtype=np.float32).ravel())
    b = _l2(np.asarray(b, dtype=np.float32).ravel())
    return float(1.0 - np.dot(a, b))


def _mean_feat(feats: List[np.ndarray]) -> Optional[np.ndarray]:
    if not feats:
        return None
    stacked = np.stack([np.asarray(f, dtype=np.float32).ravel() for f in feats], axis=0)
    mean = stacked.mean(axis=0)
    return _l2(mean)


def read_face(
    shot_id: str,
    person_tracks: dict,
    shot_seg: dict,
) -> List[FaceTrack]:
    """Read face tracks for a shot. No side effects.

    Returns FaceTrack objects for each face track in the shot.
    Tracks without face features have has_face=False.
    """
    sid = str(shot_id)

    face_data = person_tracks.get("face", {})
    face_feats = person_tracks.get("face_feats", {})

    result: List[FaceTrack] = []

    for tid in face_data:
        if not (isinstance(tid, str) and "_" in tid and tid.split("_")[0] == sid):
            continue

        # Build embedding from face_feats
        raw_feat = face_feats.get(tid)
        embedding = None
        has_face = False

        if raw_feat is not None:
            arr = np.asarray(raw_feat, dtype=np.float32)
            if arr.ndim == 1:
                embedding = _l2(arr)
            elif arr.ndim == 2 and arr.shape[0] > 0:
                embedding = _l2(arr.mean(axis=0))
            has_face = embedding is not None

        # Bounding box from last detection
        bbox = None
        detections = face_data[tid]
        if len(detections) > 0:
            last = detections[-1]
            if len(last) >= 5:
                bbox = [float(last[1]), float(last[2]),
                        float(last[3]), float(last[4])]

        # Frame indices
        frame_set = set()
        for det in detections:
            if len(det) >= 1:
                frame_set.add(int(det[0]))

        result.append(FaceTrack(
            track_id=tid,
            shot_id=sid,
            embedding=embedding,
            bbox=bbox,
            frame_indices=sorted(frame_set),
            has_face=has_face,
        ))

    return result


def read_name(
    shot_id: str,
    sid_to_subtitles: dict,
    lang: str = "en",
) -> List[NameMention]:
    """Extract PERSON names from subtitles aligned to *shot_id*. No side effects.

    Returns de-duplicated NameMention objects per subtitle segment.
    """
    from name_detection import detect_names

    sid = str(shot_id)
    subtitles = sid_to_subtitles.get(sid, [])
    result: List[NameMention] = []

    for sub in subtitles:
        text = sub.get("text", "")
        start = float(sub.get("start", 0.0))
        end = float(sub.get("end", 0.0))

        # Skip stage directions / narration
        stripped = text.strip()
        is_onscreen = not (stripped.startswith("[") or stripped.startswith("("))

        raw_speaker = sub.get("speaker")
        speaker = raw_speaker if (raw_speaker and raw_speaker != "none") else None
        raw_diar = sub.get("diar_speaker")
        diar_speaker = raw_diar if (raw_diar and raw_diar != "none") else None

        names = detect_names(text, lang=lang)
        seen = set()
        for name in names:
            if name.lower() not in seen:
                seen.add(name.lower())
                result.append(NameMention(
                    name=name,
                    shot_id=sid,
                    subtitle_start=start,
                    subtitle_end=end,
                    text=text,
                    is_onscreen=is_onscreen,
                    speaker=speaker,
                    diar_speaker=diar_speaker,
                ))

    return result


def write_face(
    face_track: FaceTrack,
    shot_id: str,
    bank: CharacterBank,
    raw_feats_store: Dict[int, List[np.ndarray]],
    voice_feat: Optional[np.ndarray] = None,
) -> int:
    """Add a face track to the bank, returning the assigned cluster_id.

    Matching: face cosine distance (primary) then voice (secondary).
    Face wins on conflicts. Creates a new cluster if neither matches.
    """
    if not face_track.has_face or face_track.embedding is None:
        raise ValueError(
            f"write_face called with faceless track {face_track.track_id}"
        )

    incoming = _l2(np.asarray(face_track.embedding, dtype=np.float32).ravel())

    # --- Face matching ---
    best_face_id: Optional[int] = None
    best_face_dist = float("inf")

    for entry in bank.all_entries():
        if entry.face_feat is None:
            continue
        stored = np.asarray(entry.face_feat, dtype=np.float32)
        d = _cos_dist(incoming, stored)
        if d <= bank.tau_face and d < best_face_dist:
            best_face_dist = d
            best_face_id = entry.cluster_id

    # --- Voice matching (secondary signal) ---
    best_voice_id: Optional[int] = None
    best_voice_dist = float("inf")

    if voice_feat is not None:
        incoming_voice = _l2(np.asarray(voice_feat, dtype=np.float32).ravel())
        for entry in bank.all_entries():
            if entry.voice_feat is None:
                continue
            stored_voice = np.asarray(entry.voice_feat, dtype=np.float32)
            d = _cos_dist(incoming_voice, stored_voice)
            if d <= bank.tau_voice and d < best_voice_dist:
                best_voice_dist = d
                best_voice_id = entry.cluster_id

    # Face wins on conflict
    best_id = best_face_id
    if best_id is None:
        best_id = best_voice_id

    if best_id is not None:
        # Assign to existing cluster
        entry = bank.get(best_id)
        raw_feats_store.setdefault(best_id, [])
        bank.update_face_feat(best_id, incoming, raw_feats_store[best_id])
        raw_feats_store[best_id].append(incoming)

        if face_track.track_id not in entry.track_ids:
            entry.track_ids.append(face_track.track_id)
        if shot_id not in entry.shots_seen:
            entry.shots_seen.append(shot_id)

        return best_id

    # Create a brand-new cluster
    member_ids = face_track.member_track_ids or [face_track.track_id]
    entry = bank.create_cluster(
        face_feat=incoming,
        track_ids=member_ids,
        shot_id=shot_id,
    )
    raw_feats_store[entry.cluster_id] = [incoming]
    return entry.cluster_id


def write_name(
    name: str,
    face_cluster_id: int,
    shot_id: str,
    confidence: float,
    bank: CharacterBank,
    reason: str = "heuristic_1to1",
) -> None:
    """Bind *name* to *face_cluster_id* in the bank.

    Accumulates name votes; repeated co-occurrence raises confidence.
    Recomputes the canonical name after each write.
    """
    entry = bank.get(face_cluster_id)
    if entry is None:
        raise KeyError(f"Cluster {face_cluster_id} not found in bank")

    # Accumulate vote
    entry.name_votes[name] = entry.name_votes.get(name, 0.0) + confidence

    entry.evidence_log.append(EvidenceEntry(
        shot_id=shot_id,
        name=name,
        confidence=confidence,
        reason=reason,
    ))

    entry.update_canonical()


def merge_cluster(
    cluster_id_a: int,
    cluster_id_b: int,
    reason: str,
    bank: CharacterBank,
) -> int:
    """Merge two clusters representing the same person.

    Keeps the cluster with higher confidence. Unions track IDs, evidence,
    and name votes. Returns the surviving cluster_id.
    """
    if cluster_id_a == cluster_id_b:
        raise ValueError("Cannot merge a cluster with itself")

    a = bank.get(cluster_id_a)
    b = bank.get(cluster_id_b)
    if a is None:
        raise KeyError(f"Cluster {cluster_id_a} not found")
    if b is None:
        raise KeyError(f"Cluster {cluster_id_b} not found")

    # Keep higher-confidence cluster
    total_a = sum(a.name_votes.values())
    total_b = sum(b.name_votes.values())
    keep, drop = (a, b) if total_a >= total_b else (b, a)

    for tid in drop.track_ids:
        if tid not in keep.track_ids:
            keep.track_ids.append(tid)

    seen = set(keep.shots_seen)
    for sid in drop.shots_seen:
        if sid not in seen:
            keep.shots_seen.append(sid)
            seen.add(sid)

    for name, conf in drop.name_votes.items():
        keep.name_votes[name] = keep.name_votes.get(name, 0.0) + conf

    keep.evidence_log.extend(drop.evidence_log)
    keep.evidence_log.append(EvidenceEntry(
        shot_id=drop.shots_seen[-1] if drop.shots_seen else "unknown",
        name=drop.name or "(unnamed)",
        confidence=max(drop.confidence, 0.0),
        reason=f"merge:{reason}",
    ))

    # Recompute face centroid
    feat_a = np.asarray(keep.face_feat, dtype=np.float32) if keep.face_feat else None
    feat_b = np.asarray(drop.face_feat, dtype=np.float32) if drop.face_feat else None
    if feat_a is not None and feat_b is not None:
        merged_feat = _l2((feat_a + feat_b) / 2.0)
        keep.face_feat = merged_feat.tolist()
    elif feat_b is not None:
        keep.face_feat = drop.face_feat

    keep.update_canonical()
    bank.remove(drop.cluster_id)

    return keep.cluster_id


def query_bank(
    bank: CharacterBank,
    name: Optional[str] = None,
    face_cluster_id: Optional[int] = None,
) -> Union[Optional[CharacterEntry], List[CharacterEntry]]:
    """Lookup the bank by name or cluster ID. No side effects."""
    return bank.query(name=name, face_cluster_id=face_cluster_id)


def flush_unresolved(
    bank: CharacterBank,
    min_shots_seen: int = 3,
) -> List[int]:
    """Return persistent unnamed clusters (>= *min_shots_seen* shots)."""
    unresolved = []
    for entry in bank.all_entries():
        if entry.name is not None:
            continue
        if len(entry.shots_seen) >= min_shots_seen:
            unresolved.append(entry.cluster_id)
    return sorted(unresolved)
