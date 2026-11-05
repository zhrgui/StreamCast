"""
Sync-score-based speaker label assignment for the streaming pipeline.

Assigns "speaker" color labels to subtitles using per-track audio-visual
sync scores. Three-phase: accumulate → pick best color per speaker → label.
"""

from __future__ import annotations

import os
import pickle
from collections import defaultdict
from typing import Dict, List, Optional


def load_track_scores(
    score_dir: str,
    imdbid: str,
    shot_ids: List[str],
) -> Dict[str, dict]:
    """Load sync scores for tracks in *shot_ids*. Returns {track_id: {"scores": [...]}}."""
    if not score_dir or not imdbid:
        return {}
    tid_score_dict: Dict[str, dict] = {}
    for sid in shot_ids:
        score_file = os.path.join(score_dir, imdbid, f"{sid}.pkl")
        if not os.path.exists(score_file):
            continue
        try:
            with open(score_file, "rb") as fh:
                shot_scores = pickle.load(fh)
        except Exception:
            continue
        for sid, obj_scores in shot_scores.items():
            for obj_id, data in obj_scores.items():
                tid_score_dict[f"{sid}_{obj_id}"] = data
    return tid_score_dict


def _scores_overlap(scores: list, entry_start: Optional[float], entry_end: Optional[float]) -> bool:
    if not scores:
        return False
    if entry_start is None or entry_end is None:
        return False
    first_t = scores[0].get("t")
    last_t = scores[-1].get("t")
    if first_t is None or last_t is None:
        return False
    if first_t > entry_end or last_t < entry_start:
        return False
    for item in scores:
        t = item.get("t")
        if t is None:
            continue
        if t < entry_start:
            continue
        if t > entry_end:
            break
        return True
    return False


def _mean_score_in_window(
    scores: list, entry_start: Optional[float], entry_end: Optional[float]
) -> Optional[float]:
    if not scores:
        return None
    if entry_start is None or entry_end is None:
        return None
    vals = []
    for item in scores:
        t = item.get("t")
        if t is None:
            continue
        if t < entry_start:
            continue
        if t > entry_end:
            break
        s = item.get("score")
        if s is None:
            continue
        vals.append(float(s))
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _overlap_duration(
    scores: list, entry_start: Optional[float], entry_end: Optional[float]
) -> float:
    """Duration of track timestamps inside the subtitle window."""
    ts = [
        item["t"] for item in scores
        if item.get("t") is not None
        and entry_start <= item["t"] <= entry_end
    ]
    if len(ts) < 2:
        return 0.0
    return max(ts) - min(ts)


def _color_scores_for_entry(
    tid_score_dict: dict,
    color_map: Dict[str, str],
    entry_start: Optional[float],
    entry_end: Optional[float],
    min_overlap: float = 0.15,
) -> Dict[str, float]:
    """Compute best mean sync score per color for a subtitle window."""
    color_best: Dict[str, float] = {}
    for track_id, track_data in tid_score_dict.items():
        if track_id not in color_map:
            continue
        scores = track_data.get("scores", [])
        if not _scores_overlap(scores, entry_start, entry_end):
            continue
        if _overlap_duration(scores, entry_start, entry_end) < min_overlap:
            continue
        mean_score = _mean_score_in_window(scores, entry_start, entry_end)
        if mean_score is None:
            continue
        color = color_map[track_id]
        if color not in color_best or mean_score > color_best[color]:
            color_best[color] = mean_score
    return color_best


def assign_speaker_labels(
    subtitles: List[dict],
    track_color_map: Dict[str, str],
    tid_score_dict: Dict[str, dict],
    sync_threshold: float = -0.075,
    overwrite_threshold: float = 0.1,
    diar_threshold: float = 0.05,
) -> None:
    """Assign "speaker" color labels to subtitles in-place (three-phase sync-score logic)."""
    # Phase 1
    speaker_color_scores: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    subtitle_to_color_scores: List[Dict[str, float]] = []

    for entry in subtitles:
        diar_speaker = entry.get("diar_speaker")
        entry_start = entry.get("start")
        entry_end = entry.get("end")
        color_scores = _color_scores_for_entry(
            tid_score_dict, track_color_map, entry_start, entry_end
        )
        for color, score in color_scores.items():
            speaker_color_scores[diar_speaker][color].append(score)
        subtitle_to_color_scores.append(color_scores)

    # Phase 2
    speaker_to_color: Dict[str, str] = {}
    for spk, color_dict in speaker_color_scores.items():
        best_color: Optional[str] = None
        best_max = float("-inf")
        for color, score_list in color_dict.items():
            max_val = max(score_list)
            if max_val > best_max:
                best_max = max_val
                best_color = color
        if best_color is not None and best_max >= sync_threshold:
            speaker_to_color[spk] = best_color

    # Phase 3
    for entry, color_scores in zip(subtitles, subtitle_to_color_scores):
        diar_speaker = entry.get("diar_speaker")
        diar_iou = entry.get("diar_iou")

        if color_scores:
            max_color_label = max(color_scores, key=color_scores.__getitem__)
            max_color_score = color_scores[max_color_label]
        else:
            max_color_label = "none"
            max_color_score = 0.0

        if max_color_score >= overwrite_threshold:
            entry["speaker"] = max_color_label
            continue

        if diar_iou is None or diar_iou < diar_threshold:
            entry["speaker"] = "none"
        else:
            entry["speaker"] = speaker_to_color.get(diar_speaker, "none")
