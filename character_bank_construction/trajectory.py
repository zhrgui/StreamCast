"""Trajectory record for the character-bank agent.

A trajectory is the per-shot log of (function, params, outcome) triples that
gets pickled to disk for offline analysis. The pickled shape is preserved
across the registry → direct-call refactor: every step still serializes as
``{step_id, action: {function, parameters}, observation}``.
"""

from dataclasses import dataclass
from typing import Any, Dict, List


# Parameter keys that are safe (small, JSON-friendly) to keep in the trace.
_KEEP_KEYS = {
    "shot_id", "name", "face_cluster_id", "confidence", "reason",
    "lang", "cluster_id_a", "cluster_id_b", "min_shots_seen",
}

# Parameters omitted from the human-readable observation string because their
# repr is huge or noisy (large arrays, dicts, bank object, ...).
_LARGE_PARAMS = {
    "person_tracks", "shot_seg", "bank", "raw_feats_store",
    "sid_to_subtitles", "face_track",
}


def _trim_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Strip heavy objects from action params for the persisted trace."""
    light: Dict[str, Any] = {}
    for k, v in params.items():
        if k in _KEEP_KEYS:
            light[k] = v
        elif k == "face_track" and hasattr(v, "track_id"):
            light["track_id"] = v.track_id
            light["member_track_ids"] = v.member_track_ids
    return light


def _format_observation(function: str, params: Dict[str, Any], outcome: Any) -> str:
    visible = ", ".join(
        f"{k}={v!r}" for k, v in params.items() if k not in _LARGE_PARAMS
    )
    return f"[{function}({visible})] → {outcome}"


@dataclass
class ShotStep:
    step_id: int
    function: str
    params: Dict[str, Any]   # already trimmed via _trim_params
    outcome: str             # already formatted via _format_observation

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "action": {"function": self.function, "parameters": self.params},
            "observation": self.outcome,
        }


@dataclass
class ShotTrajectory:
    shot_id: str
    steps: List[ShotStep]
    result: Dict[str, Any]
    finish_reason: str


def log_step(
    steps: List[ShotStep],
    function: str,
    params: Dict[str, Any],
    outcome: Any,
) -> None:
    """Append a step to *steps* with trimmed params and a stringified outcome."""
    steps.append(ShotStep(
        step_id=len(steps) + 1,
        function=function,
        params=_trim_params(params),
        outcome=_format_observation(function, params, outcome),
    ))
