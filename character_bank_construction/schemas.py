"""Typed data structures for the character bank pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class FaceTrack:
    """A single person track within a shot."""
    track_id: str
    shot_id: str
    embedding: Optional[object]          # np.ndarray | None
    bbox: Optional[List[float]]
    frame_indices: List[int]
    has_face: bool = True
    member_track_ids: Optional[List[str]] = None


@dataclass
class NameMention:
    """A PERSON name extracted from subtitle text aligned to a shot."""
    name: str
    shot_id: str
    subtitle_start: float
    subtitle_end: float
    text: str
    is_onscreen: bool = True
    speaker: Optional[str] = None
    diar_speaker: Optional[str] = None


@dataclass
class EvidenceEntry:
    """One auditable record of a name↔cluster association."""
    shot_id: str
    name: str
    confidence: float
    reason: str
    logged_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "shot_id": self.shot_id,
            "name": self.name,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "logged_at": self.logged_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EvidenceEntry":
        return cls(
            shot_id=d["shot_id"],
            name=d["name"],
            confidence=d["confidence"],
            reason=d["reason"],
            logged_at=d.get("logged_at", 0.0),
        )


@dataclass
class CharacterEntry:
    """Persistent record for one character cluster in the bank."""
    cluster_id: int
    name: Optional[str]
    confidence: float
    track_ids: List[str]
    face_feat: Optional[object]  # np.ndarray or None
    voice_feat: Optional[object] = None  # np.ndarray or None
    shots_seen: List[str] = field(default_factory=list)
    name_votes: Dict[str, float] = field(default_factory=dict)
    aliases: List[str] = field(default_factory=list)
    evidence_log: List[EvidenceEntry] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "name": self.name,
            "confidence": round(self.confidence, 4),
            "track_ids": sorted(self.track_ids),
            "face_feat": self.face_feat,
            "voice_feat": self.voice_feat,
            "shots_seen": self.shots_seen,
            "name_votes": {k: round(v, 4) for k, v in self.name_votes.items()},
            "aliases": self.aliases,
            "evidence_log": [e.to_dict() for e in self.evidence_log],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CharacterEntry":
        return cls(
            cluster_id=d["cluster_id"],
            name=d.get("name"),
            confidence=d.get("confidence", 0.0),
            track_ids=list(d.get("track_ids", [])),
            face_feat=d.get("face_feat"),
            voice_feat=d.get("voice_feat"),
            shots_seen=list(d.get("shots_seen", [])),
            name_votes=dict(d.get("name_votes", {})),
            aliases=list(d.get("aliases", [])),
            evidence_log=[EvidenceEntry.from_dict(e) for e in d.get("evidence_log", [])],
        )

    def update_canonical(self, min_similar_votes: int = 1) -> None:
        """Recompute name and confidence from name_votes.

        Requires *min_similar_votes* evidence entries with a token overlap
        to the best-scoring name before attributing.
        """
        if not self.name_votes:
            self.name = None
            self.confidence = 0.0
            return
        best_name = max(self.name_votes, key=self.name_votes.__getitem__)
        best_score = self.name_votes[best_name]
        total = sum(self.name_votes.values()) or 1.0

        # Count evidence entries with a name similar to the best candidate.
        best_tokens = set(best_name.lower().split())
        similar_count = sum(
            1 for e in self.evidence_log
            if set(e.name.lower().split()) & best_tokens
        )
        if similar_count < min_similar_votes:
            self.name = None
            self.confidence = 0.0
            self.aliases = []
            return

        self.name = best_name
        support = min(len(self.evidence_log) / 3, 1.0)
        self.confidence = min(best_score / total, 1.0) * support
        self.aliases = [n for n in self.name_votes if n != best_name]
