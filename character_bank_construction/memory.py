"""
CharacterBank — persistent, pickle-backed store for character clusters.

Provides storage, serialization, and lookup for CharacterEntry objects.
Reads/writes go through action functions in actions.py.
"""

from __future__ import annotations

import os
import pickle
import tempfile
from typing import Dict, Iterable, List, Optional, Union

import numpy as np

from schemas import CharacterEntry


class CharacterBank:
    """Central storage mapping cluster_id → CharacterEntry."""

    def __init__(self, tau_face: float = 0.55, tau_voice: float = 0.25) -> None:
        self.tau_face = tau_face
        self.tau_voice = tau_voice
        self._store: Dict[int, CharacterEntry] = {}
        self._next_id: int = 0

    @classmethod
    def load(cls, path: str, tau_face: float = 0.55, tau_voice: float = 0.25) -> "CharacterBank":
        """Load from *path*, or return an empty bank if file doesn't exist."""
        bank = cls(tau_face=tau_face, tau_voice=tau_voice)
        if not os.path.isfile(path):
            return bank

        with open(path, "rb") as fh:
            data = pickle.load(fh)

        bank.tau_face = data.get("tau_face", tau_face)
        bank.tau_voice = data.get("tau_voice", tau_voice)
        bank._next_id = data.get("next_id", 0)
        bank._store = data.get("store", {})

        return bank

    def save(self, path: str) -> None:
        """Atomically write the bank to *path* as pickle."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        payload = {
            "tau_face": self.tau_face,
            "tau_voice": self.tau_voice,
            "next_id": self._next_id,
            "store": self._store,
        }

        dir_ = os.path.dirname(os.path.abspath(path)) or "."
        fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    @property
    def next_id(self) -> int:
        """Return the next cluster_id without consuming it."""
        return self._next_id

    def _consume_id(self) -> int:
        cid = self._next_id
        self._next_id += 1
        return cid

    def add(self, entry: CharacterEntry) -> None:
        """Insert *entry* into the bank. Raises if cluster_id already exists."""
        if entry.cluster_id in self._store:
            raise ValueError(f"Cluster {entry.cluster_id} already in bank")
        self._store[entry.cluster_id] = entry
        if entry.cluster_id >= self._next_id:
            self._next_id = entry.cluster_id + 1

    def get(self, cluster_id: int) -> Optional[CharacterEntry]:
        """Return the entry for *cluster_id*, or None if absent."""
        return self._store.get(cluster_id)

    def remove(self, cluster_id: int) -> Optional[CharacterEntry]:
        """Delete and return the entry for *cluster_id*, or None if absent."""
        return self._store.pop(cluster_id, None)

    def all_entries(self) -> Iterable[CharacterEntry]:
        """Iterate over all CharacterEntry objects in insertion order."""
        return self._store.values()

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, cluster_id: int) -> bool:
        return cluster_id in self._store

    def query(
        self,
        name: Optional[str] = None,
        face_cluster_id: Optional[int] = None,
    ) -> Union[Optional[CharacterEntry], List[CharacterEntry]]:
        """Lookup by cluster_id (returns entry or None), by name (returns list),
        or with no args (returns all entries).
        """
        if face_cluster_id is not None:
            return self._store.get(face_cluster_id)

        if name is not None:
            needle = name.strip().lower()
            results = []
            for entry in self._store.values():
                if entry.name and entry.name.lower() == needle:
                    results.append(entry)
                    continue
                if any(k.lower() == needle for k in entry.name_votes):
                    results.append(entry)
            return results

        return list(self._store.values())

    def face_feat_as_array(self, cluster_id: int) -> Optional[np.ndarray]:
        """Return the mean face feature as a numpy array, or None."""
        entry = self._store.get(cluster_id)
        if entry is None or entry.face_feat is None:
            return None
        return np.asarray(entry.face_feat, dtype=np.float32)

    def update_face_feat(
        self,
        cluster_id: int,
        new_feat: np.ndarray,
        existing_feats: List[np.ndarray],
    ) -> None:
        """Recompute the running-mean face feature for a cluster."""
        entry = self._store.get(cluster_id)
        if entry is None:
            return
        all_feats = existing_feats + [np.asarray(new_feat, dtype=np.float32).ravel()]
        stacked = np.stack(all_feats, axis=0)
        mean = stacked.mean(axis=0)
        norm = float(np.linalg.norm(mean))
        mean = mean / max(norm, 1e-12)
        entry.face_feat = mean

    def voice_feat_as_array(self, cluster_id: int) -> Optional[np.ndarray]:
        """Return the mean voice feature as a numpy array, or None."""
        entry = self._store.get(cluster_id)
        if entry is None or entry.voice_feat is None:
            return None
        return np.asarray(entry.voice_feat, dtype=np.float32)

    def update_voice_feat(
        self,
        cluster_id: int,
        new_embeddings: List[np.ndarray],
        raw_voice_feats: List[np.ndarray],
    ) -> None:
        """Recompute the running-mean voice feature (L2-normalised) for a cluster."""
        entry = self._store.get(cluster_id)
        if entry is None:
            return
        all_feats = list(raw_voice_feats) + [
            np.asarray(e, dtype=np.float32).ravel() for e in new_embeddings
        ]
        if not all_feats:
            return
        # L2-normalise → average → L2-normalise
        normalized = []
        for f in all_feats:
            norm = float(np.linalg.norm(f))
            normalized.append(f / max(norm, 1e-12))
        stacked = np.stack(normalized, axis=0)
        mean = stacked.mean(axis=0)
        norm = float(np.linalg.norm(mean))
        mean = mean / max(norm, 1e-12)
        entry.voice_feat = mean

    def create_cluster(
        self,
        face_feat: np.ndarray,
        track_ids: Optional[List[str]] = None,
        shot_id: Optional[str] = None,
    ) -> CharacterEntry:
        """Allocate a new cluster and insert it into the bank."""
        cid = self._consume_id()
        feat_arr = np.asarray(face_feat, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(feat_arr))
        feat_arr = feat_arr / max(norm, 1e-12)

        entry = CharacterEntry(
            cluster_id=cid,
            name=None,
            confidence=0.0,
            track_ids=list(track_ids or []),
            face_feat=feat_arr,
            shots_seen=[shot_id] if shot_id else [],
            name_votes={},
            aliases=[],
            evidence_log=[],
        )
        self._store[cid] = entry
        return entry
