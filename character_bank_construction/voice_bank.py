"""
Voice bank: extract and accumulate ECAPA-TDNN speech embeddings per cluster.

Uses sync-score speaker labels (colors) for speaker-to-cluster mapping.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Dict, List, Optional

import numpy as np
import torch
import torchaudio

# Module-level cache (loaded once per process)
_CLASSIFIER = None


def _get_classifier():
    """Load (or return cached) SpeechBrain ECAPA-TDNN classifier."""
    global _CLASSIFIER
    if _CLASSIFIER is None:
        from speechbrain.inference.speaker import EncoderClassifier
        _CLASSIFIER = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
        )
    return _CLASSIFIER


def _crop_audio(
    video_path: str,
    out_wav: str,
    start: float,
    end: float,
) -> bool:
    """Extract a mono 16 kHz WAV clip. Returns True on success."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-to", str(end),
        "-i", video_path,
        "-vn",
        "-ar", "16000",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        out_wav,
    ]
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return r.returncode == 0


def _extract_embedding(wav_path: str) -> Optional[np.ndarray]:
    """Extract ECAPA-TDNN embedding from a WAV file, or None on failure."""
    try:
        signal, fs = torchaudio.load(wav_path)
        if signal.shape[1] == 0:
            return None
        classifier = _get_classifier()
        embedding = classifier.encode_batch(signal)  # shape: (1, 1, D)
        return embedding[0][0].detach().cpu().numpy().astype(np.float32)
    except Exception:
        return None


def average_normalized_embedding(embeddings: List[np.ndarray]) -> Optional[np.ndarray]:
    """L2-normalise each embedding, average, then L2-normalise the result."""
    if not embeddings:
        return None
    normalized = []
    for emb in embeddings:
        norm = np.linalg.norm(emb)
        normalized.append(emb / norm if norm != 0 else emb)
    avg = np.mean(normalized, axis=0)
    final_norm = np.linalg.norm(avg)
    if final_norm != 0:
        avg = avg / final_norm
    return avg.astype(np.float32)


def extract_voice_embeddings(
    video_path: str,
    subtitles: List[dict],
    color_to_cluster: Dict[str, int],
    min_duration: float = 0.5,
) -> Dict[int, List[np.ndarray]]:
    """Extract speech embeddings per cluster using sync-score speaker labels.

    Returns {cluster_id: [embedding_array, ...]}.
    """
    if not subtitles or not color_to_cluster:
        return {}
    if not os.path.isfile(video_path):
        return {}

    cluster_embeddings: Dict[int, List[np.ndarray]] = {}
    tmp_dir = tempfile.mkdtemp(prefix="voice_bank_")

    try:
        for idx, sub in enumerate(subtitles):
            color = sub.get("speaker")
            if not color or color == "none":
                continue
            cid = color_to_cluster.get(color)
            if cid is None:
                continue

            start = float(sub.get("start", 0.0))
            end = float(sub.get("end", 0.0))
            if end - start < min_duration:
                continue

            wav_path = os.path.join(tmp_dir, f"sub_{idx}.wav")
            if not _crop_audio(video_path, wav_path, start, end):
                continue

            emb = _extract_embedding(wav_path)
            if emb is not None:
                cluster_embeddings.setdefault(cid, []).append(emb)

            try:
                os.remove(wav_path)
            except OSError:
                pass
    finally:
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass

    return cluster_embeddings


MAX_VOICE_SAMPLES_PER_CLUSTER: int = 3


def save_voice_samples(
    video_path: str,
    subtitles: List[dict],
    color_to_cluster: Dict[str, int],
    output_dir: str,
    max_per_cluster: int = MAX_VOICE_SAMPLES_PER_CLUSTER,
    min_duration: float = 0.5,
) -> Dict[int, List[str]]:
    """Save representative voice WAV clips per cluster (longest segments first).

    Returns {cluster_id: [wav_path, ...]}.
    """
    if not subtitles or not color_to_cluster:
        return {}
    if not os.path.isfile(video_path):
        return {}

    # Group by cluster, sorted by duration
    cluster_subs: Dict[int, List[dict]] = {}
    for sub in subtitles:
        color = sub.get("speaker")
        if not color or color == "none":
            continue
        cid = color_to_cluster.get(color)
        if cid is None:
            continue
        start = float(sub.get("start", 0.0))
        end = float(sub.get("end", 0.0))
        if end - start < min_duration:
            continue
        cluster_subs.setdefault(cid, []).append(sub)

    # Take top-k by duration
    for cid in cluster_subs:
        cluster_subs[cid].sort(
            key=lambda s: float(s.get("end", 0)) - float(s.get("start", 0)),
            reverse=True,
        )
        cluster_subs[cid] = cluster_subs[cid][:max_per_cluster]

    saved: Dict[int, List[str]] = {}
    for cid, subs in cluster_subs.items():
        cid_dir = os.path.join(output_dir, f"cluster_{cid}")
        os.makedirs(cid_dir, exist_ok=True)
        for i, sub in enumerate(subs):
            wav_path = os.path.join(cid_dir, f"sample_{i}.wav")
            start = float(sub["start"])
            end = float(sub["end"])
            if _crop_audio(video_path, wav_path, start, end):
                saved.setdefault(cid, []).append(wav_path)

    return saved
