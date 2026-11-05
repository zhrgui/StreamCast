"""
Speaker diarization and subtitle labeling for the streaming pipeline.

run_diarization: extract audio → DiariZen → sorted (start, end, speaker) turns.
assign_diar_to_subtitles: label subtitles in-place with diar_speaker / diar_iou.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import wave
from typing import List, Optional, Tuple

PAD_TARGET_SECONDS = 20.0

# Module-level cache (loaded once per process)
_DIAR_PIPELINE = None
_DIAR_MODEL_NAME: Optional[str] = None


def _extract_audio(
    video_path: str,
    out_wav: str,
    start: float = 0.0,
    end: Optional[float] = None,
) -> None:
    """Extract mono 16 kHz audio from *video_path* to *out_wav*."""
    cmd = ["ffmpeg", "-y"]
    if start > 0.0:
        cmd += ["-ss", str(start)]
    cmd += ["-i", video_path]
    if end is not None and end > start:
        cmd += ["-t", str(end - start)]
    cmd += ["-vn", "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", out_wav]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise RuntimeError(
            f"ffmpeg audio extraction failed: {r.stderr.decode('utf-8', errors='ignore')}"
        )


def _pad_wav(wav_path: str, target_seconds: float = PAD_TARGET_SECONDS) -> None:
    """Pad a 16-bit mono WAV with silence to reach at least *target_seconds*."""
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        audio_data = wf.readframes(n_frames)
    if n_frames / sr >= target_seconds:
        return
    pad_frames = int((target_seconds - n_frames / sr) * sr)
    silence = b"\x00" * (pad_frames * n_channels * sampwidth)
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sr)
        wf.writeframes(audio_data + silence)


def _get_diar_pipeline(model_name: str):
    """Load or return cached DiariZenPipeline."""
    global _DIAR_PIPELINE, _DIAR_MODEL_NAME
    if _DIAR_PIPELINE is None or _DIAR_MODEL_NAME != model_name:
        import functools, torch
        _orig_load = torch.load
        torch.load = functools.partial(_orig_load, weights_only=False)
        try:
            from diarizen.pipelines.inference import DiariZenPipeline
            _DIAR_PIPELINE = DiariZenPipeline.from_pretrained(model_name)
        finally:
            torch.load = _orig_load
        _DIAR_MODEL_NAME = model_name
    return _DIAR_PIPELINE


def run_diarization(
    video_path: str,
    model_name: str = "BUT-FIT/diarizen-wavlm-large-s80-md-v2",
    start: float = 0.0,
    end: Optional[float] = None,
) -> List[Tuple[float, float, str]]:
    """Run DiariZen diarization. Returns sorted (start, end, speaker) in absolute time."""
    if not os.path.isfile(video_path):
        return []

    try:
        pipeline = _get_diar_pipeline(model_name)
    except ImportError:
        print("[diarization] DiariZen not installed — skipping")
        return []
    except Exception as exc:
        print(f"[diarization] Failed to load model: {exc}")
        return []

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = tmp.name

    try:
        _extract_audio(video_path, tmp_wav, start=start, end=end)
        _pad_wav(tmp_wav)
        result = pipeline(tmp_wav)
    except Exception as exc:
        print(f"[diarization] Failed: {exc}")
        return []
    finally:
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)

    # Offset to absolute video time
    turns: List[Tuple[float, float, str]] = []
    for turn, _, speaker in result.itertracks(yield_label=True):
        turns.append((float(turn.start) + start, float(turn.end) + start, str(speaker)))
    turns.sort(key=lambda x: x[0])
    return turns


def _temporal_iou(s1: float, e1: float, s2: float, e2: float) -> float:
    inter = max(0.0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    return 0.0 if union <= 0 else inter / union


def assign_diar_to_subtitles(
    subtitles: List[dict],
    turns: List[Tuple[float, float, str]],
) -> None:
    """Label each subtitle in-place with diar_speaker and diar_iou (best IoU match)."""
    for sub in subtitles:
        sub_start = float(sub.get("start", 0.0))
        sub_end = float(sub.get("end", 0.0))
        best_speaker = "none"
        best_iou = 0.0
        for t_start, t_end, spk in turns:
            # Early exit: turns are sorted, so once start is well past sub_end we're done
            if t_start > sub_end + 1.0:
                break
            iou = _temporal_iou(sub_start, sub_end, t_start, t_end)
            if iou > best_iou:
                best_iou = iou
                best_speaker = spk
        sub["diar_speaker"] = best_speaker
        sub["diar_iou"] = round(best_iou, 4)
