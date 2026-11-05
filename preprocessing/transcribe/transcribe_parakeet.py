"""Transcribe a folder of WAV files with Nvidia Parakeet TDT.

Writes one JSON per input audio file containing
    [{"start": float, "end": float, "text": str}, ...]
under --save_dir, skipping files that already have an output. If the model OOMs
on a long file we recursively split it in half, transcribe each half, and merge
the timestamps.
"""

import argparse
import json
import os
import tempfile

import nemo.collections.asr as nemo_asr
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm


def transcribe_audio(asr_model, audio_file):
    output = asr_model.transcribe([audio_file], timestamps=True)
    segments = output[0].timestamp["segment"]
    return [
        {
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["segment"],
        }
        for seg in segments
    ]


def transcribe_audio_split(asr_model, audio_file):
    """Split the audio in half, transcribe each part, then offset timestamps."""
    data, sr = sf.read(audio_file)
    mid = len(data) // 2
    duration_first = mid / sr

    tmpdir = tempfile.mkdtemp()
    first_path = os.path.join(tmpdir, "part1.wav")
    second_path = os.path.join(tmpdir, "part2.wav")
    try:
        sf.write(first_path, data[:mid], sr)
        sf.write(second_path, data[mid:], sr)
        first = transcribe_audio(asr_model, first_path)
        second = transcribe_audio(asr_model, second_path)
    finally:
        for p in (first_path, second_path):
            if os.path.exists(p):
                os.remove(p)
        os.rmdir(tmpdir)

    for seg in second:
        seg["start"] = round(seg["start"] + duration_first, 2)
        seg["end"] = round(seg["end"] + duration_first, 2)
    return first + second


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio_folder", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str,
                        default="nvidia/parakeet-tdt-0.6b-v3")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=args.model_name)
    # Local relative attention keeps memory bounded on long-form audio.
    asr_model.change_attention_model(
        self_attention_model="rel_pos_local_attn",
        att_context_size=[256, 256],
    )

    for filename in tqdm(sorted(os.listdir(args.audio_folder))):
        if not filename.lower().endswith(".wav"):
            continue
        audio_file = os.path.join(args.audio_folder, filename)
        save_path = os.path.join(args.save_dir, os.path.splitext(filename)[0] + ".json")
        if os.path.exists(save_path):
            continue

        try:
            results = transcribe_audio(asr_model, audio_file)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            msg = str(e)
            if "CUDA" in msg or "out of memory" in msg.lower():
                print(f"[WARN] OOM on {filename}; splitting and retrying.")
                torch.cuda.empty_cache()
                try:
                    results = transcribe_audio_split(asr_model, audio_file)
                except Exception as e2:
                    print(f"[ERROR] Split also failed for {filename}: {e2}")
                    continue
            else:
                print(f"[ERROR] {filename}: {e}")
                continue

        with open(save_path, "w") as outfile:
            json.dump(results, outfile, indent=4)


if __name__ == "__main__":
    main()
