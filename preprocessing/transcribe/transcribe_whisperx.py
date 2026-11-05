"""Transcribe + align + diarize audio files with WhisperX.

For each WAV under --audio_folder:
    1. Transcribe with whisper large-v3,
    2. Force-align word timestamps via wav2vec2,
    3. Run pyannote diarization and assign speaker labels per word,
and write the per-segment results to --save_dir as JSON.

The HF access token for the diarization pipeline must be provided via the
HUGGINGFACE_TOKEN environment variable (or --hf_token).
"""

import argparse
import gc
import json
import os

import torch
import whisperx
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio_folder", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--language", type=str, default="en",
                        help='Whisper language code, e.g. "en" or "zh".')
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--compute_type", type=str, default="float16")
    parser.add_argument("--model", type=str, default="large-v3")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--hf_token", type=str,
                        default=os.environ.get("HUGGINGFACE_TOKEN", ""),
                        help="Hugging Face token for pyannote diarization. "
                             "Defaults to $HUGGINGFACE_TOKEN.")
    return parser.parse_args()


def transcribe_one(audio_path, args, asr_model, align_model, align_meta,
                   diarize_model):
    audio = whisperx.load_audio(audio_path)
    result = asr_model.transcribe(audio, batch_size=args.batch_size,
                                  language=args.language)
    result_aligned = whisperx.align(
        result["segments"], align_model, align_meta, audio, args.device,
        return_char_alignments=False,
    )
    diarize_segments = diarize_model(audio)
    result_diarize = whisperx.assign_word_speakers(diarize_segments, result_aligned)

    return [
        {
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"],
            "speaker": seg.get("speaker"),
        }
        for seg in result_diarize["segments"]
    ]


def main():
    args = parse_args()
    if not args.hf_token:
        raise SystemExit("Hugging Face token required for diarization. "
                         "Pass --hf_token or set HUGGINGFACE_TOKEN.")
    os.makedirs(args.save_dir, exist_ok=True)

    asr_model = whisperx.load_model(
        args.model, args.device, compute_type=args.compute_type,
        language=args.language,
    )
    align_model, align_meta = whisperx.load_align_model(
        language_code=args.language, device=args.device,
    )
    diarize_model = whisperx.diarize.DiarizationPipeline(
        use_auth_token=args.hf_token, device=args.device,
    )

    for filename in tqdm(sorted(os.listdir(args.audio_folder))):
        if not filename.lower().endswith(".wav"):
            continue
        audio_path = os.path.join(args.audio_folder, filename)
        save_path = os.path.join(args.save_dir,
                                 os.path.splitext(filename)[0] + ".json")
        if os.path.exists(save_path):
            continue

        results = transcribe_one(audio_path, args, asr_model, align_model,
                                 align_meta, diarize_model)
        with open(save_path, "w") as outfile:
            json.dump(results, outfile, indent=4)

        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
