"""Adaptive shot-boundary detection.

Runs PySceneDetect's AdaptiveDetector on every video in --video_dir and writes
one record per video to --jsonl_path (used for resume) plus an aggregated JSON
dump at --save_file.

Each record is keyed by the video stem (treated here as an IMDb id) and lists
the frame range and time range of every detected shot.
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

from decord import VideoReader, cpu
from scenedetect import detect, AdaptiveDetector
from tqdm import tqdm

# Cap intra-op threading so each worker stays single-threaded.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def process_one(video_dir, video_file):
    idx = os.path.splitext(video_file)[0]
    video_path = os.path.join(video_dir, video_file)

    vr = VideoReader(video_path, ctx=cpu(0))
    scene_list = detect(video_path, AdaptiveDetector())

    episode_shot_metadata = {}
    for i, (start_time, end_time) in enumerate(scene_list):
        start_frame = start_time.get_frames()
        end_frame = min(end_time.get_frames() - 1, len(vr) - 1)
        episode_shot_metadata[i] = {
            "frame_range": (start_frame, end_frame),
            "time_range": (start_time.get_seconds(), end_time.get_seconds()),
        }
    return idx, episode_shot_metadata


def load_progress(jsonl_path):
    """Return (processed_ids, shot_metadata) from a resume jsonl file."""
    processed_ids = set()
    shot_metadata = {}
    if not os.path.exists(jsonl_path):
        return processed_ids, shot_metadata

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            imdbid = item.get("imdbid")
            ep_meta = item.get("shot_metadata")
            if imdbid and ep_meta is not None:
                processed_ids.add(imdbid)
                shot_metadata[imdbid] = ep_meta
    return processed_ids, shot_metadata


def build_shot_metadata(video_dir, video_list, jsonl_path, max_workers=None):
    shot_metadata = {}
    worker = partial(process_one, video_dir)

    with open(jsonl_path, "a", encoding="utf-8") as jsonl_f, \
            ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(worker, vf) for vf in video_list]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            try:
                idx, ep_meta = fut.result()
            except Exception as e:
                print(f"[WARN] Worker failed: {e}")
                continue
            shot_metadata[idx] = ep_meta
            jsonl_f.write(
                json.dumps({"imdbid": idx, "shot_metadata": ep_meta},
                           ensure_ascii=False) + "\n"
            )
            jsonl_f.flush()
    return shot_metadata


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Folder containing input videos.")
    parser.add_argument("--save_file", type=str, required=True,
                        help="Final aggregated JSON output.")
    parser.add_argument("--jsonl_path", type=str, required=True,
                        help="Per-video progress JSONL (also used to resume).")
    parser.add_argument("--max_workers", type=int, default=8,
                        help="Parallel worker processes.")
    return parser.parse_args()


def main():
    args = parse_args()

    video_list = sorted(os.listdir(args.video_dir))
    processed_ids, shot_metadata = load_progress(args.jsonl_path)
    pending = [vf for vf in video_list
               if os.path.splitext(vf)[0] not in processed_ids]

    new_meta = build_shot_metadata(
        args.video_dir, pending, args.jsonl_path, max_workers=args.max_workers,
    )
    shot_metadata.update(new_meta)

    os.makedirs(os.path.dirname(args.save_file), exist_ok=True)
    with open(args.save_file, "w") as outfile:
        json.dump(shot_metadata, outfile, indent=4)


if __name__ == "__main__":
    main()
