"""Aggregate per-shot face tracks into a single per-corpus pickle.

Reads every pickle under --face_track_dir/<imdbid>/<shot_id>.pkl whose
(imdbid, shot_id) appears in --sid_file, and emits a nested dict:

    {imdbid: {"face": {f"{shot_id}_{obj_id}": np.ndarray, ...}}}

The output pickle is what face_feature_vpcd_det_cuda.py and the ASD inference
expect.
"""

import argparse
import json
import os
import pickle
from collections import defaultdict
from glob import glob

import numpy as np
from tqdm import tqdm


def _tid_sort_key(key):
    if isinstance(key, str):
        parts = key.split("_", 1)
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            return (int(parts[0]), int(parts[1]))
    return (float("inf"), str(key))


def to_dict(d):
    """Recursively convert defaultdicts to dicts with deterministic key order."""
    if isinstance(d, defaultdict):
        d = dict(d)
    if isinstance(d, dict):
        items = sorted(d.items(), key=lambda kv: _tid_sort_key(kv[0]))
        return {k: to_dict(v) for k, v in items}
    return d


def merge_tracks(save_tracks, track_dir, track_type, target_key_set):
    new_count = 0
    skip_count = 0
    for file in tqdm(sorted(glob(os.path.join(track_dir, "*", "*"))), desc=track_type):
        imdbid = file.split("/")[-2]
        shot_id = file.split("/")[-1].split(".")[0]

        key = (imdbid, int(shot_id))
        if key not in target_key_set:
            continue

        with open(file, "rb") as f:
            tracks = pickle.load(f)

        save_tracks.setdefault(imdbid, {}).setdefault(track_type, {})
        for obj_id, box_track in tracks.items():
            tid = f"{shot_id}_{obj_id}"
            if tid in save_tracks[imdbid][track_type]:
                skip_count += 1
                continue
            save_tracks[imdbid][track_type][tid] = np.array(box_track)
            new_count += 1

    print(f"[{track_type}] Added {new_count} new tracks, skipped {skip_count} existing")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sid_file", type=str, required=True,
                        help='JSON mapping {imdbid: [shot_id, ...]}.')
    parser.add_argument("--face_track_dir", type=str, required=True,
                        help="Directory holding <imdbid>/<shot_id>.pkl per-shot face tracks.")
    parser.add_argument("--save_file", type=str, required=True,
                        help="Output pickle path.")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.sid_file, "r") as infile:
        target_imdbid_sid = json.load(infile)
    target_key_set = {(imdbid, sid)
                      for imdbid, sid_lst in target_imdbid_sid.items()
                      for sid in sid_lst}

    save_tracks = {}
    merge_tracks(save_tracks, args.face_track_dir, "face", target_key_set)
    save_tracks = to_dict(save_tracks)

    os.makedirs(os.path.dirname(args.save_file), exist_ok=True)
    with open(args.save_file, "wb") as f:
        pickle.dump(save_tracks, f)
    print(f"Saved to {args.save_file}")


if __name__ == "__main__":
    main()
