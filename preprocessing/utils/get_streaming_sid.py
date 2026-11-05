"""List shot keys (imdbid, shot_id) that still need face tracking.

Compares the full set of shots in --shot_metafile against face tracks already
written under --person_track and dumps the missing (imdbid, shot_id) pairs
to --save_file. The output is consumed by face_track_fast_sam3.py as its
--sid_file argument.
"""

import argparse
import json
import os
from collections import defaultdict
from glob import glob


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot_metafile", type=str, required=True,
                        help="Shot metadata JSON from shot_detect.py.")
    parser.add_argument("--person_track", type=str, required=True,
                        help="Root containing face_tracks/face_box_track/.")
    parser.add_argument("--imdbid_list", type=str, default=None,
                        help="JSON list of imdbids to consider.")
    parser.add_argument("--imdbid", type=str, default=None,
                        help="If set, restrict to this single imdbid.")
    parser.add_argument("--save_file", type=str, required=True,
                        help="Output JSON list of unprocessed [imdbid, sid].")
    parser.add_argument("--all_key_file", type=str, required=True,
                        help="Output JSON dictionary of used [imdbid, sid].")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.shot_metafile, "r") as f:
        shot_meta = json.load(f)

    if args.imdbid is not None:
        imdbid_list = [args.imdbid]
    else:
        with open(args.imdbid_list, "r") as f:
            imdbid_list = json.load(f)

    streaming_sid_list = []
    streaming_sid_dict = defaultdict(list)
    for imdbid in imdbid_list:
        for sid in shot_meta[imdbid]:
            streaming_sid_list.append((imdbid, int(sid)))
            streaming_sid_dict[imdbid].append(int(sid))

    face_done = set()
    for file in glob(os.path.join(args.person_track,
                                  "face_tracks", "face_box_track", "*", "*")):
        imdbid = file.split("/")[-2]
        shot_id = int(file.split("/")[-1].split(".")[0])
        face_done.add((imdbid, shot_id))

    target_lst = [key for key in streaming_sid_list if key not in face_done]

    print(f"Total streaming sids: {len(streaming_sid_list)}")
    print(f"  Per imdbid: { {k: len(v) for k, v in streaming_sid_dict.items()} }")
    print(f"Total unprocessed sids: {len(target_lst)}")

    os.makedirs(os.path.dirname(args.save_file), exist_ok=True)
    with open(args.save_file, "w") as f:
        json.dump(target_lst, f, indent=4)
    print(f"Saved to {args.save_file}")

    os.makedirs(os.path.dirname(args.all_key_file), exist_ok=True)
    with open(args.all_key_file, "w") as f:
        json.dump(streaming_sid_dict, f, indent=4)
    print(f"Saved to {args.all_key_file}")


if __name__ == "__main__":
    main()
