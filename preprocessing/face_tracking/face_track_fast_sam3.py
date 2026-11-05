"""Per-shot face tracking with SAM 3.

For each (imdbid, shot_id) listed in --sid_file:
    1. dumps the shot's frames to a temp folder,
    2. prompts SAM 3 with the text "face" on a few anchor frames,
    3. propagates the prompts through the shot,
    4. (optionally) splits long shots into sub-chunks and re-stitches the
       object ids across chunks via IoU on the join frame,
    5. pickles a {obj_id: [[frame, x1, y1, x2, y2], ...]} dict to
       <save_dir>/face_box_track/<imdbid>/<shot_id>.pkl.

The script is sharded via --num_chunks / --chunk_idx so it can be launched in
parallel on multiple GPUs.
"""

import argparse
import gc
import json
import os
import pickle
import random
import shutil
import tempfile
from collections import defaultdict
from glob import glob

import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm

from sam3.model_builder import build_sam3_video_predictor


def initialise_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_list(lst, n):
    if n <= 0:
        raise ValueError("Number of chunks must be positive")
    if not lst:
        return [[] for _ in range(n)]
    q, r = divmod(len(lst), n)
    chunks, start = [], 0
    for i in range(n):
        end = start + q + (1 if i < r else 0)
        chunks.append(lst[start:end])
        start = end
    return chunks


def split_frames_into_temp_folders(shot_vid_folder, max_frames_per_chunk):
    """Return [(folder, track_start_fid, num_frames), ...].

    If the shot is short enough we use it directly. Otherwise frames are
    copied into sub-folders of a freshly-made temp dir so SAM 3 can be run
    chunk-by-chunk under tight VRAM.
    """
    frame_files = sorted(os.listdir(shot_vid_folder),
                         key=lambda x: int(os.path.splitext(x)[0]))
    num_frames = len(frame_files)

    if num_frames <= max_frames_per_chunk:
        track_start_fid = min(int(os.path.splitext(x)[0]) for x in frame_files)
        return [(shot_vid_folder, track_start_fid, num_frames)], None

    base_temp_dir = tempfile.mkdtemp(prefix="sam3_temp_")
    num_chunks = (num_frames + max_frames_per_chunk - 1) // max_frames_per_chunk
    temp_folders = []
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * max_frames_per_chunk
        end_idx = min((chunk_idx + 1) * max_frames_per_chunk, num_frames)
        chunk_frames = frame_files[start_idx:end_idx]

        temp_folder = os.path.join(base_temp_dir, f"chunk_{chunk_idx}")
        os.makedirs(temp_folder, exist_ok=True)
        for frame_file in chunk_frames:
            shutil.copy2(os.path.join(shot_vid_folder, frame_file),
                         os.path.join(temp_folder, frame_file))

        track_start_fid = int(os.path.splitext(chunk_frames[0])[0])
        temp_folders.append((temp_folder, track_start_fid, len(chunk_frames)))
    return temp_folders, base_temp_dir


def save_shot_frames_to_temp_dir(vr, start_fid, end_fid, prefix):
    """Decode [start_fid, end_fid] into a temp folder of {fid}.jpg images."""
    if end_fid < start_fid:
        raise ValueError(f"Invalid frame range: {start_fid}..{end_fid}")
    temp_dir = tempfile.mkdtemp(prefix=prefix)
    try:
        for fid in range(start_fid, end_fid + 1):
            frame = vr[fid].asnumpy()
            Image.fromarray(frame).convert("RGB").save(
                os.path.join(temp_dir, f"{fid}.jpg"), "JPEG"
            )
        return temp_dir
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def is_cuda_oom_error(error):
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    msg = str(error).lower()
    return "out of memory" in msg or "cuda out of memory" in msg


def process_single_folder(predictor, folder_path, track_start_fid,
                          W, H, prompt_text_str="face"):
    """Run SAM 3 on a single image folder and return per-object boxes.

    Boxes are returned in absolute pixel coords as
    {obj_id: [[global_fid, x1, y1, x2, y2], ...]}.
    """
    session_id = None
    try:
        response = predictor.handle_request(
            request=dict(type="start_session", resource_path=folder_path)
        )
        session_id = response["session_id"]

        predictor.handle_request(
            request=dict(type="reset_session", session_id=session_id)
        )

        num_frames = len(os.listdir(folder_path))
        # Prompt at three evenly-spaced anchor frames so the propagator can
        # discover faces that appear / disappear mid-shot.
        for fid in np.linspace(0, num_frames - 1, 3, dtype=int).tolist():
            predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=int(fid),
                    text=prompt_text_str,
                )
            )

        outputs_per_frame = {}
        for response in predictor.handle_stream_request(
            request=dict(type="propagate_in_video", session_id=session_id)
        ):
            outputs_per_frame[response["frame_index"]] = response["outputs"]
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        if is_cuda_oom_error(e):
            if session_id is not None:
                try:
                    predictor.handle_request(
                        request=dict(type="close_session", session_id=session_id)
                    )
                except Exception:
                    pass
            clear_gpu_cache()
            raise RuntimeError(f"CUDA OOM error processing {folder_path}: {e}")
        raise

    boxes = defaultdict(list)
    for relative_fid in outputs_per_frame:
        global_fid = relative_fid + track_start_fid
        obj_ids = outputs_per_frame[relative_fid]["out_obj_ids"].tolist()
        out_boxes_xywh = outputs_per_frame[relative_fid]["out_boxes_xywh"]
        out_boxes_xyxy = out_boxes_xywh.copy()
        out_boxes_xyxy[..., 2] = out_boxes_xywh[..., 0] + out_boxes_xywh[..., 2]
        out_boxes_xyxy[..., 3] = out_boxes_xywh[..., 1] + out_boxes_xywh[..., 3]

        for obj_id, out_box in zip(obj_ids, out_boxes_xyxy):
            boxes[obj_id].append([
                global_fid,
                int(out_box[0] * W), int(out_box[1] * H),
                int(out_box[2] * W), int(out_box[3] * H),
            ])

    for obj_id in boxes:
        boxes[obj_id].sort(key=lambda x: x[0])

    predictor.handle_request(
        request=dict(type="close_session", session_id=session_id)
    )
    return boxes, outputs_per_frame


def calculate_iou(box1, box2):
    """IoU on [fid, x1, y1, x2, y2] boxes (fid is ignored)."""
    x1_min, y1_min, x1_max, y1_max = box1[1:5]
    x2_min, y2_min, x2_max, y2_max = box2[1:5]

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def merge_chunk_results(all_chunk_results):
    """Stitch the per-chunk box dicts into one global obj_id space.

    Objects from chunk i+1 are matched to chunk i by IoU between the last
    frame of i and the first frame of i+1 (threshold 0.5). Unmatched objects
    get fresh ids.
    """
    if len(all_chunk_results) == 1:
        return dict(all_chunk_results[0])

    merged_boxes = defaultdict(list)
    next_obj_id = 0

    # Seed with chunk 0.
    obj_id_mapping_0 = {}
    for obj_id in all_chunk_results[0].keys():
        obj_id_mapping_0[obj_id] = next_obj_id
        next_obj_id += 1
    for old_obj_id, box_list in all_chunk_results[0].items():
        merged_boxes[obj_id_mapping_0[old_obj_id]].extend(box_list)

    for chunk_idx in range(1, len(all_chunk_results)):
        boxes_curr = all_chunk_results[chunk_idx]
        if not boxes_curr or not merged_boxes:
            for obj_id in boxes_curr.keys():
                merged_boxes[next_obj_id].extend(boxes_curr[obj_id])
                next_obj_id += 1
            continue

        last_fid_prev = max(b[0] for box_list in merged_boxes.values()
                            for b in box_list)
        first_fid_curr = min(b[0] for box_list in boxes_curr.values()
                             for b in box_list)

        prev_last_boxes = {}
        for gid, box_list in merged_boxes.items():
            for box in box_list:
                if box[0] == last_fid_prev:
                    prev_last_boxes[gid] = box
                    break
        curr_first_boxes = {}
        for cid, box_list in boxes_curr.items():
            for box in box_list:
                if box[0] == first_fid_curr:
                    curr_first_boxes[cid] = box
                    break

        obj_id_mapping = {}
        used_prev = set()
        for cid, curr_box in curr_first_boxes.items():
            best_iou, best_prev = 0.0, None
            for gid, prev_box in prev_last_boxes.items():
                if gid in used_prev:
                    continue
                iou = calculate_iou(prev_box, curr_box)
                if iou > best_iou:
                    best_iou, best_prev = iou, gid
            if best_iou > 0.5 and best_prev is not None:
                obj_id_mapping[cid] = best_prev
                used_prev.add(best_prev)
            else:
                obj_id_mapping[cid] = next_obj_id
                next_obj_id += 1
        for cid in boxes_curr.keys():
            if cid not in obj_id_mapping:
                obj_id_mapping[cid] = next_obj_id
                next_obj_id += 1

        for old_obj_id, box_list in boxes_curr.items():
            merged_boxes[obj_id_mapping[old_obj_id]].extend(box_list)

    for obj_id in merged_boxes:
        merged_boxes[obj_id] = sorted(merged_boxes[obj_id], key=lambda x: x[0])
    return dict(merged_boxes)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--sid_file", type=str, required=True,
                        help="JSON list of [imdbid, shot_id] to process.")
    parser.add_argument("--shotmeta_file", type=str, required=True,
                        help="Shot metadata JSON from shot_detect.py.")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--max_frames_per_chunk", type=int, default=500,
                        help="Max frames given to SAM 3 in one pass.")
    parser.add_argument("--prompt", type=str, default="face")
    return parser.parse_args()


def main():
    args = parse_args()
    initialise_seed()
    os.makedirs(args.save_dir, exist_ok=True)

    print("Initialising SAM 3")
    predictor = build_sam3_video_predictor(
        gpus_to_use=[torch.cuda.current_device()]
    )

    with open(args.shotmeta_file, "r") as f:
        shot_metadata = json.load(f)
    with open(args.sid_file, "r") as f:
        target_sid_lst = json.load(f)

    # Skip shots that already have a saved track.
    past_file_set = set()
    for file in glob(os.path.join(args.save_dir, "face_box_track", "*", "*")):
        imdbid = file.split("/")[-2]
        shot_id = int(file.split("/")[-1].split(".")[0])
        past_file_set.add((imdbid, shot_id))

    tbp_sid_lst = [tuple(key) for key in target_sid_lst
                   if tuple(key) not in past_file_set]
    chunked_lsts = split_list(tbp_sid_lst, args.num_chunks)
    work = chunked_lsts[args.chunk_idx]

    vr = None
    last_imdbid = None
    for imdbid, shot_id in tqdm(work,
            desc=f"Processing chunk {args.chunk_idx+1}/{args.num_chunks}"):

        if imdbid != last_imdbid:
            vr = VideoReader(os.path.join(args.video_dir, f"{imdbid}.mp4"),
                             ctx=cpu(0))
            last_imdbid = imdbid

        start_fid, end_fid = shot_metadata[imdbid][str(shot_id)]["frame_range"]
        shot_temp_dir = save_shot_frames_to_temp_dir(
            vr, start_fid, end_fid, prefix=f"sam3_shot_{imdbid}_{shot_id}_"
        )

        img_path = os.path.join(shot_temp_dir, f"{start_fid}.jpg")
        with Image.open(img_path) as img:
            W, H = img.convert("RGB").size

        # Retry with smaller chunks on OOM, down to 50 frames.
        current_max_frames = args.max_frames_per_chunk
        max_retries = 3
        success = False
        folder_list = None
        base_temp_dir = None
        all_chunk_results = []

        for retry_count in range(max_retries):
            if base_temp_dir is not None and os.path.exists(base_temp_dir):
                shutil.rmtree(base_temp_dir)
                base_temp_dir = None
            try:
                folder_list, base_temp_dir = split_frames_into_temp_folders(
                    shot_temp_dir, current_max_frames,
                )
                all_chunk_results = []
                for folder_path, chunk_track_start_fid, _ in folder_list:
                    boxes, _ = process_single_folder(
                        predictor, folder_path, chunk_track_start_fid,
                        W, H, args.prompt,
                    )
                    all_chunk_results.append(boxes)
                    clear_gpu_cache()
                success = True
                break
            except RuntimeError as e:
                if not is_cuda_oom_error(e):
                    raise
                current_max_frames = max(50, current_max_frames // 2)
                print(f"[WARNING] CUDA OOM on {shot_temp_dir}; reducing "
                      f"chunk size to {current_max_frames}")
                clear_gpu_cache()
                all_chunk_results = []

        if not success:
            print(f"[ERROR] Failed {shot_temp_dir} after {max_retries} retries.")
            if base_temp_dir is not None and os.path.exists(base_temp_dir):
                shutil.rmtree(base_temp_dir)
            shutil.rmtree(shot_temp_dir, ignore_errors=True)
            continue

        try:
            boxes = (merge_chunk_results(all_chunk_results)
                     if len(folder_list) > 1 else all_chunk_results[0])

            save_face_file = os.path.join(args.save_dir, "face_box_track",
                                          imdbid, f"{shot_id}.pkl")
            os.makedirs(os.path.dirname(save_face_file), exist_ok=True)
            with open(save_face_file, "wb") as f:
                pickle.dump(dict(boxes), f)
        except (RuntimeError, MemoryError) as e:
            if is_cuda_oom_error(e):
                print(f"[ERROR] CUDA OOM during merge/save for {shot_temp_dir}.")
                clear_gpu_cache()
            else:
                raise
        finally:
            if base_temp_dir is not None and os.path.exists(base_temp_dir):
                shutil.rmtree(base_temp_dir)
            shutil.rmtree(shot_temp_dir, ignore_errors=True)
            del all_chunk_results
            gc.collect()


if __name__ == "__main__":
    main()
