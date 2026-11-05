"""Per-track face feature extraction with InsightFace (buffalo_l).

Samples k frames uniformly from each face track in the input pickle, runs
InsightFace face detection on those frames, and matches detections back to the
track's reference bbox by IoU to recover the embedding. The output pickle
mirrors the input plus a 'face_feats' field per episode:

    {imdbid: {'face': ..., 'face_feats': {track_id: (k, 512) np.float32}}}

Supports sharding (--shard_id / --num_shards) for trivial multi-GPU parallel
launches, and skips tracks that already have a saved embedding.
"""

import os
# Tame ONNX Runtime / OpenMP so each shard runs single-threaded.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("ORT_DISABLE_CPU_AFFINITY", "1")
os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "2")
os.environ.setdefault("ORT_FORCE_CONCURRENT_EXECUTION", "0")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

import argparse
import copy
import json
import pickle
import time

import numpy as np
from decord import VideoReader, cpu
from insightface.app import FaceAnalysis
from tqdm import tqdm

def ensure_dir_for(path: str):
    if not path:
        return
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def set_seed(seed: int = 42):
    np.random.seed(seed)

def iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    iw = max(0, x2 - x1); ih = max(0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a1 = max(0, (b1[2]-b1[0])) * max(0, (b1[3]-b1[1]))
    a2 = max(0, (b2[2]-b2[0])) * max(0, (b2[3]-b2[1]))
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anno_path", type=str, required=True,
                        help="Input pickle with per-episode face tracks (output of select_track_file.py).")
    parser.add_argument("--vid_dir", type=str, required=True,
                        help="Directory holding <imdbid>.mp4 video files.")
    parser.add_argument("--save_path", type=str, required=True,
                        help="Output pickle path. Existing entries are reused so the job is resumable.")
    parser.add_argument("--det_thresh", type=float, default=0.5)
    parser.add_argument("--det_size", type=int, default=640)
    parser.add_argument("--k", type=int, default=15, help="Samples per track.")
    parser.add_argument("--chunk", type=int, default=256,
                        help="Unused; kept for metadata logging only.")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--episodes", type=str, default="",
                        help="Comma-separated list of imdbids to restrict to.")
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout_sec", type=float, default=0.0,
                        help="Soft per-frame budget; frames over budget yield no feature.")
    return parser.parse_args()

def _match_and_pick_feats(det_bboxes, det_feats, ref_bboxes, min_iou=0.3):
    if not det_bboxes:
        return [None] * len(ref_bboxes)
    out = []
    for rb in ref_bboxes:
        ious = [iou(rb, db) for db in det_bboxes]
        best = int(np.argmax(ious)) if np.max(ious) >= min_iou else -1
        if best == -1 or best >= len(det_feats) or det_feats[best] is None:
            out.append(None)
        else:
            emb = det_feats[best]
            if isinstance(emb, np.ndarray):
                emb = emb.astype(np.float32).tolist()
            else:
                emb = list(emb)
            out.append(emb)
    return out

def sample_k_indices(n: int, k: int):
    if n <= k:
        return list(range(n))
    chunk = n / float(k)
    sampled = [np.random.randint(int(i * chunk), int((i + 1) * chunk)) for i in range(k)]
    return [min(idx, n - 1) for idx in sampled]

def process_episode(episode_idx, episode_data, args, existing_feats=None):
    episode_vid = os.path.join(args.vid_dir, f"{episode_idx}.mp4")
    if not os.path.exists(episode_vid):
        return episode_idx, {}

    try:
        vr = VideoReader(episode_vid, ctx=cpu(0))
        nframes = len(vr)
    except Exception:
        return episode_idx, {}

    providers = [("CUDAExecutionProvider", {"device_id": int(args.device_id)})]
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=int(args.device_id), det_thresh=float(args.det_thresh),
                det_size=(int(args.det_size), int(args.det_size)))

    skip_tids = set()
    if existing_feats:
        skip_tids = set(existing_feats.keys())

    # Collate sampled frames & reference boxes
    ep_frame_idxs = []
    ep_face_boxes = []
    ep_track_ids = []
    for track_id, track_face in episode_data.get('face', {}).items():
        if track_id in skip_tids:
            continue
        frame_indices = [int(i[0]) for i in track_face]
        face_boxes = [[int(i[1]), int(i[2]), int(i[3]), int(i[4])] for i in track_face]
        n = len(frame_indices)
        if n == 0:
            continue
        sel = sample_k_indices(n, args.k)
        for s in sel:
            fi = frame_indices[s]
            if 0 <= fi < nframes:
                ep_frame_idxs.append(fi)
                ep_face_boxes.append(face_boxes[s])
                ep_track_ids.append(track_id)

    if not ep_frame_idxs:
        return episode_idx, {}

    # Deduplicate frames and build per-frame reference mapping
    uniq_frames = sorted(set(ep_frame_idxs))
    pos = {fi: p for p, fi in enumerate(uniq_frames)}
    per_frame_ref_boxes = [[] for _ in uniq_frames]
    reverse_map = [[] for _ in uniq_frames]  # sampled indices pointing to this frame

    for gi, (fi, bx) in enumerate(zip(ep_frame_idxs, ep_face_boxes)):
        if fi in pos:
            k = pos[fi]
            per_frame_ref_boxes[k].append(bx)
            reverse_map[k].append(gi)

    sampled_feats = [None] * len(ep_frame_idxs)

    # Sequential inference over unique frames
    pbar = tqdm(total=len(uniq_frames), desc=f"Episode {episode_idx}", leave=False)
    soft_budget = float(args.timeout_sec or 0.0)
    for k, fi in enumerate(uniq_frames):
        t0 = time.time()
        try:
            frame = vr[fi].asnumpy()   # RGB
            img_bgr = frame[..., ::-1] # BGR

            # Optional soft time budget: skip heavy frames (cannot interrupt native call once started)
            if soft_budget > 0.0 and (time.time() - t0) > soft_budget:
                results = [(si, None) for si in reverse_map[k]]
            else:
                faces = app.get(img_bgr)

                det_bboxes, det_feats = [], []
                for f in faces:
                    bb = f['bbox'].tolist() if isinstance(f['bbox'], np.ndarray) else list(f['bbox'])
                    det_bboxes.append([int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])])
                    det_feats.append(f['embedding'])

                det_bboxes = np.clip(np.array(det_bboxes, dtype=np.int32), 0, None).tolist()
                picked = _match_and_pick_feats(det_bboxes, det_feats, per_frame_ref_boxes[k])
                results = list(zip(reverse_map[k], picked))

        except Exception:
            results = [(si, None) for si in reverse_map[k]]

        for si, feat in results:
            sampled_feats[si] = feat
        pbar.update(1)
    pbar.close()

    # Aggregate per-track arrays
    per_track_feats = {}
    for feat, tid in zip(sampled_feats, ep_track_ids):
        if feat is None:
            continue
        per_track_feats.setdefault(tid, []).append(feat)

    for tid in list(per_track_feats.keys()):
        arr = np.asarray(per_track_feats[tid], dtype=np.float32)
        if arr.size == 0:
            del per_track_feats[tid]
        else:
            per_track_feats[tid] = arr
    return episode_idx, per_track_feats

def main():
    args = parse_args()
    set_seed(args.seed)

    with open(args.anno_path, "rb") as f:
        data = pickle.load(f)

    if args.episodes.strip():
        keep = set([e.strip() for e in args.episodes.split(",") if e.strip()])
        data = {k: v for k, v in data.items() if k in keep}

    # Load existing output if present so we can skip already-extracted tracks
    if os.path.exists(args.save_path):
        with open(args.save_path, "rb") as f:
            save_data = pickle.load(f)
        # Ensure all episodes from current anno_path are present
        for k, v in data.items():
            if k not in save_data:
                save_data[k] = copy.deepcopy(v)
    else:
        save_data = copy.deepcopy(data)

    ep_keys = sorted(data.keys())
    if args.num_shards > 1:
        ep_keys = ep_keys[args.shard_id::args.num_shards]
        print(f"Shard {args.shard_id}/{args.num_shards}: processing {len(ep_keys)} episodes")

    # Per-shard save path to avoid concurrent writes
    if args.num_shards > 1:
        base, ext = os.path.splitext(args.save_path)
        shard_save_path = f"{base}_shard{args.shard_id}{ext}"
    else:
        shard_save_path = args.save_path

    total_new = 0
    total_skip = 0

    for ep in tqdm(ep_keys, desc="Episodes"):
        existing_feats = save_data.get(ep, {}).get('face_feats', {})
        skip_count = len(existing_feats)
        total_track_count = len(data[ep].get('face', {}))

        if skip_count >= total_track_count and total_track_count > 0:
            total_skip += skip_count
            continue

        ep_idx, track_feats = process_episode(ep, data[ep], args, existing_feats=existing_feats)

        if 'face_feats' not in save_data[ep_idx]:
            save_data[ep_idx]['face_feats'] = {}
        save_data[ep_idx]['face_feats'].update(track_feats)

        total_new += len(track_feats)
        total_skip += skip_count

    print(f"Shard {args.shard_id}: extracted {total_new} new tracks, skipped {total_skip} existing")

    # When sharded, only save episodes this shard processed
    if args.num_shards > 1:
        shard_data = {ep: save_data[ep] for ep in ep_keys if ep in save_data}
    else:
        shard_data = save_data

    ensure_dir_for(shard_save_path)
    with open(shard_save_path, "wb") as f:
        pickle.dump(shard_data, f)

    meta_path = shard_save_path + ".meta.json"
    ensure_dir_for(meta_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "anno_path": args.anno_path,
            "vid_dir": args.vid_dir,
            "det_thresh": args.det_thresh,
            "det_size": args.det_size,
            "k": args.k,
            "chunk": args.chunk,
            "device_id": args.device_id,
            "shard_id": args.shard_id,
            "num_shards": args.num_shards,
            "timeout_sec": args.timeout_sec,
            "single_thread": True,
        }, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
