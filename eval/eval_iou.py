import argparse
import glob
import json
import os
import cv2
import numpy as np
from utils import find_character_embeddings, hungarian_match, load_character_bank, names_match

def get_video_duration(video_path: str) -> float:
    """Total duration of a video file (seconds) via OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    if fps <= 0 or n_frames <= 0:
        raise RuntimeError(f"Bad fps/frame_count for {video_path}: fps={fps}, n={n_frames}")
    return float(n_frames / fps)

def load_annotations(path: str) -> dict:
    """Load {char_name: {linkage_time, note}} from the annotation JSON."""
    with open(path) as f:
        return json.load(f)

def load_cluster_features(timeline_dir: str) -> dict:
    """Load {cluster_id: np.ndarray} from cluster_features.pkl."""
    import pickle

    path = os.path.join(timeline_dir, "cluster_features.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

def load_timelines(timeline_dir: str) -> dict:
    """Load all cluster_*.json files → {cluster_id: record}."""
    timelines = {}
    for fpath in sorted(glob.glob(os.path.join(timeline_dir, "cluster_*.json"))):
        with open(fpath) as f:
            rec = json.load(f)
        timelines[rec["cluster_id"]] = rec
    return timelines

def compute_name_iou(timeline: list, gt_name: str, start_time: float, movie_duration: float) -> tuple[float, float | None]:
    """IoU between GT interval ``[start_time, movie_duration]`` and the
    union of all predicted segments whose name matches the GT.

    Returns ``(iou, first_name_start)``. ``first_name_start``
    is the ``time_start`` of the first timeline entry whose name matches,
    or ``None`` when nothing matches.

    Each timeline entry at index ``i`` covers
    ``[entry[i].time_start, entry[i+1].time_start)`` (or up to
    ``movie_duration`` for the last). Predictions may flip correct → wrong
    → correct, so we union every matching segment rather than stopping at
    the first name change.
    """
    pred_segments = []
    first_name_start: float | None = None
    for i, entry in enumerate(timeline):
        if not names_match(gt_name, entry["name"]):
            continue
        seg_start = entry["time_start"]
        if first_name_start is None:
            first_name_start = seg_start
        seg_end = (
            timeline[i + 1]["time_start"]
            if i + 1 < len(timeline)
            else movie_duration
        )
        if seg_end > seg_start:
            pred_segments.append((seg_start, seg_end))

    if not pred_segments:
        return 0.0, first_name_start

    gt_start, gt_end = start_time, movie_duration
    inter = 0.0
    pred_total = 0.0
    for s, e in pred_segments:
        pred_total += e - s
        inter += max(0.0, min(e, gt_end) - max(s, gt_start))

    union = pred_total + (gt_end - gt_start) - inter
    iou = inter / union if union > 0 else 0.0
    return iou, first_name_start

def compute_visual_iou(timeline: list, start_time: float, movie_duration: float) -> tuple[float, float | None]:
    """IoU between GT interval ``[start_time, movie_duration]`` and the
    cluster's visible interval ``[first_visual_start, movie_duration]``.

    ``first_visual_start`` is the ``time_start`` of the first entry in
    the cluster's timeline, regardless of the assigned name — i.e. when
    the system first saw this identity at all. Returns
    ``(iou, first_visual_start)``.
    """
    if not timeline:
        return 0.0, None
    first_visual_start = timeline[0]["time_start"]
    pred_end = movie_duration
    gt_start, gt_end = start_time, movie_duration
    inter = max(0.0, min(pred_end, gt_end) - max(first_visual_start, gt_start))
    union = (pred_end - first_visual_start) + (gt_end - gt_start) - inter
    iou = inter / union if union > 0 else 0.0
    return iou, first_visual_start

def process_imdbid(args, imdbid: str, imdbid_anno: dict) -> dict | None:
    """Evaluate one movie. Returns {imdbid, results, weighted_miou} or None."""
    timeline_dir = os.path.join(args.output_dir, imdbid, "timeline")
    video_path = os.path.join(args.video_dir, f"{imdbid}.mp4")
    movie_duration = get_video_duration(video_path)
    cast_names, cast_features = load_character_bank(args.character_bank_pkl, imdbid)
    cluster_feats = load_cluster_features(timeline_dir)
    timelines = load_timelines(timeline_dir)

    print(f"\n=== {imdbid} ===")
    print(f"Movie duration : {movie_duration:.1f}s")
    print(f"Clusters       : {len(timelines)}")
    print(f"Annotated chars: {len(imdbid_anno)}")
    print(f"Cast profiles  : {len(cast_names)}")

    header = (
        f"{'Character':<35s} {'Cluster':>7s}  {'Cluster Name':<22s}  "
        f"{'Sim':>5s}  {'NameIoU':>7s}  {'VisIoU':>6s}  {'Weight':>6s}  "
        f"{'GTvis':>7s}  {'GTname':>7s}  {'PredVis':>8s}  {'PredName':>8s}"
    )
    print(header)
    print("-" * len(header))

    eligible = []
    for char_name, char_info in imdbid_anno.items():
        visual_time = char_info.get("visual_time")
        linkage_time = char_info.get("linkage_time")
        weight = float(char_info.get("weight", 0.0))
        if visual_time is None or linkage_time is None or weight <= 0.0:
            continue

        char_embs = find_character_embeddings(char_name, cast_names, cast_features)
        if char_embs is None:
            print(f"{char_name:<35s}  -- no profile image in character bank --")
            continue
        eligible.append((char_name, visual_time, linkage_time, weight, char_embs))

    assignments = hungarian_match([e[4] for e in eligible], cluster_feats, tau=args.tau_face)

    results = []
    name_weighted_sum = 0.0
    vis_weighted_sum = 0.0
    weight_total = 0.0
    for item, (best_cid, best_sim) in zip(eligible, assignments):
        char_name, visual_time, linkage_time, weight = item[0], item[1], item[2], item[3]
        if best_cid is None or best_cid not in timelines:
            name_iou = 0.0
            vis_iou = 0.0
            cluster_name = "(no match)"
            results.append({
                "character": char_name,
                "visual_time": visual_time,
                "linkage_time": linkage_time,
                "weight": weight,
                "cluster_id": None,
                "cluster_name": cluster_name,
                "similarity": round(best_sim, 4),
                "name_iou": name_iou,
                "vis_iou": vis_iou,
                "name_start": None,
                "vis_start": None,
            })
            name_weighted_sum += weight * name_iou
            vis_weighted_sum += weight * vis_iou
            weight_total += weight
            print(
                f"{char_name:<35s} {'--':>7s}  {cluster_name:<22s}  "
                f"{best_sim:>5.3f}  {name_iou:>7.3f}  {vis_iou:>6.3f}  "
                f"{weight:>6.3f}  {visual_time:>6.1f}s  {linkage_time:>6.1f}s  "
                f"{'--':>8s}  {'--':>8s}"
            )
            continue

        rec = timelines[best_cid]
        name_iou, name_start = compute_name_iou(
            rec["timeline"], char_name, linkage_time, movie_duration,
        )
        vis_iou, vis_start = compute_visual_iou(
            rec["timeline"], visual_time, movie_duration,
        )

        cluster_name = rec["final_name"] or "(unnamed)"
        results.append({
            "character": char_name,
            "visual_time": visual_time,
            "linkage_time": linkage_time,
            "weight": weight,
            "cluster_id": best_cid,
            "cluster_name": cluster_name,
            "similarity": round(best_sim, 4),
            "name_iou": round(name_iou, 4),
            "vis_iou": round(vis_iou, 4),
            "name_start": None if name_start is None else round(name_start, 2),
            "vis_start": None if vis_start is None else round(vis_start, 2),
        })
        name_weighted_sum += weight * name_iou
        vis_weighted_sum += weight * vis_iou
        weight_total += weight

        name_start_str = "--" if name_start is None else f"{name_start:.1f}s"
        vis_start_str = "--" if vis_start is None else f"{vis_start:.1f}s"
        print(
            f"{char_name:<35s} {best_cid:>7d}  {cluster_name:<22s}  "
            f"{best_sim:>5.3f}  {name_iou:>7.3f}  {vis_iou:>6.3f}  "
            f"{weight:>6.3f}  {visual_time:>6.1f}s  {linkage_time:>6.1f}s  "
            f"{vis_start_str:>8s}  {name_start_str:>8s}"
        )

    name_miou = name_weighted_sum / weight_total if weight_total > 0 else 0.0
    vis_miou = vis_weighted_sum / weight_total if weight_total > 0 else 0.0
    print(
        f"[{imdbid}] weighted name-mIoU = {name_miou:.4f}  "
        f"weighted vis-mIoU = {vis_miou:.4f}  "
        f"(evaluated {len(results)} / {len(imdbid_anno)} chars)"
    )

    return {
        "imdbid": imdbid,
        "results": results,
        "weighted_name_miou": name_miou,
        "weighted_vis_miou": vis_miou,
    }


def main(args):
    annotations = load_annotations(args.annotation_file)

    if args.imdbid:
        if args.imdbid not in annotations:
            print(f"[ERROR] {args.imdbid} not in annotations.")
            return
        imdbids = [args.imdbid]
    else:
        imdbids = list(annotations.keys())
        print(f"[INFO] Evaluating {len(imdbids)} imdbids from annotations.")

    all_results = []
    name_mious = []
    vis_mious = []
    for imdbid in imdbids:
        if imdbid == "tt0119822":
            continue

        try:
            out = process_imdbid(args, imdbid, annotations[imdbid])
        except Exception as e:
            print(f"[ERROR] {imdbid} failed: {e}")
            continue
        if out is None:
            continue
        all_results.append(out)
        name_mious.append(out["weighted_name_miou"])
        vis_mious.append(out["weighted_vis_miou"])

    if name_mious:
        print()
        print("=" * 60)
        print(f"Movies evaluated : {len(name_mious)} / {len(imdbids)}")
        print(f"Mean weighted name-mIoU (across movies): {np.mean(name_mious):.4f}")
        print(f"Median weighted name-mIoU              : {np.median(name_mious):.4f}")
        print(f"Mean weighted vis-mIoU  (across movies): {np.mean(vis_mious):.4f}")
        print(f"Median weighted vis-mIoU               : {np.median(vis_mious):.4f}")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults → {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate character naming IoU from cluster timelines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--annotation_file",
        required=True,
        help="GT annotations: {char_name: {time, weighting}}.",
    )
    parser.add_argument(
        "--character_bank_pkl",
        required=True,
        help="Pre-built IMDB character bank with InsightFace embeddings.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help=(
            "Root directory with per-imdbid runs. Timeline for each imdbid is "
            "expected at {output_dir}/{imdbid}/timeline/."
        ),
    )
    parser.add_argument(
        "--video_dir",
        required=True,
        help="Directory containing {imdbid}.mp4 files (for reading total movie duration).",
    )
    parser.add_argument(
        "--output", default="",
        help="Optional path to save per-movie results as JSON.",
    )
    parser.add_argument(
        "--imdbid", default=None,
        help="IMDB to evaluate; if empty, evaluate all imdbids in annotations.",
    )
    parser.add_argument(
        "--tau_face", type=float, default=0.45,
        help="Min cosine similarity for a Hungarian assignment to be kept.",
    )

    args = parser.parse_args()
    main(args)
