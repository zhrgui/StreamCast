import argparse
import json
import os
import sys
import pickle
from collections import defaultdict
from agent import CharacterBankAgent

def load_json(file_path: str) -> object:
    with open(file_path, "r") as fh:
        return json.load(fh)

def load_tracks(file_path: str, imdbid: str | None = None) -> dict:
    """Load person tracks from JSON or PKL. Supports per-imdbid or flat format."""
    if file_path.endswith(".pkl") or file_path.endswith(".pickle"):
        with open(file_path, "rb") as fh:
            data = pickle.load(fh)
    else:
        data = load_json(file_path)

    if isinstance(data, dict):
        if "face" in data or "body" in data:
            return data
        if imdbid is not None:
            if str(imdbid) not in data:
                raise KeyError(f"{imdbid} is not in track file {file_path}")
            return data[str(imdbid)]
        first_key = next(iter(data.keys()))
        return data.get(first_key, data)
    return {}

def split_subtitles(shot_seg: dict, subtitles: list) -> dict:
    """Assign each subtitle segment to the shot with the greatest overlap."""
    sid_to_subtitles: dict = defaultdict(list)
    for sub in subtitles:
        sub_start, sub_end = sub["start"], sub["end"]
        best_sid, best_overlap = None, 0
        for sid, shot_range in shot_seg.items():
            shot_start, shot_end = shot_range["time_range"]
            overlap = min(sub_end, shot_end) - max(sub_start, shot_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_sid = str(sid)
        if best_sid is not None:
            sid_to_subtitles[best_sid].append(sub)
    for sid in sid_to_subtitles:
        sid_to_subtitles[sid].sort(key=lambda s: s["start"])
    return sid_to_subtitles

def process_imdbid(args: argparse.Namespace, imdbid: str) -> None:
    print(f"\n[INFO] === Processing {imdbid} ===")

    run_dir = os.path.join(args.output_dir, imdbid)
    if os.path.isdir(run_dir):
        print(f"[SKIP] {run_dir} already exists.")
        return

    print(f"[INFO] Loading shot segmentation from: {args.shot_seg_file}")
    shot_seg = load_json(args.shot_seg_file)[imdbid]

    print(f"[INFO] Loading transcript from: {os.path.join(args.transcript_dir, f'{imdbid}.json')}")
    subtitles = load_json(os.path.join(args.transcript_dir, f'{imdbid}.json'))

    print(f"[INFO] Loading tracks from: {args.track_file}")
    person_tracks = load_tracks(args.track_file, imdbid=imdbid)

    os.makedirs(run_dir, exist_ok=True)
    character_bank_path = os.path.join(run_dir, "character_bank.pkl")
    trajectory_log_path = os.path.join(run_dir, "trajectory_log.pkl")
    timeline_dir = os.path.join(run_dir, "timeline")
    vis_output_dir = os.path.join(run_dir, "bank_vis")
    vlm_frames_vis_dir = os.path.join(run_dir, "vlm_frames")
    summary_path = os.path.join(run_dir, "summary.json")
    movie_file = os.path.join(args.video_dir, f"{imdbid}.mp4")

    # Split subtitles by shot
    sid_to_subtitles = split_subtitles(shot_seg, subtitles)

    agent = CharacterBankAgent(
        tau_face=args.tau_face_merge,
        tau_tight_face=args.tau_tight_face,
        tau_voice=args.tau_voice,
        lang=args.name_lang,
        flush_min_shots=args.flush_min_shots,
        video_file=movie_file,
        base_vlm_url=args.base_vlm_url or None,
        base_vlm_model_id=args.base_vlm_model_id,
        vlm_linker_url=args.vlm_linker_url or None,
        vlm_linker_model_id=args.vlm_linker_model_id or None,
        vlm_sample_fps=args.vlm_sample_fps,
        vlm_max_tokens=args.vlm_max_tokens,
        llm_url=args.llm_url or None,
        llm_model_id=args.llm_model_id,
        diar_model=args.diar_model,
        score_dir=args.score_dir,
        imdbid=imdbid,
        sync_threshold=args.sync_threshold,
        overwrite_threshold=args.overwrite_threshold,
        diar_threshold=args.diar_threshold,
        vis_every_k=args.vis_every_k,
        vis_output_dir=vis_output_dir,
        vlm_frames_vis_dir=vlm_frames_vis_dir,
        vlm_face_filter_vis_dir=args.vlm_face_filter_vis_dir,
        timeline_dir=timeline_dir,
        verbose=args.verbose,
    )

    result = agent.run(
        shot_seg=shot_seg,
        person_tracks=person_tracks,
        sid_to_subtitles=sid_to_subtitles,
        bank_path=character_bank_path,
        trajectory_log_path=trajectory_log_path,
    )

    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    print(
        f"\n[DONE] {result['total_clusters']} clusters "
        f"({result['named_clusters']} named, "
        f"{len(result['unresolved_cluster_ids'])} unresolved)."
    )
    print(f"[DONE] Character bank written to: {character_bank_path}")
    print(f"[DONE] Pipeline summary written to: {summary_path}")

    if result["unresolved_cluster_ids"]:
        print(
            f"[WARN] Unresolved cluster IDs (>={args.flush_min_shots} shots, no name): "
            f"{result['unresolved_cluster_ids']}"
        )

def main(args: argparse.Namespace) -> None:
    if args.imdbid_lst is not None:
        imdbids = load_json(args.imdbid_lst)
        total = len(imdbids)
        if args.num_chunks > 1:
            start = total * args.chunk_idx // args.num_chunks
            end = total * (args.chunk_idx + 1) // args.num_chunks
            imdbids = imdbids[start:end]
            print(f"[INFO] Chunk {args.chunk_idx}/{args.num_chunks}: processing {len(imdbids)}/{total} imdbids (indices {start}:{end})")
        else:
            print(f"[INFO] Iterating over {total} imdbids from {args.imdbid_lst}")
    else:
        imdbids = [args.imdbid]

    for imdbid in imdbids:
        process_imdbid(args, imdbid)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Agentic streaming character bank construction from a movie.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--shot_seg_file", required=True,
        help="Path to shot segmentation JSON ({shot_id: {time_range: [start, end]}}).",
    )
    parser.add_argument(
        "--track_file", required=True,
        help="Path to person tracks JSON or PKL.",
    )
    parser.add_argument(
        "--transcript_dir", required=True,
        help="Path to subtitles JSON (list of {start, end, text}).",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Root output directory. Per-run artifacts are written to {output_dir}/{imdbid}/ — character_bank.pkl, trajectory_log.pkl, summary.json, and timeline/.",
    )
    parser.add_argument(
        "--video_dir", required=True,
        help="Directory containing {imdbid}.mp4 files (required for VLM name-face linking).",
    )
    parser.add_argument(
        "--imdbid", default="tt0330373",
        help="IMDb ID when the track file is keyed by episode/imdbid.",
    )
    parser.add_argument(
        "--imdbid_lst", default=None,
        help="Optional path to a JSON file containing a list of imdbids. If set, --imdbid is ignored and the pipeline iterates over the list.",
    )
    parser.add_argument(
        "--num_chunks", type=int, default=1,
        help="Split --imdbid_lst into this many contiguous chunks. Use with --chunk_idx to run one shard.",
    )
    parser.add_argument(
        "--chunk_idx", type=int, default=0,
        help="Zero-based index of the chunk this process should run (requires --num_chunks > 1).",
    )
    parser.add_argument(
        "--base_vlm_url", default="http://127.0.0.1:8001/v1",
        help="Base URL of the vLLM OpenAI-compatible API for the base agent model for face filtering and cluster correction.",
    )
    parser.add_argument(
        "--base_vlm_model_id",
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Base agent model name passed to the API — must match the SERVED_MODEL_NAME that serve_vlm.sh registers.",
    )
    parser.add_argument(
        "--vlm_linker_url", default="http://127.0.0.1:8000/v1",
        help="VLM API URL for name-to-face linking. Defaults to --base_vlm_url when not set.",
    )
    parser.add_argument(
        "--vlm_linker_model_id",
        default="qwen3vl_4b_thinking_train_stage2_grpo_rl_aug",
        help="Model name for the name-linking VLM. Defaults to --base_vlm_model_id when not set.",
    )
    parser.add_argument(
        "--vlm_sample_fps", type=float, default=2.0,
        help="Video frame sampling rate for VLM (same scheme as naming/dataloader: 2 FPS).",
    )
    parser.add_argument(
        "--vlm_max_tokens", type=int, default=1024,
        help="Token budget for the VLM response.",
    )
    parser.add_argument(
        "--llm_url", default="http://127.0.0.1:8002/v1",
        help="Base URL of a text-only LLM OpenAI-compatible API to review/correct name clustering and filter non-character names before VLM linking.",
    )
    parser.add_argument(
        "--llm_model_id", default="Qwen/Qwen3.5-9B",
        help="Model name for the text-only LLM server.",
    )
    parser.add_argument(
        "--tau_face_merge", type=float, default=0.6,
        help="Cosine-distance threshold for bank-level face cluster merging.",
    )
    parser.add_argument(
        "--tau_tight_face", type=float, default=0.6,
        help="Cosine-distance threshold for within-shot face clustering.",
    )
    parser.add_argument(
        "--tau_voice", type=float, default=0.25,
        help="Cosine-distance threshold for voice feature matching.",
    )
    parser.add_argument(
        "--name_lang", default="en", choices=["en", "zh"],
        help="Language hint for name detection.",
    )
    parser.add_argument(
        "--flush_min_shots", type=int, default=3,
        help="Minimum shots a cluster must appear in to be listed as unresolved.",
    )
    parser.add_argument(
        "--vis_every_k", type=int, default=50,
        help="Save a character-bank contact-sheet image every this many shots to {output_dir}/{imdbid}/bank_vis/.",
    )
    parser.add_argument(
        "--vlm_face_filter_vis_dir", default=None,
        help="If set, save one JPEG per face-filter VLM call: crops + raw answer + parsed JSON.",
    )
    parser.add_argument(
        "--diar_model", default="BUT-FIT/diarizen-wavlm-large-s80-md-v2",
        help="DiariZen model name (HuggingFace hub ID or local path).",
    )
    parser.add_argument(
        "--score_dir", required=True,
        help="Root directory of per-shot audio-visual sync score files ({score_dir}/{imdbid}/{shot_id}.pkl). Pass an empty string to skip sync scoring.",
    )
    parser.add_argument(
        "--sync_threshold", type=float, default=-0.075,
        help="Minimum max sync score for a diar_speaker→color mapping to be accepted.",
    )
    parser.add_argument(
        "--overwrite_threshold", type=float, default=0.1,
        help="If any color's mean sync score exceeds this, assign it directly (no diar needed).",
    )
    parser.add_argument(
        "--diar_threshold", type=float, default=0.05,
        help="Minimum diar_iou for a subtitle to use the diar-based speaker mapping.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print per-shot diagnostics.",
    )
    args = parser.parse_args()
    main(args)
