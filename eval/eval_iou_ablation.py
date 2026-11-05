"""mTIoU evaluation across error-correction ablation variants.

Reads a single full-pipeline output directory (e.g. logs/20260418) and
reports mTIoU for each post-hoc ablation in one pass. Timeline JSONs on
disk are not modified — for variants that change per-shot names we
replay them in memory from each cluster's ``evidence_log``.

Variants reported (mTIoU only — for mAP see eval_ap_ablation.py):
    full          Per-shot names from existing ``timeline/cluster_*.json``
                  files. These are already a replay from evidence_log
                  with collision resolution and without metaphone (see
                  agent._write_cluster_timelines).
    no_metaphone  Same as ``full`` in the current pipeline — timeline
                  replay never applied metaphone, so disabling it has no
                  effect on per-shot names. Surfaced for completeness.
    no_collision  Replayed from evidence_log without collision
                  resolution: each cluster keeps the argmax of its own
                  cumulative votes at every shot.

Usage:
    python eval_iou_ablation.py \
        --output_dir <logs/run> \
        --annotation_file <annotations.json> \
        --character_bank_pkl <IMDB_bank.pkl> \
        --video_dir <video_dir> \
        --shot_seg_file <shot_metadata.json>
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "character_bank_construction"))

import eval_iou  # noqa: E402
from agent import CharacterBankAgent  # noqa: E402

VARIANTS = ("full", "no_metaphone", "no_collision")


def _shot_time_range(shot_seg: dict, sid: str) -> Tuple[float, float]:
    info = shot_seg.get(sid, shot_seg.get(int(sid) if sid.isdigit() else sid, {}))
    tr = info.get("time_range", [0.0, 0.0])
    return float(tr[0]), float(tr[1])


def _replay_timelines(
    store: dict, shot_seg: dict, use_collision: bool,
) -> Dict[int, dict]:
    """Mirror agent._write_cluster_timelines, but with collision toggled.

    Returns ``{cluster_id: record}`` with the same JSON schema eval_iou
    expects from ``timeline/cluster_*.json``.
    """
    entries = [e for e in store.values() if e.shots_seen]

    entry_shots: Dict[int, List[str]] = {}
    entry_evidence: Dict[int, Dict[str, list]] = {}
    all_shots: set = set()
    for entry in entries:
        all_shot_ids = list(dict.fromkeys(entry.shots_seen))
        for ev in entry.evidence_log:
            if ev.shot_id not in all_shot_ids:
                all_shot_ids.append(ev.shot_id)
        entry_shots[entry.cluster_id] = sorted(
            all_shot_ids, key=lambda s: _shot_time_range(shot_seg, s)[0],
        )
        ev_by_shot: Dict[str, list] = defaultdict(list)
        for ev in entry.evidence_log:
            ev_by_shot[ev.shot_id].append(ev)
        entry_evidence[entry.cluster_id] = ev_by_shot
        all_shots.update(all_shot_ids)

    sorted_all = sorted(all_shots, key=lambda s: _shot_time_range(shot_seg, s)[0])
    cum_votes: Dict[int, Dict[str, float]] = {e.cluster_id: {} for e in entries}
    ev_count: Dict[int, int] = {e.cluster_id: 0 for e in entries}
    per_shot_names: Dict[int, Dict[str, Optional[str]]] = {e.cluster_id: {} for e in entries}
    own_shots: Dict[int, set] = {e.cluster_id: set(entry_shots[e.cluster_id]) for e in entries}

    for shot_id in sorted_all:
        for entry in entries:
            cid = entry.cluster_id
            for ev in entry_evidence[cid].get(shot_id, []):
                cum_votes[cid][ev.name] = cum_votes[cid].get(ev.name, 0.0) + float(ev.confidence)
                ev_count[cid] += 1

        if use_collision:
            state = {
                e.cluster_id: (cum_votes[e.cluster_id], ev_count[e.cluster_id])
                for e in entries
            }
            resolved = CharacterBankAgent._resolve_collisions(state)
            for entry in entries:
                cid = entry.cluster_id
                if shot_id in own_shots[cid]:
                    per_shot_names[cid][shot_id] = resolved[cid][0]
        else:
            for entry in entries:
                cid = entry.cluster_id
                if shot_id not in own_shots[cid]:
                    continue
                votes = cum_votes[cid]
                if not votes:
                    per_shot_names[cid][shot_id] = None
                else:
                    per_shot_names[cid][shot_id] = max(votes.items(), key=lambda kv: kv[1])[0]

    records: Dict[int, dict] = {}
    for entry in entries:
        cid = entry.cluster_id
        sorted_shots = entry_shots[cid]
        timeline = []
        prev_name: Optional[str] = None
        for shot_id in sorted_shots:
            ts, te = _shot_time_range(shot_seg, shot_id)
            name = per_shot_names[cid].get(shot_id)
            timeline.append({
                "shot_id": shot_id,
                "time_start": ts,
                "time_end": te,
                "name": name,
                "name_changed": name != prev_name,
            })
            prev_name = name

        visual_shots = sorted(
            entry.shots_seen, key=lambda s: _shot_time_range(shot_seg, s)[0],
        ) if entry.shots_seen else []
        first_visual = None
        if visual_shots:
            vs = visual_shots[0]
            vs_ts, vs_te = _shot_time_range(shot_seg, vs)
            first_visual = {"shot_id": vs, "time_start": vs_ts, "time_end": vs_te}

        records[cid] = {
            "cluster_id": cid,
            "final_name": entry.name,
            "first_appearance": {
                "shot_id": sorted_shots[0],
                "time_start": timeline[0]["time_start"],
                "time_end": timeline[0]["time_end"],
            },
            "first_visual_appearance": first_visual,
            "timeline": timeline,
        }
    return records


def _load_store(bank_pkl: str) -> dict:
    with open(bank_pkl, "rb") as f:
        return pickle.load(f).get("store", {})


def _build_variant_timelines(
    pred_log_dir: str, imdbid: str, variant: str, shot_seg: dict,
) -> dict:
    """Return a {cid: record} timeline dict for the given variant."""
    if variant == "full":
        return eval_iou.load_timelines(os.path.join(pred_log_dir, imdbid, "timeline"))

    bank_pkl = os.path.join(pred_log_dir, imdbid, "character_bank.pkl")
    store = _load_store(bank_pkl)
    if variant == "no_metaphone":
        # Per-shot timelines don't depend on metaphone — return the
        # collision-resolved replay (matches what the pipeline wrote).
        return _replay_timelines(store, shot_seg, use_collision=True)
    if variant == "no_collision":
        return _replay_timelines(store, shot_seg, use_collision=False)
    raise ValueError(f"Unknown variant: {variant!r}")


def _evaluate_variant(args, annotations, imdbids, shot_seg_by_imdb, variant):
    """Run eval_iou.process_imdbid for every imdbid under one variant.

    Build all in-memory timelines BEFORE patching the loader: the
    "full" variant calls ``eval_iou.load_timelines`` to read the
    on-disk JSONs, and that needs to resolve to the original loader.
    """
    timelines_for: dict = {}
    eligible: list = []
    for imdbid in imdbids:
        if imdbid == "tt0119822":
            continue
        if imdbid not in annotations:
            continue
        shot_seg = shot_seg_by_imdb.get(imdbid)
        if shot_seg is None:
            print(f"[skip] {imdbid}: missing from shot_seg_file")
            continue
        timelines_for[imdbid] = _build_variant_timelines(
            args.output_dir, imdbid, variant, shot_seg,
        )
        eligible.append(imdbid)

    _orig_loader = eval_iou.load_timelines

    def _patched(timeline_dir, _holder=timelines_for):
        imdbid = os.path.basename(os.path.dirname(timeline_dir))
        return _holder[imdbid]

    eval_iou.load_timelines = _patched
    try:
        all_results = []
        name_mious: List[float] = []
        vis_mious: List[float] = []
        for imdbid in eligible:
            try:
                out = eval_iou.process_imdbid(args, imdbid, annotations[imdbid])
            except Exception as exc:
                print(f"[ERROR] {imdbid} failed: {exc}")
                continue
            if out is None:
                continue
            all_results.append(out)
            name_mious.append(out["weighted_name_miou"])
            vis_mious.append(out["weighted_vis_miou"])
    finally:
        eval_iou.load_timelines = _orig_loader

    if not name_mious:
        return None

    return {
        "n_movies": len(name_mious),
        "mean_name_miou": float(np.mean(name_mious)),
        "median_name_miou": float(np.median(name_mious)),
        "mean_vis_miou": float(np.mean(vis_mious)),
        "median_vis_miou": float(np.median(vis_mious)),
        "per_movie": all_results,
    }


def main(args: argparse.Namespace) -> None:
    annotations = eval_iou.load_annotations(args.annotation_file)
    with open(args.shot_seg_file, "r") as f:
        shot_seg_by_imdb = json.load(f)

    imdbids = [args.imdbid] if args.imdbid else list(annotations.keys())

    summary = {}
    for variant in VARIANTS:
        print()
        print("=" * 70)
        print(f"VARIANT: {variant}")
        print("=" * 70)
        result = _evaluate_variant(
            args, annotations, imdbids, shot_seg_by_imdb, variant,
        )
        summary[variant] = result
        if result is None:
            print("  (no movies evaluated)")
            continue
        print(
            f"  movies={result['n_movies']:3d}   "
            f"mean mTIoU_n={result['mean_name_miou']:.4f}   "
            f"mTIoU_v={result['mean_vis_miou']:.4f}"
        )

    print()
    print("=" * 70)
    print("ABLATION TABLE (mean weighted mTIoU)")
    print("=" * 70)
    print(f"{'variant':<22s} {'mTIoU_v':>10s} {'mTIoU_n':>10s} {'n_movies':>10s}")
    print("-" * 56)
    for v in VARIANTS:
        r = summary.get(v)
        if r is None:
            print(f"{v:<22s} {'--':>10s} {'--':>10s} {'--':>10s}")
            continue
        print(
            f"{v:<22s} {r['mean_vis_miou']*100:>9.2f}  "
            f"{r['mean_name_miou']*100:>9.2f}  {r['n_movies']:>10d}"
        )

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"variants": summary, "tau_face": args.tau_face}, f, indent=2)
        print(f"\n[DONE] Wrote {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="mTIoU across post-hoc error-correction ablations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Full-run log directory ({output_dir}/{imdbid}/{timeline,character_bank.pkl}).",
    )
    parser.add_argument(
        "--annotation_file",
        required=True,
    )
    parser.add_argument(
        "--character_bank_pkl",
        required=True,
    )
    parser.add_argument(
        "--video_dir",
        required=True,
    )
    parser.add_argument(
        "--shot_seg_file",
        required=True,
    )
    parser.add_argument("--imdbid", default=None)
    parser.add_argument("--tau_face", type=float, default=0.45)
    parser.add_argument("--output", default=None)
    main(parser.parse_args())
