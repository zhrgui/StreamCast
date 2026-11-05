"""PCR@K evaluation across error-correction ablation variants.

Reads a single full-pipeline output directory (e.g. logs/20260418) and
reports PCR@K for each post-hoc ablation in one pass. The pickle files on
disk are not modified.

eval_pcr matches against ``entry.name`` (the single collision-resolved
canonical name). Each variant rewrites ``entry.name`` (and ``name_votes``
where applicable) according to its policy, then runs eval_pcr unchanged.

Variants reported:
    full          Bank as written by the pipeline: name_votes are
                  metaphone-merged and entry.name comes from
                  cross-cluster collision resolution
                  (CharacterBankAgent._resolve_collisions).
    no_metaphone  name_votes rebuilt from evidence_log (no phonetic
                  merging). entry.name is then re-derived from those
                  rebuilt votes via the same collision resolution.
    no_collision  name_votes left as-is. entry.name is overwritten with
                  argmax(name_votes), skipping cross-cluster
                  resolution: two clusters may now share a top name.

Usage:
    python eval_pcr_ablation.py \
        --pred_log_dir <logs/run> \
        --gt_annotations <annotations.json> \
        --gt_bank_pkl <IMDB_bank.pkl>
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from typing import Dict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "character_bank_construction"))

import eval_pcr  # noqa: E402
from agent import CharacterBankAgent  # noqa: E402

VARIANTS = ("full", "no_metaphone", "no_collision")


def _rebuild_votes_from_evidence(entry) -> Dict[str, float]:
    votes: Dict[str, float] = {}
    for ev in entry.evidence_log:
        votes[ev.name] = votes.get(ev.name, 0.0) + float(ev.confidence)
    return votes


def _set_names_via_collision(store: dict) -> None:
    """Re-run cross-cluster collision resolution and write entry.name."""
    state = {
        e.cluster_id: (dict(e.name_votes), len(e.evidence_log))
        for e in store.values()
    }
    resolved = CharacterBankAgent._resolve_collisions(state)
    for e in store.values():
        name, conf = resolved[e.cluster_id]
        e.name = name
        e.confidence = conf
        e.aliases = [n for n in e.name_votes if n != name] if e.name_votes else []


def _set_names_via_argmax(store: dict) -> None:
    """Skip collision resolution: each cluster takes its own argmax."""
    for e in store.values():
        if e.name_votes:
            best = max(e.name_votes.items(), key=lambda kv: kv[1])[0]
        else:
            best = None
        e.name = best
        e.confidence = float(e.name_votes.get(best, 0.0)) if best else 0.0
        e.aliases = [n for n in e.name_votes if n != best] if e.name_votes else []


def _apply_variant(store: dict, variant: str) -> dict:
    """Return a deep-copied store with the variant's transformation."""
    new_store = copy.deepcopy(store)
    if variant == "full":
        # Pipeline already wrote collision-resolved entry.name. No-op.
        return new_store
    if variant == "no_metaphone":
        for entry in new_store.values():
            entry.name_votes = _rebuild_votes_from_evidence(entry)
        _set_names_via_collision(new_store)
        return new_store
    if variant == "no_collision":
        _set_names_via_argmax(new_store)
        return new_store
    raise ValueError(f"Unknown variant: {variant!r}")


def _evaluate_variant(args, all_gt, imdbids, variant: str):
    """Run eval_pcr.evaluate_movie for every imdbid under one variant.

    Monkey-patches eval_pcr._load_pred_store so the variant's transformed
    store flows in without copying the rest of evaluate_movie.
    """
    _orig_loader = eval_pcr._load_pred_store

    def _patched(path, _v=variant):
        return _apply_variant(_orig_loader(path), _v)

    eval_pcr._load_pred_store = _patched
    try:
        per_movie = []
        agg_n_eval = agg_found = 0

        for imdbid in imdbids:
            if imdbid == "tt0119822":
                continue
            if imdbid not in all_gt:
                continue
            pred_path = os.path.join(args.pred_log_dir, imdbid, "character_bank.pkl")
            if not os.path.exists(pred_path):
                continue
            result = eval_pcr.evaluate_movie(
                imdbid=imdbid,
                pred_bank_path=pred_path,
                gt_characters=all_gt[imdbid],
                gt_bank_pkl=args.gt_bank_pkl,
                tau_face=args.tau_face,
                top_k=args.top_k,
            )
            if result is None:
                continue
            per_movie.append(result)
            agg_n_eval += result["n_eval"]
            agg_found += result["found"]
    finally:
        eval_pcr._load_pred_store = _orig_loader

    if not per_movie:
        return None

    micro_pcr = agg_found / agg_n_eval if agg_n_eval > 0 else 0.0
    macro_pcr = float(np.mean([m["pcr"] for m in per_movie]))

    return {
        "n_movies": len(per_movie),
        "micro": {
            "n_eval": agg_n_eval,
            "found": agg_found,
            "pcr": round(micro_pcr, 4),
        },
        "macro": {
            "pcr": round(macro_pcr, 4),
        },
        "per_movie": per_movie,
    }


def main(args: argparse.Namespace) -> None:
    with open(args.gt_annotations, "r") as f:
        all_gt = json.load(f)

    imdbids = (
        sorted(d for d in os.listdir(args.pred_log_dir)
               if os.path.isdir(os.path.join(args.pred_log_dir, d)))
        if args.imdbid is None else [args.imdbid]
    )

    summary = {}
    for variant in VARIANTS:
        print()
        print("=" * 70)
        print(f"VARIANT: {variant}")
        print("=" * 70)
        result = _evaluate_variant(args, all_gt, imdbids, variant)
        summary[variant] = result
        if result is None:
            print("  (no movies evaluated)")
            continue
        macro = result["macro"]
        micro = result["micro"]
        print(
            f"  movies={result['n_movies']:3d}   "
            f"macro PCR@{args.top_k}={macro['pcr']:.4f}"
        )
        print(
            f"  micro PCR@{args.top_k}={micro['pcr']:.4f}   "
            f"(found={micro['found']}, N={micro['n_eval']})"
        )

    print()
    print("=" * 70)
    print(f"ABLATION TABLE (PCR@{args.top_k})")
    print("=" * 70)
    print(f"{'variant':<22s} {'macro':>10s} {'micro':>10s}")
    print("-" * 44)
    for v in VARIANTS:
        r = summary.get(v)
        if r is None:
            print(f"{v:<22s} {'--':>10s} {'--':>10s}")
            continue
        print(
            f"{v:<22s} "
            f"{r['macro']['pcr']*100:>9.2f}  "
            f"{r['micro']['pcr']*100:>9.2f}"
        )

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(
                {
                    "variants": summary,
                    "tau_face": args.tau_face,
                    "top_k": args.top_k,
                },
                f,
                indent=2,
            )
        print(f"\n[DONE] Wrote {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PCR@K across post-hoc error-correction ablations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pred_log_dir",
        required=True,
        help="Full-run log directory ({pred_log_dir}/{imdbid}/character_bank.pkl).",
    )
    parser.add_argument(
        "--gt_annotations",
        required=True,
    )
    parser.add_argument(
        "--gt_bank_pkl",
        required=True,
    )
    parser.add_argument("--imdbid", default=None)
    parser.add_argument("--tau_face", type=float, default=0.45)
    parser.add_argument("--top_k", type=int, default=15)
    parser.add_argument("--output", default=None)
    main(parser.parse_args())
