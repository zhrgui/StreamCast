"""Principal Character Recall (PCR@K) against IMDB principals.

For each of the top-K IMDB principals (ranked by annotation weight), mark it
as *found* iff some predicted cluster satisfies both legs simultaneously:
  - name leg: the cluster's collision-resolved canonical name (entry.name)
              matches the GT under names_match;
  - face leg: cosine(cluster.face_feat, GT profile embedding) >= tau_face.

The face primitive is identical to eval_ap.py. The name primitive uses
entry.name (the single canonical name that survives cross-cluster
collision resolution) rather than the full name_votes dict, so PCR
reflects the bank's production output: one name per cluster.
"""
import argparse
import json
import os
import pickle
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Banks are pickled with schemas.CharacterEntry, which lives in character_bank_construction/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "character_bank_construction"))
from utils import _l2, find_character_embeddings, load_character_bank, names_match


def _gt_profile_matrix(
    gt_name: str, cast_names: list, cast_features: object,
) -> Optional[np.ndarray]:
    """Return (k, d) L2-normalized profile embeddings for a GT character, or None."""
    ch = find_character_embeddings(gt_name, cast_names, cast_features)
    if ch is None:
        return None
    if isinstance(ch, torch.Tensor):
        ch = ch.detach().cpu().numpy()
    arr = np.asarray(ch, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


def _load_pred_store(path: str) -> Dict[int, object]:
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    return data.get("store", {})


def evaluate_movie(
    imdbid: str,
    pred_bank_path: str,
    gt_characters: Dict[str, dict],
    gt_bank_pkl: str,
    tau_face: float,
    top_k: int,
) -> Optional[dict]:
    if not os.path.exists(pred_bank_path):
        print(f"[skip] {imdbid}: predicted bank missing at {pred_bank_path}")
        return None

    store = _load_pred_store(pred_bank_path)

    try:
        cast_names, cast_features = load_character_bank(gt_bank_pkl, imdbid)
    except KeyError:
        print(f"[skip] {imdbid}: missing from GT bank {gt_bank_pkl}")
        return None

    ranked = sorted(
        gt_characters.items(),
        key=lambda kv: float((kv[1] or {}).get("weight", 0.0)),
        reverse=True,
    )

    cluster_feat: Dict[int, Optional[np.ndarray]] = {}
    cluster_name: Dict[int, Optional[str]] = {}
    for cid, entry in store.items():
        f = getattr(entry, "face_feat", None)
        cluster_feat[cid] = (
            _l2(np.asarray(f, dtype=np.float32)) if f is not None else None
        )
        cluster_name[cid] = getattr(entry, "name", None)

    per_gt: List[dict] = []
    n_eval = 0
    n_found = 0

    # Walk the weight-ranked list; skip principals with no reference
    # embedding (the next-ranked one takes their slot) until top_k are
    # eligible, or the list is exhausted.
    for gt_name, _meta in ranked:
        if n_eval >= top_k:
            break
        gt_emb = _gt_profile_matrix(gt_name, cast_names, cast_features)
        if gt_emb is None:
            per_gt.append({"gt_name": gt_name, "status": "skipped_no_gt_face"})
            continue
        n_eval += 1

        joint_hit = False
        best_sim = -1.0
        winning: Optional[Tuple[int, str]] = None
        for cid, feat in cluster_feat.items():
            if feat is None:
                continue
            pred_name = cluster_name[cid]
            if not names_match(gt_name, pred_name):
                continue

            sim = float((feat @ gt_emb.T).max())
            if sim < tau_face:
                continue

            if not joint_hit or sim > best_sim:
                joint_hit = True
                best_sim = sim
                winning = (cid, pred_name)

        n_found += int(joint_hit)

        per_gt.append({
            "gt_name": gt_name,
            "status": "found" if joint_hit else "miss",
            "matched_cluster": winning[0] if winning else None,
            "matched_name": winning[1] if winning else None,
            "sim": round(best_sim, 4) if joint_hit else None,
        })

    pcr = n_found / n_eval if n_eval else 0.0

    return {
        "imdbid": imdbid,
        "n_gt": len(gt_characters),
        "n_eval": n_eval,
        "n_pred_clusters": len(store),
        "found": n_found,
        "pcr": round(pcr, 4),
        "per_gt": per_gt,
    }


def main(args: argparse.Namespace) -> None:
    with open(args.gt_annotations, "r") as f:
        all_gt = json.load(f)

    imdbids = (
        list(os.listdir(args.pred_log_dir))
        if args.imdbid is None
        else [args.imdbid]
    )

    per_movie: List[dict] = []
    agg_n_eval = 0
    agg_found = 0

    for imdbid in imdbids:

        if imdbid == "tt0119822":
            continue

        if imdbid not in all_gt:
            continue

        pred_path = os.path.join(args.pred_log_dir, imdbid, "character_bank.pkl")
        result = evaluate_movie(
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

        print(
            f"[{imdbid}] eval={result['n_eval']:3d}  "
            f"found={result['found']:3d}  "
            f"PCR@{args.top_k}={result['pcr']:.3f}"
        )

    if not per_movie:
        print("No movies evaluated.")
        return

    micro_pcr = agg_found / agg_n_eval if agg_n_eval else 0.0
    macro_pcr = float(np.mean([m["pcr"] for m in per_movie]))

    summary = {
        "n_movies": len(per_movie),
        "tau_face": args.tau_face,
        "top_k": args.top_k,
        "micro": {
            "n_eval": agg_n_eval,
            "found": agg_found,
            "pcr": round(micro_pcr, 4),
        },
        "macro": {
            "pcr": round(macro_pcr, 4),
        },
    }

    print("\n=== Overall ===")
    print(f"movies evaluated: {len(per_movie)}")
    print(
        f"micro  PCR@{args.top_k}={micro_pcr:.4f}  "
        f"(found={agg_found}, N={agg_n_eval})"
    )
    print(f"macro  PCR@{args.top_k}={macro_pcr:.4f}")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"summary": summary, "per_movie": per_movie}, f, indent=2)
        print(f"\n[DONE] Wrote {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PCR@K: principal-character recall against IMDB GT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pred_log_dir",
        required=True,
        help="Dir with {imdbid}/character_bank.pkl predicted banks.",
    )
    parser.add_argument(
        "--gt_annotations",
        required=True,
        help="Ground-truth per-movie character dict; each entry carries a 'weight'.",
    )
    parser.add_argument(
        "--gt_bank_pkl",
        required=True,
        help="IMDB character bank pickle with per-character profile embeddings.",
    )
    parser.add_argument(
        "--imdbid", default=None,
        help="If set, evaluate only this imdbid; otherwise iterate all.",
    )
    parser.add_argument(
        "--tau_face", type=float, default=0.45,
        help="Cosine threshold for the face leg.",
    )
    parser.add_argument(
        "--top_k", type=int, default=15,
        help="Take the top-K IMDB principals ranked by annotation weight.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional path to write per-movie + summary JSON.",
    )

    main(parser.parse_args())
