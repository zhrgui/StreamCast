import pickle
import re
import numpy as np
import torch
from metaphone import doublemetaphone

def _l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).ravel()
    n = np.linalg.norm(x)
    return x / max(n, 1e-12)

def load_character_bank(pkl_path: str, imdbid: str):
    """Load pre-built IMDB character bank → (cast_names, cast_features).

    cast_names : list[str]  - e.g. "Charles - Wedding One"
    cast_features : Tensor[N, 512]
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data[imdbid]

def find_character_embeddings(char_name: str, cast_names: list, cast_features):
    """Return all profile embeddings whose base name matches *char_name*.

    Cast names have the form "Character - Scene", so we split on " - " and
    compare the character portion.
    """
    indices = []
    for i, cn in enumerate(cast_names):
        base = cn.split(" - ")[0].strip()
        if base.lower() == char_name.lower():
            indices.append(i)
    if not indices:
        return None
    return cast_features[indices]  # Tensor[K, 512]

def hungarian_match(gt_embs_list: list, cluster_feats: dict, tau: float = -1.0) -> list:
    """One-to-one assignment of GT characters to predicted clusters.

    Similarity between a GT and a cluster is the max cosine similarity across
    the GT's profile embeddings. We run the Hungarian algorithm
    (``scipy.optimize.linear_sum_assignment``) on the negated similarity
    matrix to maximize the total assigned similarity.

    gt_embs_list  : list where element *i* is a [K_i, D] tensor/ndarray of
                    profile embeddings for one GT character.
    cluster_feats : {cluster_id: np.ndarray[D]}.
    tau           : assignments with similarity < ``tau`` become
                    ``(None, sim)``. Leave at the default to disable.

    Returns a list aligned to ``gt_embs_list`` of ``(cid_or_None, sim)``.
    GTs without an assignment (e.g. more GTs than clusters) get
    ``(None, -1.0)``.
    """
    from scipy.optimize import linear_sum_assignment

    n_gt = len(gt_embs_list)
    out: list = [(None, -1.0)] * n_gt
    if n_gt == 0 or not cluster_feats:
        return out

    cids = list(cluster_feats.keys())
    sim = np.full((n_gt, len(cids)), -1.0, dtype=np.float32)
    normed_cfeats = [_l2(cluster_feats[cid]) for cid in cids]
    for i, embs in enumerate(gt_embs_list):
        if isinstance(embs, torch.Tensor):
            arr = embs.detach().cpu().numpy()
        else:
            arr = np.asarray(embs)
        normed = np.stack([_l2(arr[k]) for k in range(arr.shape[0])])
        for j, cfeat in enumerate(normed_cfeats):
            sim[i, j] = float((normed @ cfeat).max())

    row_ind, col_ind = linear_sum_assignment(-sim)
    for r, c in zip(row_ind, col_ind):
        s = float(sim[r, c])
        if s < tau:
            out[r] = (None, s)
        else:
            out[r] = (cids[c], s)
    return out

def _name_tokens(name: str) -> list[str]:
    return [p for p in re.split(r"[\s\-]+", name.strip()) if p]

def _metaphone_codes(token: str) -> set:
    """Return non-empty Double Metaphone codes for a single token."""
    primary, secondary = doublemetaphone(token)
    codes: set = set()
    if primary:
        codes.add(primary)
    if secondary:
        codes.add(secondary)
    return codes

def _tokens_metaphone_match(tok_a: str, tok_b: str) -> bool:
    """True when two tokens share at least one Double Metaphone code."""
    if tok_a.lower() == tok_b.lower():
        return True
    codes_a = _metaphone_codes(tok_a)
    codes_b = _metaphone_codes(tok_b)
    if not codes_a or not codes_b:
        return tok_a.lower() == tok_b.lower()
    return bool(codes_a & codes_b)

def _levenshtein(a: str, b: str) -> int:
    a, b = a.lower(), b.lower()
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + (ca != cb),
            )
        prev = curr
    return prev[-1]

def _tokens_fuzzy_match(tok_a: str, tok_b: str, max_edits: int = 2) -> bool:
    """Metaphone match, or Levenshtein distance within ``max_edits``.

    Length guard prevents short tokens (e.g. "Bob" vs "Tom") from
    matching purely by edit distance.
    """
    if _tokens_metaphone_match(tok_a, tok_b):
        return True
    if min(len(tok_a), len(tok_b)) <= max_edits + 2:
        return False
    return _levenshtein(tok_a, tok_b) <= max_edits

def names_match(gt_name: str, pred_name: str | None) -> bool:
    """True when pred_name token-level matches gt_name.

    Every token in the shorter side must match some token in the longer
    side via Double Metaphone. A single-token side may match any token
    in the longer side (so "Mike" matches "Governor Mike Morris").
    As a fallback for typos, two multi-token names also match when both
    their first-token and last-token pairs match within Levenshtein
    distance 2 (or via the standard metaphone check).
    """
    if pred_name is None:
        return False
    pred_tokens = _name_tokens(pred_name)
    gt_tokens = _name_tokens(gt_name)
    if not pred_tokens or not gt_tokens:
        return False

    shorter, longer = (
        (pred_tokens, gt_tokens)
        if len(pred_tokens) <= len(gt_tokens)
        else (gt_tokens, pred_tokens)
    )
    if len(shorter) == 1 and len(longer) > 1:
        return any(_tokens_metaphone_match(shorter[0], l) for l in longer)

    remaining = list(longer)
    all_matched = True
    for s_tok in shorter:
        matched = False
        for i, l_tok in enumerate(remaining):
            if _tokens_metaphone_match(s_tok, l_tok):
                remaining.pop(i)
                matched = True
                break
        if not matched:
            all_matched = False
            break
    if all_matched:
        return True

    if len(pred_tokens) >= 2 and len(gt_tokens) >= 2:
        return (
            _tokens_fuzzy_match(pred_tokens[0], gt_tokens[0])
            and _tokens_fuzzy_match(pred_tokens[-1], gt_tokens[-1])
        )
    return False
