import numpy as np
from collections import defaultdict, deque


def _l2_normalize(x: np.ndarray, eps: float = 0) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, eps, None)


def _mean_track_face_feat(track: dict) -> np.ndarray:
    f = track["track_face_feats"]
    if f is None:
        return None
    if isinstance(f, list):
        f_ = [f_e for f_e in f if f_e is not None]
        f = np.stack(f_, dtype=np.float32)
    if isinstance(f, np.ndarray) and f.ndim == 1:
        return f
    return f.mean(axis=0, keepdims=False)


def _has_temporal_overlap(set_a: set, set_b: set) -> bool:
    return not set_a.isdisjoint(set_b)


def _connected_components(adj_list):
    visited = set()
    comps = []
    for start in adj_list.keys():
        if start in visited:
            continue
        q = deque([start])
        visited.add(start)
        comp = []
        while q:
            u = q.popleft()
            comp.append(u)
            for v in adj_list[u]:
                if v not in visited:
                    visited.add(v)
                    q.append(v)
        comps.append(comp)
    return comps


def _get_mean_feat_from_bank(feat_bank, member_indices):
    feat_list = [feat_bank[idx] for idx in member_indices]
    return np.mean(feat_list, axis=0, dtype=np.float32)


def cluster_tracks(
    episode_tracks,
    tau_tight_face=0.55,
):
    """
    Cluster person tracks using face features.

    Face-based agglomerative clustering (tight NN + connected components).
    Tracks without face features are dropped.

    Args:
        episode_tracks: list of track dicts, each containing:
            - track_occurence_indices: list[int] of frame indices
            - track_face_feats: ndarray or None
            - label: ground-truth label (optional, carried through)
        tau_tight_face: cosine distance threshold for face NN linking.

    Returns:
        list of cluster dicts, each with keys ``members`` (list of track indices
        into *episode_tracks*) and ``labels``.
    """
    if len(episode_tracks) == 0:
        return []

    face_feat_bank = [_mean_track_face_feat(tr) for tr in episode_tracks]

    clusters = []
    for i, tr in enumerate(episode_tracks):
        if face_feat_bank[i] is None:
            continue
        entry = {
            "members": [i],
            "labels": [tr.get("label", None)],
            "time_set": set(tr["track_occurence_indices"]),
        }
        clusters.append(entry)

    while True:
        K = len(clusters)
        if K <= 1:
            break

        feats = np.stack(
            [_get_mean_feat_from_bank(face_feat_bank, c["members"]) for c in clusters],
            axis=0,
        ).astype(np.float32)
        feats = _l2_normalize(feats)

        nn = [None] * K
        nn_dist = [np.inf] * K

        for i in range(K):
            best_j, best_d = None, np.inf
            time_i = clusters[i]["time_set"]
            fi = feats[i]
            for j in range(K):
                if i == j:
                    continue
                if _has_temporal_overlap(time_i, clusters[j]["time_set"]):
                    continue
                d = 1.0 - float(np.dot(fi, feats[j]))
                if d <= tau_tight_face and d < best_d:
                    best_d = d
                    best_j = j
            nn[i] = best_j
            nn_dist[i] = best_d

        adj = {i: set() for i in range(K)}
        for i in range(K):
            j = nn[i]
            if j is None:
                continue
            adj[i].add(j)
            adj[j].add(i)

        # Common-NN linking: connect nodes that share the same first NN
        inv_map = defaultdict(list)
        for i, j in enumerate(nn):
            if j is not None:
                inv_map[j].append(i)

        for common, group in inv_map.items():
            if len(group) < 2:
                continue
            for a in range(len(group)):
                for b in range(a + 1, len(group)):
                    ii, jj = group[a], group[b]
                    if _has_temporal_overlap(clusters[ii]["time_set"], clusters[jj]["time_set"]):
                        continue
                    adj[ii].add(jj)
                    adj[jj].add(ii)

        comps = _connected_components(adj)
        if all(len(c) == 1 for c in comps):
            break

        new_clusters = []
        for comp in comps:
            if len(comp) == 1:
                new_clusters.append(clusters[comp[0]])
                continue
            merged_members = []
            merged_labels = []
            merged_time = set()
            for idx in comp:
                merged_members.extend(clusters[idx]["members"])
                merged_labels.extend(clusters[idx]["labels"])
                merged_time |= clusters[idx]["time_set"]
            new_clusters.append({
                "members": merged_members,
                "labels": merged_labels,
                "time_set": merged_time,
            })
        clusters = new_clusters

    for c in clusters:
        c.pop("time_set", None)

    return clusters
