"""Contact-sheet visualization of the streaming CharacterBank."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from memory import CharacterBank

CROP_H: int = 140
CROP_W: int = 140
LABEL_W: int = 480
GAP: int = 3
TRACK_SEP: int = 8
ROW_PAD: int = 6
FACE_PAD: float = 0.15

N_FRAMES_PER_TRACK: int = 1
MAX_TRACKS_PER_CLUSTER: int = 15

_BG = (20, 20, 20)
_NAMED_COLOR = (60, 210, 60)
_UNNAMED_COLOR = (110, 110, 190)
_SECONDARY_COLOR = (210, 170, 60)
_SEP_COLOR = (55, 55, 55)
_FONT_SCALE = 0.70
_FONT_SCALE_SMALL = 0.55
_FONT_THICK = 1
_LINE_STEP = 30


def _crop_face(frame: np.ndarray, det: np.ndarray) -> np.ndarray:
    """Crop a padded face region from *frame*."""
    import cv2
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = float(det[1]), float(det[2]), float(det[3]), float(det[4])
    dx = (x2 - x1) * FACE_PAD
    dy = (y2 - y1) * FACE_PAD
    x1 = max(0, int(x1 - dx))
    y1 = max(0, int(y1 - dy))
    x2 = min(W, int(x2 + dx))
    y2 = min(H, int(y2 + dy))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((CROP_H, CROP_W, 3), dtype=np.uint8)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((CROP_H, CROP_W, 3), dtype=np.uint8)
    return cv2.resize(crop, (CROP_W, CROP_H), interpolation=cv2.INTER_LINEAR)


def _sample_track_crops(
    cap,
    detections,
    n_frames: int = N_FRAMES_PER_TRACK,
) -> List[np.ndarray]:
    """Sample *n_frames* evenly-spaced face crops from a track's detections."""
    import cv2
    dets = np.asarray(detections)
    if len(dets) == 0:
        return []
    indices = np.round(
        np.linspace(0, len(dets) - 1, min(n_frames, len(dets)))
    ).astype(int)
    crops = []
    for idx in indices:
        det = dets[idx]
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(int(det[0])))
        ret, frame = cap.read()
        if ret:
            crops.append(_crop_face(frame, det))
        else:
            crops.append(np.zeros((CROP_H, CROP_W, 3), dtype=np.uint8))
    return crops


def _label_line_count(
    name_votes: Dict[str, float],
    aliases: List[str],
) -> int:
    """Return the number of text lines the label cell needs."""
    n = 3  # header + name + confidence
    if aliases:
        n += 1
    n += len(name_votes)
    return n


def _make_label_cell(
    cluster_id: int,
    name: Optional[str],
    confidence: float,
    n_tracks: int,
    n_shots: int,
    name_votes: Dict[str, float],
    aliases: List[str],
    height: int,
) -> np.ndarray:
    """Render a text label cell."""
    import cv2
    cell = np.full((height, LABEL_W, 3), _BG, dtype=np.uint8)
    color = _NAMED_COLOR if name else _UNNAMED_COLOR
    cv2.rectangle(cell, (1, 1), (LABEL_W - 2, height - 2), color, 2)

    y = 26
    # Header: id / tracks / shots
    cv2.putText(cell, f"id={cluster_id}  tracks={n_tracks}  shots={n_shots}",
                (8, y), cv2.FONT_HERSHEY_SIMPLEX, _FONT_SCALE, color, _FONT_THICK,
                cv2.LINE_AA)
    y += _LINE_STEP

    # Canonical name
    cv2.putText(cell, name if name else "<unnamed>",
                (8, y), cv2.FONT_HERSHEY_SIMPLEX, _FONT_SCALE, color, _FONT_THICK,
                cv2.LINE_AA)
    y += _LINE_STEP

    # Confidence
    cv2.putText(cell, f"conf = {confidence:.2f}",
                (8, y), cv2.FONT_HERSHEY_SIMPLEX, _FONT_SCALE, color, _FONT_THICK,
                cv2.LINE_AA)
    y += _LINE_STEP

    # Aliases
    if aliases:
        alias_str = "aka: " + ", ".join(aliases)
        while len(alias_str) > 3 and cv2.getTextSize(
            alias_str, cv2.FONT_HERSHEY_SIMPLEX, _FONT_SCALE_SMALL, _FONT_THICK
        )[0][0] > LABEL_W - 16:
            alias_str = alias_str[:-4] + "…"
        cv2.putText(cell, alias_str,
                    (8, y), cv2.FONT_HERSHEY_SIMPLEX, _FONT_SCALE_SMALL,
                    color, _FONT_THICK, cv2.LINE_AA)
        y += _LINE_STEP

    for vname, vscore in sorted(name_votes.items(), key=lambda kv: kv[1], reverse=True):
        if y + _LINE_STEP > height - 4:
            break
        label = f"  {vname} = {vscore:.2f}"
        while len(label) > 2 and cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, _FONT_SCALE_SMALL, _FONT_THICK
        )[0][0] > LABEL_W - 16:
            label = label[:-4] + "…"
        cv2.putText(cell, label,
                    (8, y), cv2.FONT_HERSHEY_SIMPLEX, _FONT_SCALE_SMALL,
                    color, _FONT_THICK, cv2.LINE_AA)
        y += _LINE_STEP

    return cell


def _make_secondary_label_cell(
    names: List[str],
    track_ids: List[str],
    shot_id: str,
    height: int,
) -> np.ndarray:
    """Render a label cell for a secondary entry."""
    import cv2
    cell = np.full((height, LABEL_W, 3), _BG, dtype=np.uint8)
    color = _SECONDARY_COLOR
    cv2.rectangle(cell, (1, 1), (LABEL_W - 2, height - 2), color, 2)

    y = 26
    cv2.putText(cell, f"[secondary]  shot={shot_id}",
                (8, y), cv2.FONT_HERSHEY_SIMPLEX, _FONT_SCALE, color, _FONT_THICK,
                cv2.LINE_AA)
    y += _LINE_STEP

    name_str = ", ".join(names) if names else "<unnamed>"
    cv2.putText(cell, name_str,
                (8, y), cv2.FONT_HERSHEY_SIMPLEX, _FONT_SCALE, color, _FONT_THICK,
                cv2.LINE_AA)
    y += _LINE_STEP

    cv2.putText(cell, f"tracks={len(track_ids)}",
                (8, y), cv2.FONT_HERSHEY_SIMPLEX, _FONT_SCALE, color, _FONT_THICK,
                cv2.LINE_AA)
    return cell


def _placeholder_crop(text: str = "no crops") -> np.ndarray:
    """Return a dark placeholder crop."""
    import cv2
    img = np.full((CROP_H, CROP_W, 3), (50, 50, 50), dtype=np.uint8)
    cv2.putText(img, text, (4, CROP_H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (140, 140, 140), 1, cv2.LINE_AA)
    return img


def save_cluster_assets(
    bank: "CharacterBank",
    person_tracks: dict,
    video_path: str,
    save_folder: str,
    n_frames_per_track: int = N_FRAMES_PER_TRACK,
    max_tracks_per_cluster: int = 50,
) -> Optional[str]:
    """Save per-cluster face-crop canvases and bank_summary.json to *save_folder*."""
    entries = list(bank.all_entries())
    if not entries:
        return None

    try:
        import cv2
    except ImportError:
        return None

    if not os.path.isfile(video_path):
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    os.makedirs(save_folder, exist_ok=True)
    face_data = person_tracks.get("face", {})

    summary: Dict[str, dict] = {}

    for entry in sorted(entries, key=lambda e: e.cluster_id):
        cid = entry.cluster_id

        name_list: List[str] = []
        if entry.name:
            name_list.append(entry.name)
        name_list.extend(a for a in entry.aliases if a not in name_list)

        summary[f"cluster_{cid}"] = {
            "cluster_id": cid,
            "names": name_list,
            "canonical_name": entry.name,
            "confidence": round(entry.confidence, 4),
            "aliases": entry.aliases,
            "name_votes": {k: round(v, 4) for k, v in entry.name_votes.items()},
            "track_ids": sorted(entry.track_ids),
            "n_tracks": len(entry.track_ids),
            "shots_seen": entry.shots_seen,
            "n_shots": len(entry.shots_seen),
        }

        track_ids = sorted(
            entry.track_ids,
            key=lambda tid: len(face_data.get(str(tid), [])),
            reverse=True,
        )[:max_tracks_per_cluster]

        all_crops: List[List[np.ndarray]] = []
        for tid in track_ids:
            dets = face_data.get(str(tid), [])
            if len(dets) == 0:
                continue
            crops = _sample_track_crops(cap, dets, n_frames=n_frames_per_track)
            if crops:
                all_crops.append(crops)

        if not all_crops:
            canvas = _placeholder_crop(f"cluster {cid}")
        else:
            flat_crops = [c for track_crops in all_crops for c in track_crops]
            n_crops = len(flat_crops)
            if n_crops <= 4:
                grid_size = 2
            elif n_crops <= 9:
                grid_size = 3
            else:
                grid_size = 4
            total_slots = grid_size * grid_size
            flat_crops = flat_crops[:total_slots]
            while len(flat_crops) < total_slots:
                flat_crops.append(np.full((CROP_H, CROP_W, 3), _BG, dtype=np.uint8))

            grid_rows: List[np.ndarray] = []
            for r in range(grid_size):
                row_parts: List[np.ndarray] = []
                for c in range(grid_size):
                    if c > 0:
                        row_parts.append(
                            np.full((CROP_H, GAP, 3), _BG, dtype=np.uint8)
                        )
                    row_parts.append(flat_crops[r * grid_size + c])
                grid_rows.append(np.hstack(row_parts))

            row_w = grid_rows[0].shape[1]
            sep = np.full((GAP, row_w, 3), _BG, dtype=np.uint8)
            stacked: List[np.ndarray] = []
            for ri, row in enumerate(grid_rows):
                if ri > 0:
                    stacked.append(sep)
                stacked.append(row)
            canvas = np.vstack(stacked)

        out_img = os.path.join(save_folder, f"cluster_{cid}.png")
        cv2.imwrite(out_img, canvas)

    cap.release()

    json_path = os.path.join(save_folder, "bank_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return json_path


def visualize_bank(
    bank: "CharacterBank",
    person_tracks: dict,
    video_path: str,
    shot_idx: int,
    output_dir: str,
    n_frames_per_track: int = N_FRAMES_PER_TRACK,
    max_tracks_per_cluster: int = MAX_TRACKS_PER_CLUSTER,
    secondary_bank: Optional[List[Dict]] = None,
    voice_sample_log: Optional[Dict[int, List]] = None,
    max_voice_samples: int = 3,
) -> Optional[str]:
    """Render a contact-sheet image of the bank and optionally save voice clips."""
    entries = list(bank.all_entries())
    if not entries and not secondary_bank:
        return None

    # Print cluster summary.
    print(f"\n[Bank @ shot {shot_idx}]  {len(entries)} cluster(s)")
    for entry in sorted(entries, key=lambda e: e.cluster_id):
        canonical = entry.name if entry.name else "<unnamed>"
        shots = len(entry.shots_seen)
        tracks = len(entry.track_ids)
        if entry.name_votes:
            votes_str = "  votes: " + ", ".join(
                f"{n}={v:.2f}"
                for n, v in sorted(entry.name_votes.items(),
                                   key=lambda kv: kv[1], reverse=True)
            )
        else:
            votes_str = ""
        aliases_str = f"  aka: {', '.join(entry.aliases)}" if entry.aliases else ""
        print(
            f"  [{entry.cluster_id:3d}] {canonical:<24s} "
            f"conf={entry.confidence:.2f}  shots={shots}  tracks={tracks}"
            f"{aliases_str}{votes_str}"
        )
    if secondary_bank:
        print(f"  [Secondary bank: {len(secondary_bank)} entry(ies)]")
        for sb in secondary_bank:
            names = ", ".join(sb["names"]) if sb["names"] else "<unnamed>"
            print(f"    shot={sb['shot_id']}  names={names}  "
                  f"tracks={len(sb['track_ids'])}")
    print()

    try:
        import cv2
    except ImportError:
        return None

    if not os.path.isfile(video_path):
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    face_data = person_tracks.get("face", {})
    min_row_height = CROP_H + ROW_PAD * 2

    row_images: List[np.ndarray] = []

    for entry in sorted(entries, key=lambda e: e.cluster_id):
        n_lines = _label_line_count(entry.name_votes, entry.aliases)
        label_height = _LINE_STEP * n_lines + 16
        row_height = max(min_row_height, label_height)

        track_ids = sorted(
            entry.track_ids,
            key=lambda tid: len(face_data.get(str(tid), [])),
            reverse=True,
        )[:max_tracks_per_cluster]

        track_groups: List[List[np.ndarray]] = []
        for tid in track_ids:
            dets = face_data.get(str(tid), [])
            if len(dets) == 0:
                continue
            crops = _sample_track_crops(cap, dets, n_frames=n_frames_per_track)
            if crops:
                track_groups.append(crops)

        label_cell = _make_label_cell(
            entry.cluster_id, entry.name, entry.confidence,
            len(entry.track_ids), len(entry.shots_seen),
            entry.name_votes, entry.aliases, row_height,
        )

        crop_cells: List[np.ndarray] = []
        if not track_groups:
            placeholder = np.full((row_height, CROP_W, 3), _BG, dtype=np.uint8)
            placeholder[ROW_PAD:ROW_PAD + CROP_H, :] = _placeholder_crop()
            crop_cells.append(placeholder)
        else:
            for gi, group in enumerate(track_groups):
                if gi > 0:
                    sep = np.full((row_height, TRACK_SEP, 3), _SEP_COLOR, dtype=np.uint8)
                    crop_cells.append(sep)
                for crop in group:
                    cell = np.full((row_height, CROP_W + GAP, 3), _BG, dtype=np.uint8)
                    cell[ROW_PAD:ROW_PAD + CROP_H, GAP:GAP + CROP_W] = crop
                    crop_cells.append(cell)

        row = np.hstack([label_cell] + crop_cells)
        row_images.append(row)

    secondary_row_images: List[np.ndarray] = []
    if secondary_bank:
        for sb in secondary_bank:
            row_height = min_row_height
            track_ids = sorted(
                sb["track_ids"],
                key=lambda tid: len(face_data.get(str(tid), [])),
                reverse=True,
            )[:max_tracks_per_cluster]

            track_groups: List[List[np.ndarray]] = []
            for tid in track_ids:
                dets = face_data.get(str(tid), [])
                if len(dets) == 0:
                    continue
                crops = _sample_track_crops(cap, dets, n_frames=n_frames_per_track)
                if crops:
                    track_groups.append(crops)

            label_cell = _make_secondary_label_cell(
                sb["names"], sb["track_ids"], sb["shot_id"], row_height,
            )

            crop_cells: List[np.ndarray] = []
            if not track_groups:
                placeholder = np.full((row_height, CROP_W, 3), _BG, dtype=np.uint8)
                placeholder[ROW_PAD:ROW_PAD + CROP_H, :] = _placeholder_crop()
                crop_cells.append(placeholder)
            else:
                for gi, group in enumerate(track_groups):
                    if gi > 0:
                        sep = np.full((row_height, TRACK_SEP, 3), _SEP_COLOR, dtype=np.uint8)
                        crop_cells.append(sep)
                    for crop in group:
                        cell = np.full((row_height, CROP_W + GAP, 3), _BG, dtype=np.uint8)
                        cell[ROW_PAD:ROW_PAD + CROP_H, GAP:GAP + CROP_W] = crop
                        crop_cells.append(cell)

            row = np.hstack([label_cell] + crop_cells)
            secondary_row_images.append(row)

    cap.release()

    if not row_images and not secondary_row_images:
        return None

    all_rows = row_images + secondary_row_images

    max_w = max(r.shape[1] for r in all_rows)
    padded: List[np.ndarray] = []
    h_sep = np.full((2, max_w, 3), _SEP_COLOR, dtype=np.uint8)
    h_sec_sep = np.full((6, max_w, 3), _SECONDARY_COLOR, dtype=np.uint8)
    for i, row in enumerate(all_rows):
        if row.shape[1] < max_w:
            pad = np.full((row.shape[0], max_w - row.shape[1], 3), _BG, dtype=np.uint8)
            row = np.hstack([row, pad])
        padded.append(row)
        if i < len(all_rows) - 1:
            if i == len(row_images) - 1 and secondary_row_images:
                padded.append(h_sec_sep)
            else:
                padded.append(h_sep)

    grid = np.vstack(padded)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"bank_shot_{shot_idx:05d}.jpg")
    cv2.imwrite(out_path, grid, [cv2.IMWRITE_JPEG_QUALITY, 90])

    cluster_dir = os.path.join(output_dir, f"clusters_shot_{shot_idx:05d}")
    save_cluster_assets(
        bank=bank,
        person_tracks=person_tracks,
        video_path=video_path,
        save_folder=cluster_dir,
        n_frames_per_track=n_frames_per_track,
        max_tracks_per_cluster=max_tracks_per_cluster,
    )

    if voice_sample_log and os.path.isfile(video_path):
        _save_voice_samples_for_bank(
            video_path=video_path,
            voice_sample_log=voice_sample_log,
            output_dir=os.path.join(output_dir, f"voice_samples_shot_{shot_idx:05d}"),
            max_per_cluster=max_voice_samples,
        )

    return out_path


def _save_voice_samples_for_bank(
    video_path: str,
    voice_sample_log: Dict[int, List],
    output_dir: str,
    max_per_cluster: int = 3,
    min_duration: float = 0.5,
) -> None:
    """Save representative voice WAV clips from *voice_sample_log*."""
    import subprocess

    for cid, segments in voice_sample_log.items():
        valid = [(s, e) for s, e in segments if e - s >= min_duration]
        valid.sort(key=lambda se: se[1] - se[0], reverse=True)
        valid = valid[:max_per_cluster]
        if not valid:
            continue

        cid_dir = os.path.join(output_dir, f"cluster_{cid}")
        os.makedirs(cid_dir, exist_ok=True)

        for i, (start, end) in enumerate(valid):
            wav_path = os.path.join(cid_dir, f"sample_{i}.wav")
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start),
                "-to", str(end),
                "-i", video_path,
                "-vn", "-ar", "16000", "-ac", "1",
                "-acodec", "pcm_s16le",
                wav_path,
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
