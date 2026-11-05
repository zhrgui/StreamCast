"""
VLM-based name-to-face linking via a vLLM OpenAI-compatible API.

Extracts frames, overlays colored bounding boxes, sends to VLM, and parses
the JSON response mapping character names to box colors. Client-side only.
"""

import base64
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

DEFAULT_VLM_URL = "http://localhost:8000/v1"
DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_SAMPLE_FPS = 2.0

# Color palette (BGR for OpenCV). Shared with agent.py.
_COLOR_NAMES: List[str] = [
    "red", "green", "blue", "yellow", "orange",
    "cyan", "magenta", "brown", "pink", "teal",
]
_BOX_COLORS_BGR: List[Tuple[int, int, int]] = [
    (0,   0,   255),  # red
    (0,   200, 0),    # green
    (255, 0,   0),    # blue
    (0,   215, 255),  # yellow
    (0,   140, 255),  # orange
    (220, 220, 0),    # cyan
    (200, 0,   255),  # magenta
    (19,  69,  139),  # brown
    (180, 105, 255),  # pink
    (130, 130, 0),    # teal
]
_COLOR_NAME_TO_BGR: Dict[str, Tuple[int, int, int]] = dict(
    zip(_COLOR_NAMES, _BOX_COLORS_BGR)
)
_BOX_THICKNESS = 6
_BOX_EXPAND = 0.1


def _extract_frames(
    video_path: str,
    start: float,
    end: float,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
) -> Tuple[List, List[int]]:
    """Extract BGR frames from [start, end] at *sample_fps*. Returns (frames, frame_ids)."""
    try:
        import cv2
    except ImportError:
        return [], []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], []

    frames: List = []
    frame_ids: List[int] = []

    if sample_fps <= 0:
        cap.release()
        return frames, frame_ids

    t_start = float(start)
    t_end = float(max(end, start))
    step = 1.0 / float(sample_fps)
    sample_times: List[float] = []

    if t_end <= t_start:
        sample_times = [t_start]
    else:
        t = t_start
        while t < t_end:
            sample_times.append(t)
            t += step
        if not sample_times:
            sample_times = [t_start]

    for t in sample_times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        fid = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
            frame_ids.append(fid)

    cap.release()
    return frames, frame_ids


def _color_name_to_bgr(color_name: str) -> Tuple[int, int, int]:
    """Map a color name → BGR tuple."""
    return _COLOR_NAME_TO_BGR.get(color_name, (255, 255, 255))


def _clamp_box(x1: float, y1: float, x2: float, y2: float,
               w: int, h: int) -> Tuple[int, int, int, int]:
    """Clamp box coordinates to image dimensions."""
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


def _overlay_track_boxes(
    frames: List,
    frame_ids: List[int],
    track_color_map: Dict[str, str],
    track_detections: Dict[str, list],
) -> List:
    """Draw colored bounding boxes on frames. Returns new annotated frames."""
    try:
        import cv2
    except ImportError:
        return frames

    # Pre-index: fid → boxes
    fid_to_boxes: Dict[int, List[Tuple]] = {}
    for tid, color_name in track_color_map.items():
        detections = track_detections.get(str(tid), [])
        bgr = _color_name_to_bgr(color_name)
        for det in detections:
            if len(det) < 5:
                continue
            fid = int(det[0])
            box = (float(det[1]), float(det[2]), float(det[3]), float(det[4]), bgr, str(tid))
            fid_to_boxes.setdefault(fid, []).append(box)

    annotated = []
    for frame, fid in zip(frames, frame_ids):
        img = frame.copy()
        H, W = img.shape[:2]

        boxes = fid_to_boxes.get(fid, [])

        for (x1, y1, x2, y2, bgr, tid) in boxes:
            bx1, by1, bx2, by2 = _clamp_box(x1, y1, x2, y2, W, H)
            delta_x = (bx2 - bx1) * _BOX_EXPAND
            delta_y = (by2 - by1) * _BOX_EXPAND
            rx1 = int(max(bx1 - delta_x, 0))
            ry1 = int(max(by1 - delta_y, 0))
            rx2 = int(min(bx2 + delta_x, W))
            ry2 = int(min(by2 + delta_y, H))
            cv2.rectangle(img, (rx1, ry1), (rx2, ry2), bgr, _BOX_THICKNESS)
            # Draw track ID
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.8, min(H, W) / 600)
            thickness = max(2, int(font_scale * 2.5))
            text_y = max(ry1 - 8, int(font_scale * 25))
            cv2.putText(img, tid, (rx1, text_y), font, font_scale,
                        (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.putText(img, tid, (rx1, text_y), font, font_scale,
                        bgr, thickness, cv2.LINE_AA)

        annotated.append(img)
    return annotated


def _save_vlm_frames_montage(
    frames: List,
    frame_ids: List[int],
    out_path: str,
    thumb_h: int = 240,
    jpeg_quality: int = 88,
    vlm_raw_text: Optional[str] = None,
    name_to_color: Optional[Dict] = None,
    subtitles: Optional[List[Dict]] = None,
    start_offset: float = 0.0,
) -> None:
    """Save a JPEG montage: frame thumbnails + VLM output text panel."""
    try:
        import cv2
    except ImportError:
        return

    if not frames or len(frames) != len(frame_ids):
        return

    # Build frame strip
    _BG_DARK = (30, 30, 30)
    _BG_CELL = (45, 45, 45)
    pad_h = 26

    thumbs: List = []
    for frame, fid in zip(frames, frame_ids):
        if frame is None or frame.size == 0:
            continue
        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            continue
        scale = thumb_h / float(h)
        tw = max(1, int(round(w * scale)))
        small = cv2.resize(frame, (tw, thumb_h), interpolation=cv2.INTER_AREA)
        cell = np.full((thumb_h + pad_h, tw, 3), _BG_CELL, dtype=np.uint8)
        cell[pad_h:, :tw] = small
        cv2.putText(
            cell, f"fid {int(fid)}", (4, 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA,
        )
        thumbs.append(cell)

    if not thumbs:
        return

    # Pad to equal width and hstack
    max_tw = max(t.shape[1] for t in thumbs)
    row_cells: List = []
    for t in thumbs:
        if t.shape[1] < max_tw:
            pad = np.full((t.shape[0], max_tw - t.shape[1], 3), _BG_CELL, dtype=np.uint8)
            t = np.hstack([t, pad])
        row_cells.append(t)
    frame_strip = np.hstack(row_cells)
    total_w = frame_strip.shape[1]

    # Build text panel
    _LINE_H = 22
    _FONT = cv2.FONT_HERSHEY_SIMPLEX
    _FSCALE = 0.48
    _FTHICK = 1
    _MAX_LINES = 30
    _CHAR_W = 9   # approximate pixel width per character at _FSCALE

    chars_per_line = max(40, (total_w - 16) // _CHAR_W)

    text_lines: List[tuple] = []  # (text_str, color_bgr)

    # Subtitles
    if subtitles:
        text_lines.append(("Subtitles:", (180, 180, 100)))
        for sub in subtitles:
            t_s = max(0.0, sub.get("start", 0.0) - start_offset)
            t_e = max(0.0, sub.get("end", 0.0) - start_offset)
            text = sub.get("text", "").strip()
            speaker = sub.get("speaker") if sub.get("speaker") != "none" else "unknown"
            sub_line = f"  <{t_s:.2f}s> [{speaker}]: {text} <{t_e:.2f}s>"
            text_lines.append((sub_line, (180, 200, 220)))
        text_lines.append(("", (180, 180, 180)))

    # Parsed result
    if name_to_color:
        result_str = "Parsed: " + json.dumps(name_to_color, ensure_ascii=False)
        text_lines.append((result_str, (100, 220, 100)))
    else:
        text_lines.append(("Parsed: (no result)", (130, 130, 180)))

    text_lines.append(("", (180, 180, 180)))

    raw = vlm_raw_text or "(no response)"

    # Thinking trace
    if "</think>" in raw:
        think_text = raw.split("</think>")[0].strip()
        if think_text.startswith("<think>"):
            think_text = think_text[len("<think>"):].strip()
        if think_text:
            text_lines.append(("Thinking:", (100, 180, 220)))
            think_text = think_text.replace("\n", " ")
            words = think_text.split(" ")
            cur_line = "  "
            for word in words:
                if len(cur_line) + len(word) + 1 <= chars_per_line:
                    cur_line += ("" if cur_line.endswith(" ") else " ") + word
                else:
                    text_lines.append((cur_line, (160, 190, 210)))
                    cur_line = "  " + word
                    if len(text_lines) >= _MAX_LINES:
                        text_lines.append(("  …", (160, 160, 160)))
                        cur_line = ""
                        break
            if cur_line.strip():
                text_lines.append((cur_line, (160, 190, 210)))
            text_lines.append(("", (180, 180, 180)))

    # Raw answer
    answer = raw.split("</think>")[-1].strip() if "</think>" in raw else raw
    answer = answer.replace("\n", " ")

    # Word-wrap
    words = answer.split(" ")
    cur_line = "Answer: "
    for word in words:
        if len(cur_line) + len(word) + 1 <= chars_per_line:
            cur_line += ("" if cur_line.endswith(" ") else " ") + word
        else:
            text_lines.append((cur_line, (200, 200, 200)))
            cur_line = "  " + word
            if len(text_lines) >= _MAX_LINES:
                text_lines.append(("  …", (160, 160, 160)))
                cur_line = ""
                break
    if cur_line.strip():
        text_lines.append((cur_line, (200, 200, 200)))

    panel_h = _LINE_H * len(text_lines) + 14
    panel = np.full((panel_h, total_w, 3), _BG_DARK, dtype=np.uint8)
    y = _LINE_H
    for line_text, color in text_lines:
        if line_text:
            cv2.putText(panel, line_text, (8, y), _FONT, _FSCALE, color, _FTHICK, cv2.LINE_AA)
        y += _LINE_H

    # Stack and save
    grid = np.vstack([frame_strip, panel])
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(out_path, grid, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])


def _frame_to_data_url(frame, quality: int = 85) -> str:
    """Encode a BGR frame as a JPEG base64 data URL."""
    import cv2
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


_FACE_CROP_PAD = 0.15
_FACE_CROP_SIZE = 224


def _sample_face_crops_for_track(
    cap,
    detections,
    n_frames: int = 3,
) -> List:
    """Sample *n_frames* evenly-spaced face crops from a track.

    Returns list of BGR crop arrays (CROP_SIZE × CROP_SIZE).
    """
    import cv2
    dets = np.asarray(detections)
    if len(dets) == 0:
        return []
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    n = min(n_frames, len(dets))
    if n == 1:
        indices = np.array([(len(dets) - 1) // 2])
    else:
        indices = np.round(np.linspace(0, len(dets) - 1, n)).astype(int)
    crops = []
    for idx in indices:
        det = dets[int(idx)]
        fid = int(det[0])
        x1, y1, x2, y2 = float(det[1]), float(det[2]), float(det[3]), float(det[4])
        dx = (x2 - x1) * _FACE_CROP_PAD
        dy = (y2 - y1) * _FACE_CROP_PAD
        cx1 = max(0, int(x1 - dx))
        cy1 = max(0, int(y1 - dy))
        cx2 = min(W, int(x2 + dx))
        cy2 = min(H, int(y2 + dy))
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(fid))
        ret, frame = cap.read()
        if ret and cx2 > cx1 and cy2 > cy1:
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size > 0:
                crop = cv2.resize(crop, (_FACE_CROP_SIZE, _FACE_CROP_SIZE),
                                  interpolation=cv2.INTER_LINEAR)
                crops.append(crop)
                continue
        crops.append(np.zeros((_FACE_CROP_SIZE, _FACE_CROP_SIZE, 3), dtype=np.uint8))
    return crops


_FACE_QUALITY_PROMPT_LOOSE = (
    "You are shown {n} face crops sampled from the same person track in a video.\n\n"
    "Decide whether this track contains a usable FOREGROUND human face — i.e. a "
    "character who is part of the scene's action, NOT a background extra.\n\n"
    'Answer {{"clear": true}} if at least one crop shows a real, live-action human face '
    "that is reasonably sharp, at any angle where facial features are at least partly "
    "visible (including side profiles), AND the person appears to be a foreground or "
    "mid-ground character actively participating in the scene.\n\n"
    'Answer {{"clear": false}} if ALL crops fall into one of these categories:\n'
    "- background / audience / crowd face: the person is a background extra, audience "
    "member, bystander, or passerby who is not a focal character in the scene — even "
    "if the face is reasonably sized in the crop. Indicators include: the person is "
    "behind or far from the main subjects, out of the depth-of-field focus plane, "
    "part of a seated/standing crowd, or clearly not involved in the scene's dialogue "
    "or action\n"
    "- extreme profile (near-complete side view, ~85°+) where almost no facial features "
    "(eyes, nose, mouth) are visible\n"
    "- back-of-head view: the camera sees only the rear of the head (hair, scalp, "
    "back of neck) with no eyes, nose, or mouth visible at all\n"
    "- blurry, out of focus, or motion-blurred face where features are indistinct\n"
    "- not an actual living human face: cartoon, anime, drawing, painting, CGI-animated, "
    "sculpture, statue, doll, mannequin, mask, poster, photo-within-photo, or any "
    "stylized/illustrated face\n"
    "- tiny background face: the face occupies most of the crop only because of padding — "
    "the actual face region is very small (a distant bystander or crowd member)\n"
    "- no human face present (animal, object, background, etc.)\n\n"
    "Output ONLY the JSON object, nothing else."
)

_FACE_QUALITY_PROMPT_STRICT = (
    "You are shown {n} face crops sampled from the same person track in a video.\n\n"
    "Decide whether this track is suitable for a character identity bank — meaning "
    "the face must be clear and recognisable enough to reliably match to the same "
    "person later.\n\n"
    'Answer {{"clear": true}} only if at least one crop satisfies ALL of these:\n'
    "- Real, live-action human face (not cartoon, anime, CGI, painting, sculpture, "
    "mask, poster, or photo-within-photo)\n"
    "- Frontal, three-quarter, or side-profile view where enough facial features "
    "are visible to distinguish the person (e.g. eye, nose, jawline). Only reject "
    "extreme near-complete rear/back-of-head views (~85°+) where almost no "
    "features are visible\n"
    "- Sharp and in focus — facial features (eyes, eyebrows, nose, mouth) are "
    "clearly distinguishable, not motion-blurred or out-of-focus\n"
    "- Reasonably well-lit — not silhouetted, extremely dark, or washed-out to the "
    "point where features are lost\n"
    "- Not heavily occluded — the face is not mostly covered by hands, hair, objects, "
    "sunglasses, or other obstructions that hide key features\n"
    "- Sufficiently large — the face is a prominent subject in the crop, not a tiny "
    "distant bystander or crowd member\n\n"
    'Answer {{"clear": false}} if NONE of the crops meet all the criteria above.\n\n'
    "Output ONLY the JSON object, nothing else."
)


def _save_face_filter_viz(
    crops: List,
    out_path: str,
    track_id: str,
    raw_text: Optional[str],
    parsed: Optional[Dict],
    keep: bool,
) -> None:
    """Save a JPEG with parsed JSON + raw text header and face crops."""
    try:
        import cv2
    except ImportError:
        return

    if not crops:
        return

    gap = 6
    pad = 8
    line_h = 18
    bg_keep = (42, 72, 42)
    bg_drop = (52, 42, 92)
    bg = bg_keep if keep else bg_drop

    def _trunc(s: str, n: int = 118) -> str:
        s = s.strip()
        return s if len(s) <= n else s[: n - 3] + "..."

    lines: List[str] = [
        _trunc(f"track_id={track_id}   keep={keep}", 118),
    ]
    if parsed is not None:
        lines.append(_trunc("parsed: " + json.dumps(parsed, ensure_ascii=False), 118))
    else:
        lines.append("parsed: (none)")
    if raw_text is not None:
        rt = raw_text.replace("\n", " ").strip()
        if len(rt) > 900:
            rt = rt[:897] + "..."
        w = 118
        first = "raw: "
        rest = first + rt
        while rest:
            lines.append(rest[:w])
            rest = rest[w:].lstrip()
            if len(lines) > 14:
                lines[-1] = lines[-1][: w - 3] + "..."
                break
    else:
        lines.append("raw: (none)")

    label_h = pad * 2 + line_h * len(lines) + 6
    h_crop = int(crops[0].shape[0])
    parts: List = []
    for i, c in enumerate(crops):
        parts.append(c)
        if i < len(crops) - 1:
            parts.append(np.full((h_crop, gap, 3), bg, dtype=np.uint8))
    strip = np.hstack(parts)
    w = strip.shape[1]
    banner = np.full((label_h, w, 3), bg, dtype=np.uint8)
    y = pad + 14
    for line in lines:
        cv2.putText(
            banner,
            line,
            (pad, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        y += line_h

    grid = np.vstack([banner, strip])
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(out_path, grid, [cv2.IMWRITE_JPEG_QUALITY, 90])


def _is_track_face_clear(
    crops: List,
    vlm_url: str,
    model_id: str,
    max_new_tokens: int = 32,
    timeout: int = 30,
    quality_prompt: Optional[str] = None,
) -> Tuple[bool, Optional[str], Optional[Dict]]:
    """Ask VLM if crops show a clear face. Returns (keep, raw_text, parsed_json)."""
    if not crops:
        return True, None, None
    content: List[Dict] = []
    for crop in crops:
        content.append({
            "type": "image_url",
            "image_url": {"url": _frame_to_data_url(crop)},
        })
    tpl = quality_prompt or _FACE_QUALITY_PROMPT_LOOSE
    prompt = tpl.format(n=len(crops))
    content.append({"type": "text", "text": prompt})
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_new_tokens,
        "temperature": 0.0,
    }
    endpoint = vlm_url.rstrip("/") + "/chat/completions"
    try:
        resp = requests.post(endpoint, json=payload, timeout=timeout)
        resp.raise_for_status()
        output_text = resp.json()["choices"][0]["message"]["content"]
    except Exception:
        return True, None, None  # conservative: keep on failure
    parsed = _parse_json_from_text(output_text)
    if parsed is None:
        return True, output_text, None
    keep = bool(parsed.get("clear", True))
    return keep, output_text, parsed


def filter_clear_face_tracks(
    face_tracks: List,
    track_detections: Dict[str, list],
    video_path: str,
    vlm_url: str = DEFAULT_VLM_URL,
    model_id: str = DEFAULT_MODEL_ID,
    n_frames: int = 5,
    max_new_tokens: int = 32,
    timeout: int = 30,
    face_filter_vis_dir: Optional[str] = None,
    face_filter_vis_tag: str = "shot",
    quality_prompt: Optional[str] = None,
) -> List:
    """Filter face tracks by VLM quality check. Keeps tracks on failure (conservative)."""
    try:
        import cv2
    except ImportError:
        return face_tracks

    if not face_tracks or not os.path.isfile(video_path):
        return face_tracks

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return face_tracks

    kept = []
    for ft in face_tracks:
        dets = track_detections.get(str(ft.track_id), [])
        if len(dets) == 0:
            kept.append(ft)
            continue
        crops = _sample_face_crops_for_track(cap, dets, n_frames=n_frames)
        clear, raw_out, parsed = _is_track_face_clear(
            crops, vlm_url, model_id,
            max_new_tokens=max_new_tokens, timeout=timeout,
            quality_prompt=quality_prompt,
        )
        if face_filter_vis_dir and crops:
            safe_tag = re.sub(r"[^\w\-.]", "_", str(face_filter_vis_tag))[:80] or "shot"
            safe_tid = re.sub(r"[^\w\-.]", "_", str(ft.track_id))[:120] or "tid"
            out_path = os.path.join(
                face_filter_vis_dir,
                f"face_filter_{safe_tag}_{safe_tid}.jpg",
            )
            _save_face_filter_viz(
                crops, out_path, str(ft.track_id), raw_out, parsed, clear,
            )
        if clear:
            kept.append(ft)

    cap.release()
    return kept


def _parse_json_from_text(text):
    if text is None:
        return None

    s = text.strip()

    # Try to extract a ```json ... ``` block
    m = re.search(r"```(?:json)?\s*\n([\s\S]*?)(?:\n```|$)", s, flags=re.IGNORECASE)
    candidate = m.group(1).strip() if m else s

    # Try loading JSON
    obj = _loads_json_maybe_nested(candidate)
    if isinstance(obj, dict):
        return obj

    # Fallback: last {...} span
    blob = _last_braced_object(candidate) or _last_braced_object(s)
    if blob:
        obj = _loads_json_maybe_nested(blob)
        if isinstance(obj, dict):
            return obj

    return None

def _loads_json_maybe_nested(x):
    if x is None:
        return None
    y = x.strip()

    # Strip extra backticks/labels
    y = y.strip("`").strip()

    try:
        obj = json.loads(y)
    except Exception:
        return None

    # Handle double-encoded JSON
    if isinstance(obj, str):
        z = obj.strip()
        if (z.startswith("{") and z.endswith("}")) or (z.startswith("[") and z.endswith("]")):
            try:
                return json.loads(z)
            except Exception:
                return obj

    return obj

def _last_braced_object(s):
    """Return the last {...} substring via brace-stack scan."""
    stack = []
    spans = []
    for i, ch in enumerate(s):
        if ch == "{":
            stack.append(i)
        elif ch == "}" and stack:
            start = stack.pop()
            spans.append((start, i + 1))
    if not spans:
        return None
    start, end = spans[-1]
    return s[start:end].strip()


def _build_prompt(
    candidate_names: List[str],
    color_list: List[str],
    subtitles: List[Dict],
    start_offset: float,
) -> str:
    lines = []
    for sub in subtitles:
        t_start = max(0.0, sub.get("start", 0.0) - start_offset)
        t_end = max(0.0, sub.get("end", 0.0) - start_offset)
        text = sub.get("text", "").strip()
        speaker = sub.get("speaker", "unknown")
        lines.append(f"<{t_start:.2f}s> [{speaker}]: {text} <{t_end:.2f}s>")
    timestamped = "\n".join(lines)

    return (
        "Match characters to bounding box colors using the subtitles and video.\n\n"
        f"Candidates:\n{', '.join(candidate_names)}\n\n"
        f"Colors:\n{', '.join(color_list)}\n\n"
        f"Subtitles:\n{timestamped}\n\n"
        "Question:\n"
        "Which character is each color?\n\n"
        "After </think>, output ONLY a JSON object:\n"
        '{"character_name": "color"} or {"character_name": "unknown"}'
    )

def run_vlm_name_linking(
    video_path: str,
    candidate_names: List[str],
    track_color_map: Dict[str, str],
    subtitles: List[Dict],
    start_offset: float = 0.0,
    end_offset: Optional[float] = None,
    vlm_url: str = DEFAULT_VLM_URL,
    model_id: str = DEFAULT_MODEL_ID,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
    max_new_tokens: int = 256,
    timeout: int = 60,
    track_detections: Optional[Dict[str, list]] = None,
    vlm_frames_vis_dir: Optional[str] = None,
    vlm_frames_vis_tag: str = "shot",
) -> Optional[Dict[str, str]]:
    """Call the VLM API to link character names to bounding-box colors.

    Returns {name: color} for resolved characters, or None on failure.
    """
    if not candidate_names or not os.path.isfile(video_path):
        return None

    color_list = sorted(set(track_color_map.values()))
    if not color_list:
        return None

    # Shot time window
    t_start = start_offset
    t_end = end_offset if end_offset is not None else start_offset + 30.0

    # Extract frames
    frames, frame_ids = _extract_frames(video_path, t_start, t_end, sample_fps=sample_fps)
    if not frames:
        return None

    # Overlay bounding boxes
    if track_detections:
        frames = _overlay_track_boxes(frames, frame_ids, track_color_map, track_detections)

    # Build multimodal content
    content: List[Dict] = []
    for frame in frames:
        content.append({
            "type": "image_url",
            "image_url": {"url": _frame_to_data_url(frame)},
        })

    prompt = _build_prompt(candidate_names, color_list, subtitles, start_offset)
    content.append({"type": "text", "text": prompt})

    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_new_tokens,
        "temperature": 0.0,
    }

    endpoint = vlm_url.rstrip("/") + "/chat/completions"
    output_text: Optional[str] = None
    result: Optional[Dict[str, str]] = None
    try:
        resp = requests.post(endpoint, json=payload, timeout=timeout)
        resp.raise_for_status()
        output_text = resp.json()["choices"][0]["message"]["content"]
        parsed = _parse_json_from_text(output_text.split("</think>")[-1])
        if parsed and isinstance(parsed, dict):
            result = {
                name: color
                for name, color in parsed.items()
                if name in candidate_names and color in color_list
            }
    except Exception:
        pass

    # Save visualization
    if vlm_frames_vis_dir:
        import cv2 as _cv2
        n_vis = 10
        n = len(frames)
        if n <= n_vis:
            vis_frames, vis_ids = frames, frame_ids
        else:
            idxs = [int(round(i * (n - 1) / (n_vis - 1))) for i in range(n_vis)]
            vis_frames = [frames[i] for i in idxs]
            vis_ids = [frame_ids[i] for i in idxs]
        safe = re.sub(r"[^\w\-.]", "_", str(vlm_frames_vis_tag))[:120] or "shot"
        montage_path = os.path.join(vlm_frames_vis_dir, f"vlm_frames_{safe}.jpg")
        _save_vlm_frames_montage(
            vis_frames, vis_ids, montage_path,
            vlm_raw_text=output_text,
            name_to_color=result,
            subtitles=subtitles,
            start_offset=start_offset,
        )

    return result

def link_names_to_clusters(
    name_to_color: Dict[str, str],
    cluster_to_color: Dict[int, str],
) -> Dict[str, int]:
    """Resolve name → cluster_id via VLM name→color and cluster→color maps."""
    color_to_cluster = {color: cid for cid, color in cluster_to_color.items()}
    result: Dict[str, int] = {}
    for name, color in name_to_color.items():
        cid = color_to_cluster.get(color)
        if cid is not None:
            result[name] = cid
    return result
