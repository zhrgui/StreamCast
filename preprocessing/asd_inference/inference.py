"""Active Speaker Detection inference over per-shot face tracks.

For every (imdbid, track_id) in --track_file:
    - loads the face boxes from <track_dir>/<imdbid>/<tid>.pkl,
    - decodes the matching frames from <video_dir>/<imdbid>.mp4 and the audio
      from --pyavi_dir (mono 16 kHz wav),
    - runs the TalkNet ASD model over multi-duration audio/visual windows
      and aggregates a per-track time-series of speaking scores,
    - writes one pickle per track to <output_dir>/<imdbid>/<tid>.pkl, with
      {obj_id: {"mean_score": float, "scores": [{"t": s, "score": f}, ...],
                "start_time": float, "fps": float}}.

The video must be decodable by OpenCV; the audio must already be 16 kHz mono.
"""

import argparse
import json
import math
import os
import pickle
from collections import defaultdict

import cv2
import numpy as np
import python_speech_features
import torch
import torch.nn.functional as F
from scipy import signal
from scipy.io import wavfile
from tqdm import tqdm

from talkNet import talkNet

def build_proc_track(track):
    """Replicate crop_video's medfilt smoothing to produce proc_track (s/x/y arrays).
    Returns dict {'x','y','s'} aligned with track['frame'].
    """
    dets = {'x': [], 'y': [], 's': []}
    for det in track['bbox']:
        # det = [x1,y1,x2,y2]
        dets['s'].append(max((det[3] - det[1]), (det[2] - det[0])) / 2)
        dets['y'].append((det[1] + det[3]) / 2)
        dets['x'].append((det[0] + det[2]) / 2)
    dets['s'] = signal.medfilt(dets['s'], kernel_size=13).tolist()
    dets['x'] = signal.medfilt(dets['x'], kernel_size=13).tolist()
    dets['y'] = signal.medfilt(dets['y'], kernel_size=13).tolist()
    return dets

def _read_audio_16k_mono(audio_wav_path: str):
    sr, audio = wavfile.read(audio_wav_path)
    if audio is None or len(audio) == 0:
        raise RuntimeError(f"Empty audio: {audio_wav_path}")
    # stereo -> mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    # normalize dtype for mfcc (python_speech_features accepts int or float)
    # but keep in numpy for speed
    if sr != 16000:
        raise RuntimeError(f"Expected 16kHz wav for TalkNet MFCC, got sr={sr}. "
            f"Resample offline once (ffmpeg -ar 16000). File: {audio_wav_path}")
    return sr, audio

def _ceil_int(x: float) -> int:
    return int(math.ceil(x - 1e-9))

def _floor_int(x: float) -> int:
    return int(math.floor(x + 1e-9))

def _mean_score_windowed(seq, radius=2):
    if seq is None:
        return 0.0
    arr = np.asarray(seq, dtype=float).reshape(-1)
    n = arr.shape[0]
    if n == 0:
        return 0.0
    if n <= 2 * radius + 1:
        return float(np.mean(arr))
    vals = []
    for i in range(n):
        s = max(i - radius, 0)
        e = min(i + radius + 1, n)
        vals.append(float(np.mean(arr[s:e])))
    return float(np.mean(vals))

def _scores_with_timestamps(scores, base_k0, out_fps):
    if scores is None or base_k0 is None:
        return []
    arr = np.asarray(scores, dtype=float).reshape(-1)
    if arr.size == 0:
        return []
    dt = 1.0 / float(out_fps)
    start_t = float(base_k0) * dt
    return [{"t": round(start_t + i * dt, 6), "score": float(arr[i])} for i in range(arr.size)]

class ASDPreprocDataset(object):
    def __init__(self, tracks, args, out_fps=25.0):
        self.tracks = tracks
        self.args = args
        self.out_fps = float(out_fps)
        self.featsA = []
        self.featsV = []
        self.lengths = []
        self.base_k0_per_track = []
        self._prepare()

    def __len__(self):
        return len(self.tracks)

    def __iter__(self):
        for i in range(len(self.tracks)):
            yield i, self.featsA[i], self.featsV[i]

    def _prepare(self):
        cap = cv2.VideoCapture(self.args.videoFilePath)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.args.videoFilePath}")
        src_fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not (src_fps > 0):
            cap.release()
            raise RuntimeError(f"Invalid source FPS from cv2: {src_fps}")

        src_dt = 1.0 / src_fps
        out_dt = 1.0 / float(self.out_fps)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            cap.release()
            raise RuntimeError("CUDA GPU required for in-memory ASD acceleration")

        try:
            from torchvision.ops import roi_align
        except Exception as e:
            cap.release()
            raise RuntimeError("torchvision.ops.roi_align is required for GPU cropping") from e

        sr, full_audio = _read_audio_16k_mono(self.args.audio_wav_path)

        proc_tracks = []
        frames_list_per_track = []
        track_start_frame = []
        track_end_frame = []

        for tr in self.tracks:
            frames = tr["frame"]
            bboxes = tr["bbox"]
            frames = frames.tolist() if hasattr(frames, "tolist") else list(frames)
            if not frames:
                proc_tracks.append(None)
                frames_list_per_track.append([])
                track_start_frame.append(None)
                track_end_frame.append(None)
                continue
            dets = build_proc_track({"frame": frames, "bbox": bboxes})
            proc_tracks.append(dets)
            frames_list_per_track.append(frames)
            track_start_frame.append(int(frames[0]))
            track_end_frame.append(int(frames[-1]))

        sampling_plan = defaultdict(list)
        out_len_per_track = [0] * len(self.tracks)

        for tidx in range(len(self.tracks)):
            s_f = track_start_frame[tidx]
            e_f = track_end_frame[tidx]
            if s_f is None or e_f is None:
                continue

            t_start = s_f * src_dt
            t_end = e_f * src_dt
            if t_end < t_start:
                continue

            k0 = _ceil_int(t_start * self.out_fps)
            k1 = _floor_int(t_end * self.out_fps)
            if k1 < k0:
                continue

            track_frames = frames_list_per_track[tidx]
            dets = proc_tracks[tidx]
            if dets is None:
                continue

            p = 0
            nF = len(track_frames)

            for k in range(k0, k1 + 1):
                t_out = k * out_dt
                f_src = int(round(t_out * src_fps))
                if f_src < s_f:
                    f_src = s_f
                elif f_src > e_f:
                    f_src = e_f

                while p < nF and track_frames[p] < f_src:
                    p += 1
                if p <= 0:
                    lidx = 0
                elif p >= nF:
                    lidx = nF - 1
                else:
                    prev_f = track_frames[p - 1]
                    next_f = track_frames[p]
                    lidx = (p - 1) if abs(prev_f - f_src) <= abs(next_f - f_src) else p

                sampling_plan[f_src].append((tidx, lidx, k))

            out_len_per_track[tidx] = (k1 - k0 + 1)

        if sum(out_len_per_track) == 0:
            cap.release()
            self.featsA = [None for _ in range(len(self.tracks))]
            self.featsV = [None for _ in range(len(self.tracks))]
            self.lengths = [0.0 for _ in range(len(self.tracks))]
            self.base_k0_per_track = [None] * len(self.tracks)
            return

        needed_frames = sorted(sampling_plan.keys())
        faces_mem = {i: [None] * out_len_per_track[i] for i in range(len(self.tracks))}

        base_k0_per_track = [None] * len(self.tracks)
        for tidx in range(len(self.tracks)):
            s_f = track_start_frame[tidx]
            e_f = track_end_frame[tidx]
            if s_f is None or e_f is None or out_len_per_track[tidx] == 0:
                continue
            t_start = s_f * src_dt
            k0 = _ceil_int(t_start * self.out_fps)
            base_k0_per_track[tidx] = k0

        if not needed_frames:
            cap.release()
            self.featsA = [None for _ in range(len(self.tracks))]
            self.featsV = [None for _ in range(len(self.tracks))]
            self.lengths = [0.0 for _ in range(len(self.tracks))]
            self.base_k0_per_track = base_k0_per_track
            return

        min_f = int(needed_frames[0])
        max_f = int(needed_frames[-1])
        cap.set(cv2.CAP_PROP_POS_FRAMES, min_f)

        for f_src in range(min_f, max_f + 1):
            ret, image = cap.read()
            if not ret:
                break

            entries = sampling_plan.get(f_src, [])
            if not entries:
                continue

            img_t = torch.from_numpy(image).to(device=device, dtype=torch.float32)
            img_t = img_t.permute(2, 0, 1).unsqueeze(0) / 255.0

            cs = float(self.args.cropScale)
            pad_used = 0
            for (tidx, lidx, _k) in entries:
                dets = proc_tracks[tidx]
                if dets is None:
                    continue
                bs = float(dets["s"][lidx])
                pad_used = max(pad_used, int(bs * (1 + 2 * cs)))

            if pad_used > 0:
                img_t = F.pad(
                    img_t,
                    (pad_used, pad_used, pad_used, pad_used),
                    mode="constant",
                    value=110.0 / 255.0
                )
            _, _, Hp, Wp = img_t.shape

            rois = []
            owners = []
            for (tidx, lidx, k) in entries:
                dets = proc_tracks[tidx]
                if dets is None:
                    continue
                bs = float(dets["s"][lidx])
                my = float(dets["y"][lidx]) + pad_used
                mx = float(dets["x"][lidx]) + pad_used

                y1 = my - bs
                y2 = my + bs * (1 + 2 * cs)
                x1 = mx - bs * (1 + cs)
                x2 = mx + bs * (1 + cs)

                x1 = max(0.0, min(x1, Wp - 1.0))
                x2 = max(0.0, min(x2, Wp - 1.0))
                y1 = max(0.0, min(y1, Hp - 1.0))
                y2 = max(0.0, min(y2, Hp - 1.0))
                if x2 <= x1 or y2 <= y1:
                    continue

                rois.append([0.0, x1, y1, x2, y2])
                owners.append((tidx, k))

            if not rois:
                continue

            rois_t = torch.tensor(rois, device=device, dtype=torch.float32)
            crops = roi_align(
                img_t, rois_t,
                output_size=(224, 224),
                spatial_scale=1.0,
                sampling_ratio=-1,
                aligned=True
            )
            crops = (crops.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
            crops = crops.permute(0, 2, 3, 1).contiguous().cpu().numpy()

            for j in range(crops.shape[0]):
                tidx, k = owners[j]
                base_k0 = base_k0_per_track[tidx]
                if base_k0 is None:
                    continue
                pos = int(k - base_k0)
                if pos < 0 or pos >= out_len_per_track[tidx]:
                    continue
                faces_mem[tidx][pos] = crops[j]
        cap.release()

        for tidx in range(len(self.tracks)):
            seq = faces_mem[tidx]
            if not seq or out_len_per_track[tidx] == 0:
                continue

            first = None
            for i, x in enumerate(seq):
                if x is not None:
                    first = i
                    break
            if first is None:
                faces_mem[tidx] = []
                continue

            last = seq[first]
            for i in range(first):
                seq[i] = last
            for i in range(first, len(seq)):
                if seq[i] is None:
                    seq[i] = last
                else:
                    last = seq[i]

            last = seq[-1]
            for i in range(len(seq) - 1, -1, -1):
                if seq[i] is None:
                    seq[i] = last
                else:
                    last = seq[i]

            faces_mem[tidx] = seq

        featsA = []
        featsV = []
        lens = []

        for tidx in range(len(self.tracks)):
            v_seq = faces_mem[tidx]
            if not v_seq:
                featsA.append(None); featsV.append(None); lens.append(0.0)
                continue

            base_k0 = base_k0_per_track[tidx]
            if base_k0 is None:
                featsA.append(None); featsV.append(None); lens.append(0.0)
                continue
            t_s = float(base_k0) * out_dt
            t_e = float(base_k0 + len(v_seq) - 1) * out_dt
            t_e = t_e + out_dt

            a0 = int(round(t_s * sr))
            a1 = int(round(t_e * sr))
            a0 = max(0, min(a0, full_audio.shape[0]))
            a1 = max(a0 + 1, min(a1, full_audio.shape[0]))

            a_seg = full_audio[a0:a1]
            if a_seg.size == 0:
                featsA.append(None); featsV.append(None); lens.append(0.0)
                continue

            v_arr = np.empty((len(v_seq), 112, 112), dtype=np.uint8)
            for i_f, face in enumerate(v_seq):
                g = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                v_arr[i_f] = g[56:168, 56:168]

            a_feat = python_speech_features.mfcc(a_seg, sr, numcep=13, winlen=0.025, winstep=0.010)
            length = min((a_feat.shape[0] - a_feat.shape[0] % 4) / 100.0, v_arr.shape[0] / self.out_fps)
            if length <= 0:
                featsA.append(None); featsV.append(None); lens.append(0.0)
                continue

            a_feat = a_feat[:int(round(length * 100)), :]
            v_arr = v_arr[:int(round(length * self.out_fps)), :, :]

            featsA.append(a_feat)
            featsV.append(v_arr)
            lens.append(length)

        self.featsA = featsA
        self.featsV = featsV
        self.lengths = lens
        self.base_k0_per_track = base_k0_per_track

class ASDRunner(object):
    def __init__(self, args):
        self.model = talkNet()
        self.model.loadParameters(args.pretrainModel)
        self.model.eval()
        self.use_dp = (torch.cuda.device_count() > 1) and (os.environ.get("ASD_FORCE_DP", "1") != "0")
        self.dp_model = torch.nn.DataParallel(self.model) if self.use_dp else None
        self.asd_batch = max(1, int(getattr(args, "asdBatch", 64)))

    @torch.no_grad()
    def infer(self, featsA, featsV, out_fps=25.0):

        durationU = [1, 2, 3, 4, 5, 6]
        Bglob = self.asd_batch
        perDurScores = {d: [None] * len(featsA) for d in durationU}

        for duration in durationU:
            winA = int(duration * 100)
            winV = int(duration * out_fps)

            n_full = []
            for i in range(len(featsA)):
                a_feat = featsA[i]; v_arr = featsV[i]
                if a_feat is None or v_arr is None:
                    n_full.append(0)
                else:
                    n_full.append(int(min(a_feat.shape[0] // winA, v_arr.shape[0] // winV)))

            scores_by_track = [[] for _ in range(len(featsA))]
            pos = [0] * len(featsA)
            remaining = sum(n_full)

            while remaining > 0:
                batchA = []
                batchV = []
                owners = []

                for i in range(len(featsA)):
                    if pos[i] >= n_full[i]:
                        continue
                    take = min(n_full[i] - pos[i], max(1, Bglob - len(batchA)))
                    a_feat = featsA[i]; v_arr = featsV[i]
                    for k in range(take):
                        idx = pos[i] + k
                        batchA.append(a_feat[idx * winA:(idx + 1) * winA, :])
                        batchV.append(v_arr[idx * winV:(idx + 1) * winV, :, :])
                        owners.append(i)
                    pos[i] += take
                    if len(batchA) >= Bglob:
                        break

                if not batchA:
                    break

                if self.use_dp:
                    inputA = torch.from_numpy(np.stack(batchA, axis=0).astype(np.float32))
                    inputV = torch.from_numpy(np.stack(batchV, axis=0).astype(np.float32))
                    out = self.dp_model(inputA, inputV)
                else:
                    inputA = torch.FloatTensor(np.stack(batchA, axis=0)).cuda()
                    inputV = torch.FloatTensor(np.stack(batchV, axis=0)).cuda()
                    embedA = self.model.model.forward_audio_frontend(inputA)
                    embedV = self.model.model.forward_visual_frontend(inputV)
                    embedA, embedV = self.model.model.forward_cross_attention(embedA, embedV)
                    out = self.model.model.forward_audio_visual_backend(embedA, embedV)

                scoreBatch = self.model.lossAV.forward(out, labels=None)
                scoreBatch = np.asarray(scoreBatch)

                nW = len(owners)
                per_len = int(scoreBatch.shape[0] // nW) if nW > 0 else 0

                if per_len <= 0:
                    for j, iowner in enumerate(owners):
                        val = float(scoreBatch[j]) if j < scoreBatch.shape[0] else 0.0
                        scores_by_track[iowner].append(val)
                else:
                    for j, iowner in enumerate(owners):
                        start = j * per_len
                        end = start + per_len
                        scores_by_track[iowner].extend(scoreBatch[start:end].tolist())

                remaining -= len(batchA)

            for i in range(len(featsA)):
                a_feat = featsA[i]; v_arr = featsV[i]
                if a_feat is None or v_arr is None:
                    continue
                usedA = n_full[i] * winA
                usedV = n_full[i] * winV
                a_tail = a_feat[usedA: usedA + winA, :]
                v_tail = v_arr[usedV: usedV + winV, :, :]
                if a_tail.shape[0] > 0 and v_tail.shape[0] > 0:
                    if self.use_dp:
                        inputA = torch.from_numpy(a_tail.astype(np.float32)).unsqueeze(0)
                        inputV = torch.from_numpy(v_tail.astype(np.float32)).unsqueeze(0)
                        out = self.dp_model(inputA, inputV)
                    else:
                        inputA = torch.FloatTensor(a_tail).unsqueeze(0).cuda()
                        inputV = torch.FloatTensor(v_tail).unsqueeze(0).cuda()
                        embedA = self.model.model.forward_audio_frontend(inputA)
                        embedV = self.model.model.forward_visual_frontend(inputV)
                        embedA, embedV = self.model.model.forward_cross_attention(embedA, embedV)
                        out = self.model.model.forward_audio_visual_backend(embedA, embedV)

                    score_tail = self.model.lossAV.forward(out, labels=None)
                    try:
                        scores_by_track[i].extend(np.asarray(score_tail).tolist())
                    except Exception:
                        scores_by_track[i].append(float(score_tail))

            perDurScores[duration] = scores_by_track

        weights = {1: 3, 2: 3, 3: 2, 4: 1, 5: 1, 6: 1}

        allScores = []
        for i in range(len(featsA)):
            seqs = [perDurScores[d][i] for d in durationU]
            nonempty = [np.array(x, dtype=float) for x in seqs if isinstance(x, (list, tuple)) and len(x) > 0]
            if not nonempty:
                allScores.append([])
                continue
            Lmin = min(arr.shape[0] for arr in nonempty)
            if Lmin <= 0:
                allScores.append([])
                continue

            stack_list = []
            for d, x in zip(durationU, seqs):
                if not (isinstance(x, (list, tuple)) and len(x) > 0):
                    continue
                w = int(weights.get(d, 1))
                if w <= 0:
                    continue
                arrd = np.array(x[:Lmin], dtype=float)
                for _ in range(w):
                    stack_list.append(arrd)

            if not stack_list:
                allScores.append([])
                continue

            arr = np.stack(stack_list, axis=0)
            allScores.append(np.round(arr.mean(axis=0), 1).astype(float))
        return allScores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track_file", type=str, required=True)
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--pyavi_dir", type=str, default="", help="Directory of extracted audio; if empty, audio is looked up in --video_dir")
    parser.add_argument("--track_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--pretrainModel", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrain_TalkSet.model"), help="Path for the pretrained TalkNet model")
    parser.add_argument("--cropScale", type=float, default=0.40, help="Scale bounding box")
    parser.add_argument("--asdBatch", type=int, default=64, help="Batch size for ASD window inference")
    parser.add_argument("--shard_id", type=int, default=0, help="Shard index for multi-GPU parallelism")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    args = parser.parse_args()

    with open(args.track_file, 'r') as f:
        imdbid_track_id = json.load(f)

    pending_tids_by_video = {}
    print("[INFO] Collecting tracks to process.")
    for imdbid, tid_lst in imdbid_track_id.items():
        pending_tids = []
        for tid in tid_lst:
            output_path = os.path.join(args.output_dir, imdbid, f"{str(tid)}.pkl")
            if os.path.exists(output_path):
                continue

            track_file_path = os.path.join(args.track_dir, imdbid, f"{str(tid)}.pkl")
            if not os.path.exists(track_file_path):
                print(f"[WARNING] Track file does not exist: {track_file_path}")
                continue
            pending_tids.append(tid)

        if pending_tids:
            pending_tids_by_video[imdbid] = pending_tids

    if args.num_shards > 1:
        all_imdbids = sorted(pending_tids_by_video.keys())
        shard_imdbids = all_imdbids[args.shard_id::args.num_shards]
        pending_tids_by_video = {k: pending_tids_by_video[k] for k in shard_imdbids}
        print(f"[INFO] Shard {args.shard_id}/{args.num_shards}: {len(shard_imdbids)} videos assigned.")

    total_pending = sum(len(tids) for tids in pending_tids_by_video.values())
    print(f"[INFO] Preprocessing {len(pending_tids_by_video)} videos / {total_pending} track files.")
    runner = ASDRunner(args)
    progress_bar = tqdm(total=total_pending, desc="Processing track files")

    try:
        for imdbid, pending_tids in pending_tids_by_video.items():
            video_path = os.path.join(args.video_dir, f"{imdbid}.mp4")
            if not os.path.exists(video_path):
                for ext in ['.mkv', '.avi', '.mov']:
                    alt_path = os.path.join(args.video_dir, f"{imdbid}{ext}")
                    if os.path.exists(alt_path):
                        video_path = alt_path
                        break
                else:
                    print(f"[WARNING] Video file not found for {imdbid}")
                    progress_bar.update(len(pending_tids))
                    continue

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"[WARNING] Failed to open video: {video_path}")
                progress_bar.update(len(pending_tids))
                continue
            src_fps = float(cap.get(cv2.CAP_PROP_FPS))
            cap.release()
            if not (src_fps > 0):
                print(f"[WARNING] Invalid source FPS from cv2: {src_fps} for {imdbid}")
                progress_bar.update(len(pending_tids))
                continue
            out_fps = 25.0

            audio_candidates = []
            if args.pyavi_dir:
                audio_candidates.append(os.path.join(args.pyavi_dir, f"{imdbid}.wav"))
                audio_candidates.append(os.path.join(args.pyavi_dir, imdbid, "audio.wav"))
            else:
                audio_candidates.append(os.path.join(args.video_dir, f"{imdbid}.wav"))
                audio_candidates.append(os.path.join(args.video_dir, imdbid, "audio.wav"))

            audio_wav_path = None
            for p in audio_candidates:
                if os.path.exists(p):
                    audio_wav_path = p
                    break
            if not audio_wav_path:
                print(f"[WARNING] Audio file not found for {imdbid}")
                progress_bar.update(len(pending_tids))
                continue

            class Args:
                pass
            inference_args = Args()
            inference_args.videoFilePath = video_path
            inference_args.cropScale = args.cropScale
            inference_args.pretrainModel = args.pretrainModel
            inference_args.audio_wav_path = audio_wav_path
            inference_args.asdBatch = args.asdBatch

            max_tracks = 10
            chunk_tracks = []
            chunk_metadata = []

            def flush_chunk():
                if not chunk_tracks:
                    return
                dataset = ASDPreprocDataset(tracks_chunk, inference_args, out_fps=out_fps)
                scores = runner.infer(dataset.featsA, dataset.featsV, out_fps=out_fps)

                # Map results back to each track (store average + timestamped scores per obj)
                for track_idx, (tid, obj_id) in enumerate(meta_chunk):
                    if track_idx >= len(scores):
                        continue
                    if tid not in tid_scores:
                        tid_scores[tid] = {}
                    seq = scores[track_idx]
                    base_k0 = None
                    if track_idx < len(dataset.base_k0_per_track):
                        base_k0 = dataset.base_k0_per_track[track_idx]
                    tid_scores[tid][obj_id] = {"mean_score": _mean_score_windowed(seq, radius=2), "scores": _scores_with_timestamps(seq, base_k0, out_fps), "start_time": None if base_k0 is None else float(base_k0) / float(out_fps), "fps": float(out_fps)}
                chunk_tracks.clear()
                chunk_metadata.clear()

            for tid in pending_tids:
                track_file_path = os.path.join(args.track_dir, imdbid, f"{str(tid)}.pkl")
                with open(track_file_path, 'rb') as f:
                    try:
                        obj_tracks_dict = pickle.load(f)
                    except Exception as e:
                        print(f"Failed to load pickle {track_file_path}: {e}")
                        progress_bar.update(1)
                        continue

                if not isinstance(obj_tracks_dict, dict):
                    print(f"Track file {track_file_path} does not contain a dict of obj_id:track.")
                    progress_bar.update(1)
                    continue

                tid_scores = {}
                for obj_id, track_array in obj_tracks_dict.items():
                    if track_array is None or len(track_array) == 0:
                        continue
                    track_array = np.asarray(track_array)
                    if track_array.ndim != 2 or track_array.shape[1] != 5:
                        continue

                    track_dict = {
                        'frame': track_array[:, 0].astype(int),
                        'bbox': track_array[:, 1:5].tolist()
                    }
                    chunk_tracks.append(track_dict)
                    chunk_metadata.append((tid, obj_id))

                    if len(chunk_tracks) >= max_tracks:
                        tracks_chunk = list(chunk_tracks)
                        meta_chunk = list(chunk_metadata)
                        flush_chunk()

                if chunk_tracks:
                    tracks_chunk = list(chunk_tracks)
                    meta_chunk = list(chunk_metadata)
                    flush_chunk()

                if tid_scores:
                    output_path = os.path.join(args.output_dir, imdbid, f"{str(tid)}.pkl")
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    with open(output_path, 'wb') as f:
                        pickle.dump(tid_scores, f)
                progress_bar.update(1)
    finally:
        progress_bar.close()

if __name__ == "__main__":
    main()
