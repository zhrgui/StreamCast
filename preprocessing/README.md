# Preprocessing

This folder contains the preprocessing pipeline used to turn raw videos into
per-track face features, transcripts, and Active Speaker Detection (ASD)
synchronisation scores.

The stages run sequentially:

1. **Shot segmentation** — split the input video into shots.
2. **Face tracking** (per shot, SAM 3) — track faces within each shot.
3. **Face feature extraction** (InsightFace) — embed each track.
4. **Transcription + name extraction** (Parakeet / WhisperX + spaCy) — runs on the audio track.
5. **ASD inference** (TalkNet) — produce per-track speaking scores from the face tracks + audio.

Each stage is a standalone CLI; intermediate artifacts on disk are the only
contract between stages. All scripts are resumable — re-running a stage skips
shots / tracks / files that already have an output.

## Layout

```
preprocessing/
├── utils/
│   ├── shot_detect.py          # stage 1
│   ├── get_streaming_sid.py    # pick (imdbid, shot_id) pairs that still need tracking
│   └── select_track_file.py    # aggregate per-shot tracks into one pickle
├── face_tracking/
│   ├── face_track_fast_sam3.py        # stage 2
│   └── face_feature_vpcd_det_cuda.py  # stage 3
├── transcribe/
│   ├── transcribe_parakeet.py  # stage 4 (English)
│   ├── transcribe_whisperx.py  # stage 4 (multilingual + diarization)
│   └── detect_name.py          # PERSON NER over transcript text
└── asd_inference/
    ├── inference.py            # stage 5
    ├── talkNet.py / loss.py    # TalkNet wrapper
    ├── model/                  # TalkNet sub-modules (audio/visual encoders, attention)
    └── pretrain_TalkSet.model  # TalkNet weights pretrained on TalkSet
```

## Stage 1: Shot segmentation

Detects shot boundaries with PySceneDetect's `AdaptiveDetector`. One JSONL
record per video is appended to `--jsonl_path` as it finishes (used to resume),
and a final aggregated JSON is written to `--save_file`:

```json
{
  "tt0109831": {
    "0": {"frame_range": [0, 312], "time_range": [0.0, 13.0]},
    "1": {"frame_range": [313, 540], "time_range": [13.04, 22.5]},
    ...
  }
}
```

```bash
python utils/shot_detect.py \
  --video_dir   /path/to/video \
  --save_file   /path/to/meta/shot_metadata.json \
  --jsonl_path  /path/to/meta/shot_metadata.jsonl \
  --max_workers 8
```
## Stage 2: Face tracking

For every `(imdbid, shot_id)` in `--sid_file`, decodes the shot's frames to a
temp folder, prompts SAM 3 with the text `"face"` at three evenly-spaced
anchor frames, propagates through the shot, and saves the per-object boxes:

```
<save_dir>/face_box_track/<imdbid>/<shot_id>.pkl
   {obj_id: [[frame, x1, y1, x2, y2], ...], ...}
```

Long shots are split into sub-chunks (configurable via
`--max_frames_per_chunk`); chunks are then re-stitched into a single global
object id space using IoU between the last frame of one chunk and the first
frame of the next. The script auto-halves the chunk size on CUDA OOM and
retries up to 3 times.

First, generate the list of shots that still need tracking:

```bash
python utils/get_streaming_sid.py \
  --shot_metafile /path/to/meta/shot_metadata.json \
  --person_track  /path/to/person_tracks \
  --imdbid_list   /path/to/test_imdbid.json \
  --save_file     /path/to/meta/unprocessed_keys.json
```

Then run the tracker (shard across GPUs by varying `--chunk_idx`):

```bash
python face_tracking/face_track_fast_sam3.py \
  --video_dir     /path/to/video \
  --save_dir      /path/to/person_tracks/face_tracks \
  --sid_file      /path/to/meta/unprocessed_keys.json \
  --shotmeta_file /path/to/meta/shot_metadata.json \
  --num_chunks 8 --chunk_idx 0 \
  --max_frames_per_chunk 500
```

Once all shards are done, aggregate per-shot pickles into a single
per-corpus pickle that downstream stages consume:

```bash
python utils/select_track_file.py \
  --sid_file       /path/to/meta/streaming_sid.json \
  --face_track_dir /path/to/person_tracks/face_tracks/face_box_track \
  --save_file      /path/to/track_file/person_tracks.pkl
```

## Stage 3: Face features

For each face track in the aggregated pickle, samples `--k` frames uniformly,
runs InsightFace (`buffalo_l`) face detection on those frames, and matches
detections back to the track's reference bbox by IoU to recover the embedding.
The output pickle mirrors the input with an added `face_feats` field per
episode:

```
{imdbid: {"face": {...}, "face_feats": {track_id: (k, 512) np.float32}}}
```

Shardable via `--shard_id` / `--num_shards`; each shard writes
`<save_path>_shardN.pkl` plus a `.meta.json`.

```bash
python face_tracking/face_feature_vpcd_det_cuda.py \
  --anno_path /path/to/track_file/person_tracks.pkl \
  --vid_dir   /path/to/video \
  --save_path /path/to/track_file/person_tracks_with_feats.pkl \
  --k 15 --device_id 0
```

## Stage 4: Transcripts

Fast, single-pass English ASR with Nvidia Parakeet TDT. Recursively
splits-and-merges on CUDA OOM:

```bash
python transcribe/transcribe_parakeet.py \
  --audio_folder /path/to/audio \
  --save_dir     /path/to/transcripts/parakeet
```

WhisperX transcription + wav2vec2 alignment + pyannote diarization. Requires a
Hugging Face access token for the diarization model (pyannote/speaker-diarization):

```bash
export HUGGINGFACE_TOKEN=hf_xxx
python transcribe/transcribe_whisperx.py \
  --audio_folder /path/to/audio \
  --save_dir     /path/to/transcripts/whisperx \
  --language en
```

Both scripts write per-file JSON of the form:

```json
[{"start": 0.12, "end": 2.40, "text": "...", "speaker": "SPEAKER_00"}, ...]
```

(The `speaker` field is only set by WhisperX.)

Runs spaCy NER (`en_core_web_trf` by default) over the transcript text and
keeps only segments containing at least one PERSON entity, with the names
attached:

```bash
python transcribe/detect_name.py \
  --subtitle_dir /path/to/transcripts/parakeet \
  --output_dir   /path/to/transcripts/names
```

## Stage 5: ASD inference

For every `(imdbid, track_id)` in `--track_file`, decodes the matching frames
and audio, runs TalkNet over audio/visual windows of multiple durations
(1–6 s, weighted) and emits a per-track speaking-score time-series:

```
<output_dir>/<imdbid>/<track_id>.pkl
   {obj_id: {
     "mean_score": float,
     "scores": [{"t": seconds, "score": float}, ...],
     "start_time": float,
     "fps": 25.0,
   }}
```

Audio must already be mono 16 kHz wav (resample once offline with
`ffmpeg -ar 16000 -ac 1`). Multi-GPU is supported via PyTorch DataParallel
when more than one GPU is visible; the loop is also shardable via
`--shard_id` / `--num_shards` for trivial multi-machine parallelism.

```bash
python asd_inference/inference.py \
  --track_file    /path/to/meta/streaming_sid.json \
  --video_dir     /path/to/video \
  --pyavi_dir     /path/to/audio \
  --track_dir     /path/to/person_tracks/face_tracks/face_box_track \
  --output_dir    /path/to/person_tracks/asd \
  --pretrainModel asd_inference/pretrain_TalkSet.model \
  --asdBatch 64
```

## Dependencies

- Python 3.10+, CUDA 12.x
- `torch`, `torchvision`, `decord`, `opencv-python`, `pillow`
- `scenedetect` (stage 1)
- `sam3` (stage 2) — internal SAM 3 video predictor
- `insightface`, `onnxruntime-gpu` (stage 3)
- `nemo_toolkit[asr]` (Parakeet, stage 4), `whisperx` + `pyannote.audio` (WhisperX, stage 4)
- `spacy` + `en_core_web_trf` / `zh_core_web_trf` (stage 4 NER)
- `python_speech_features`, `scipy` (stage 5)

GPU is required for stages 2, 3, 4 and 5.


## Install
```
# Install from https://github.com/facebookresearch/sam3
conda create -n sam3 python=3.12 -y
conda deactivate
conda activate sam3

pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128

git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .

# For running example notebooks
pip install -e ".[notebooks]"

# For development
pip install -e ".[train,dev]"

pip install -r requirements.txt

# Download NER models (English and Chinese)
python -m spacy download en_core_web_trf && python -m spacy download zh_core_web_trf

# Parakeet Transcription
pip install -U nemo_toolkit['asr']

```

