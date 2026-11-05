# Evaluation

Scripts for the three metrics reported in the paper, run against a directory
of predicted character banks. All paths below use the defaults baked into the
scripts (override with the listed flags).

## Metrics

**Character Naming Accuracy** (`eval.py`) — per-query naming accuracy.
*Overall* covers all queries, including those whose GT is `none` (no
on-screen face track). *Known* is restricted to queries with a valid
face-track GT, isolating the ability to ground on-screen characters.

**Character Bank mean Temporal IoU** (`eval_iou.py`) — how closely the
streaming bank tracks human knowledge of each character over time. For each
character we compute a temporal IoU between the interval a human is aware of
them and the interval the bank correctly identifies them, then mention-weight
across characters. Two variants are reported from the MCN timestamps:
*visual* (interval starts at the first on-screen appearance) and *name*
(interval starts at the first moment a viewer can link face to name).
Penalises both late identification and entries later overwritten with the
wrong name.

**Principal Character Recall** (`eval_pcr.py`) — fraction of a movie's top-K
IMDB principals (ranked by narrative weight) that are recovered by the bank.
A principal counts as recovered only when the *same* bank entry passes both
checks: its canonical name phonetically matches the IMDB name, and its face
centroid is within `τ_eval` of the IMDB reference embedding. Requiring both
on a single entry penalises name/face binding failures.

`eval_iou_ablation.py` and `eval_pcr_ablation.py` rerun the same two
bank-level metrics across post-hoc variants (`full`, `no_metaphone`,
`no_collision`) of one output directory, without modifying any predicted
artifacts.

## Inputs

* **Predicted banks** — one `{imdbid}/character_bank.pkl` per movie under
  `--pred_log_dir` / `--output_dir`. mTIoU additionally reads
  `{imdbid}/timeline/cluster_*.json` and `cluster_features.pkl`.
* **GT annotations** — `annotations.json` keyed by imdbid; each character
  carries `visual_time`, `linkage_time`, and a `weight` used both for
  mention-weighting in mTIoU and for ranking IMDB principals in PCR.
* **IMDB reference bank** — `character_bank.pkl` keyed by imdbid, each entry a
  `(cast_names, cast_features)` pair of InsightFace profile embeddings. Same
  file backs the face leg of PCR and the embedding-matching step in mTIoU.
* **Naming-accuracy inputs** — `eval.py` reads a JSONL of per-query predictions
  and a `test.csv` of ground-truth labels (see `--jsonl_file`, `--csv_file`).

## Usage

```bash
# Character Naming Accuracy
python eval.py \
    --jsonl_file "<predictions.jsonl>" \
    --csv_file   "<test.csv>"

# Mean Temporal IoU (reports both visual and name variants)
python eval_iou.py \
    --output_dir "<logs/run>/" \
    --annotation_file "<annotations.json>" \
    --character_bank_pkl "<IMDB_bank.pkl>" \
    --video_dir "<video_dir>"

# PCR@K (default K=15)
python eval_pcr.py \
    --pred_log_dir "<logs/run>/" \
    --gt_annotations "<annotations.json>" \
    --gt_bank_pkl "<IMDB_bank.pkl>" \
    --top_k 15 --tau_face 0.45

# Ablations (full / no_metaphone / no_collision in one pass)
python eval_iou_ablation.py \
    --output_dir "<logs/run>/" \
    --annotation_file "<annotations.json>" \
    --character_bank_pkl "<IMDB_bank.pkl>" \
    --video_dir "<video_dir>" \
    --shot_seg_file "<shot_metadata.json>"
python eval_pcr_ablation.py \
    --pred_log_dir "<logs/run>/" \
    --gt_annotations "<annotations.json>" \
    --gt_bank_pkl "<IMDB_bank.pkl>"
```

Add `--imdbid <id>` to any of the bank-based scripts to evaluate a single
movie, or `--output <path>` to dump per-movie JSON alongside the printed
summary.

## Shared utilities

`utils.py` provides primitives reused by every script:

* `load_character_bank` — load the IMDB reference bank for one imdbid.
* `find_character_embeddings` — collect all profile embeddings tagged with a
  given character name (cast labels have the form `"Character - Scene"`).
* `hungarian_match` — one-to-one assignment of GT characters to predicted
  clusters by max-cosine similarity, used by mTIoU to pick the cluster each
  GT character is compared against.
* `names_match` — token-level phonetic comparison via Double Metaphone with a
  Levenshtein fallback, used by both naming accuracy and the name leg of PCR.
* `_l2` — L2 normalisation for face centroids before cosine similarity.
