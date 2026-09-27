# Business entity resolution: LightGBM pipeline

Status: implementation only. No training, real-data indexing, or candidate preparation
has been run. Synthetic tests exercise the metric and preprocessing; a prediction
test uses a stub scorer, never a fitted model. Accuracy and full-scale throughput
are unmeasured.

For automatic Ubuntu setup and full training, use the root-level
`run_training.sh`. From the project root:

```bash
bash run_training.sh --data /path/to/student_resource
```

On the NVIDIA A30 server, add `--device cuda`. The launcher compiles CUDA-enabled
LightGBM in a separate environment, verifies a tiny synthetic GPU fit before
preprocessing, and uses `device_type=cuda` with `max_bin=63`. CUDA run artifacts
default to `runs/full-training-cuda/`. The full setup and device-selection details
are in `../../UBUNTU_TRAINING.md`. CPU retrieval/features and the Python validation
metric are not GPU-accelerated. No GPU benchmark has been performed locally.

See `../../UBUNTU_TRAINING.md` for terminal updates, resumability, resource planning,
and the optional `--predict` stage. The commands below remain available for running
individual stages manually.

## Objective

Maximize the README's **macro F0.5**, not accuracy or an error to maximize. For a
reference with `T` true matches, `P` predicted matches, and `TP` correct matches:

`F0.5 = 1.25 * TP / (0.25 * T + P)`.

When `T = 0`, score 1 if `P = 0`, otherwise 0. Average over **every** reference,
including references with no candidates. The training loss is binary log loss,
which LightGBM can differentiate. Early stopping maximizes validation macro F0.5
over a configurable threshold grid. After training each trial, an exact threshold
sweep selects the threshold and trial with highest tuning macro F0.5. Tied scores
enter the sweep together; a higher threshold wins equal objective values.

The scorer includes true matches that blocking missed in the denominator. It does
not report candidate-only F0.5 as if it were the competition score.

## Directory layout

```
configs/default.json     reproducible starting settings, not proven optimum
src/ber/common.py        strict TSV reading, deterministic reference splits
src/ber/text.py          Unicode-preserving normalization and transliteration
src/ber/store.py         streaming SQLite/FTS5 index and label integrity checks
src/ber/retrieval.py     six retrieval routes and reciprocal-rank fusion
src/ber/features.py      64 pairwise comparison features
src/ber/prepare.py       hard negatives, manifests, disk-backed feature matrices
src/ber/metrics.py       exact competition scorer and threshold optimization
src/ber/train.py         explicit LightGBM training and independent audit
src/ber/predict.py       batched scoring and both required TSV files
src/ber/validate.py      strict output coverage, ID, duplicate, and subset checks
src/ber/cli.py           separate stage commands
src/ber/server.py        full server orchestration, logs, fingerprints, completion checks
src/ber/backend.py       explicit CPU/CUDA settings and protected trial parameters
src/ber/cuda.py          CUDA driver/build preflight and utilization reporting
scripts/setup_cuda.sh   source-build provisioning for CUDA LightGBM
tests/test_pipeline.py  synthetic regression and integration tests
```

Generated indices, prepared features, and models go in an `artifacts/` directory
chosen by the operator. Prefer a fast local SSD outside OneDrive sync for large runs.
The original dataset is read-only. Indexing and prediction refuse to overwrite
existing outputs. Preparation resumes compatible checkpoints; training reuses
completed trials with matching configuration. Incompatible states are rejected.
The server launcher preserves incomplete indices/predictions and restarts those
stages, rather than deleting them.

## Setup (when execution is authorized)

Use Python 3.11/3.12, SQLite with FTS5, and pinned dependencies. These commands do
not train a model:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m ber.cli --help
```

All following commands assume this project directory and `PYTHONPATH=src`.
Example paths for the current workspace:

```powershell
$dataset = 'C:\Users\reven\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset'
$artifacts = '.\artifacts'
```

## Preprocessing stages (not run yet)

```powershell
python -m ber.cli index --data "$dataset" --split train --index "$artifacts\train.sqlite" --config configs/default.json
python -m ber.cli prepare --index "$artifacts\train.sqlite" --out "$artifacts\prepared" --config configs/default.json
```

`index` streams the sources into a disk-backed store, verifies ground-truth
coverage, duplicate IDs, duplicate labels, shared target ownership, and missing
targets, then builds the text index. An S2/S3 ID assigned to multiple S1 IDs causes
an explicit failure because the deduplicated-reference assumption and holdout
isolation would otherwise be violated; investigate labels before proceeding.

`prepare` performs retrieval, computes features, and writes binary float32
matrices, labels, reference-group indices, pair IDs, and reference manifests. It
does **not** train. Set `max_references_per_split` in a copied config for a smaller
development run. This limits reference processing, not target-index size. Do not
truncate target files: doing so would make retrieval evaluation unrealistically easy.

The next command explicitly starts training. The Ubuntu launcher runs it automatically.

## Training stage (documented only; not executed)

```powershell
python -m ber.cli train --prepared "$artifacts\prepared" --out "$artifacts\model" --config configs/default.json
```

Stable hashed S1 IDs define 80% train, 10% threshold/model tuning, and 10% audit.
All known positive target records owned by tuning/audit references are excluded
from training candidates, including training negatives. Unowned distractors remain.
Tuning/audit retrieval searches the full target corpus, preserving realistic
distractors. Unlabeled corpus statistics (FTS ranking and token frequency) use the
full source corpus; this is a transductive retrieval setup, not fitting on holdout labels.

Training keeps all retrieved positives and up to 40 negatives per reference: up
to 30 highest-retrieval-ranked negatives plus a reproducible selection from the
remainder. No ground-truth matches are injected into candidate lists. Validation
and audit retain **all** retrieved candidates. Equal total pair weight per reference
reduces the dominance of references with many candidates. Sampling/weighting means
raw model outputs are scores, not guaranteed calibrated probabilities.

Three regularized tree configurations are compared on the tuning fold. Training
uses deterministic settings in CPU mode, or the explicit CUDA backend on the GPU,
and early stopping. CUDA is not promised to be bit-reproducible. The audit fold is scored once
after selection and is never used to select the model or threshold. Do not repeatedly
tune against audit results; create a new untouched split if redesigning extensively.
There is no automatic full-data refit, so the exported model remains the audited
model. Any later refit requires renewed threshold-transfer checks.

Training outputs include the model, iteration count, parameters, selected threshold,
tuning and audit metrics, country breakdowns, singleton accuracy, blocking recall,
blocking score ceiling, and feature importance. No scores are claimed before this run.
Completed trials are checkpointed individually. Early-stopping grid evaluation uses
a vectorized score histogram, reducing repeated scans across the validation pairs.

## Retrieval and features

Retrieval combines independent name-token, address-token, joint name/address,
name-character-trigram, exact-name, and exact-address routes. SQLite FTS5 supplies
BM25 ordering; reciprocal-rank fusion retains up to 200 candidates by default.
These values are starting points. Raise route limits and inspect validation blocking
recall/score ceiling before concluding that classifier changes are needed.

Each route dynamically accepts all country labels, including France. Country
matching is a retrieval assumption: sampled true training pairs agreed on country.
No country category is one-hot encoded or restricted to US/India. Unknown-country
performance cannot be established from US/India validation alone.

The features compare native and transliterated names/addresses, legal-suffix-reduced
names, edit distances, token overlap/containment, character-trigram Dice similarity,
token-order-independent similarity, corpus-rarity-weighted overlap, missing fields,
address numbers and ranges, possible postal-code conflicts, acronyms, script flags,
and retrieval evidence. Postal-like numbers are heuristic features, never hard rules.
Missing values never receive perfect agreement merely because both are missing.
Entity IDs are used only for indexing, grouping, source identification, and output.

AnyAscii supplies generic local transliteration (not business lookup); original
Unicode is preserved as another comparison view. Transliteration is imperfect,
especially across Indic scripts, and must be validated. No internet calls, geocoding,
business registry lookup, external identity data, or pretrained model are used.

## Prediction and validation (after training)

```powershell
python -m ber.cli index --data "$dataset" --split test --index "$artifacts\test.sqlite" --config configs/default.json
python -m ber.cli predict --index "$artifacts\test.sqlite" --model "$artifacts\model" --out ..\..\output
python -m ber.cli validate --index "$artifacts\test.sqlite" --output ..\..\output
```

Prediction writes exactly the candidates passed to model inference, after all
retrieval filtering, into `candidate_pairs.tsv`. It produces one row per S1 entity
in both files, including empty lists, and never forces a top-one assignment. Model
inference is batched. Partial outputs have `.partial` extensions until completion.

The built-in validator streams sorted reference rows and checks all target IDs on
disk, avoiding large Python ID sets. It additionally requires sorted reference IDs
(our writer guarantees this). Also run the organizer's validator:

```powershell
python "$dataset\..\utils\validate_submission.py" --matching ..\..\output\matching_results.tsv --candidate ..\..\output\candidate_pairs.tsv --test-dir "$dataset\test" --check-ids
```

## Scale and remaining experiments

This implementation avoids a Cartesian product and keeps retrieval on disk. It does
not promise that millions of FTS queries or feature computations will be fast on a
particular machine. Retrieval/preparation currently run in a single process; model
training uses the configured CPU threads. Benchmark before full execution.

Each stored pair uses 256 bytes for 64 float32 features plus 5 bytes for label/group
arrays, plus text ID manifests. At 100 million pairs the feature array alone is
25.6 GB. SQLite indices and model training need additional disk/RAM; memory mapping
does not eliminate LightGBM's own allocations. Threshold-grid evaluation on every
boosting iteration also has a real cost. Use a smaller development reference cap
to measure recall, runtime, and memory before committing to a full run.

The first accuracy experiments should measure blocking recall by country, name and
address missingness, and match count; increase retrieval coverage if its ceiling is
low. Then compare feature ablations and tune regularization/thresholds on the tuning
fold. Finally run the untouched audit once. No claim of maximum achievable accuracy
is possible without these experiments.

## Final submission

Package `output/matching_results.tsv`, `output/candidate_pairs.tsv`, this complete
project under `code/business_entity_resolution/`, and the filled methodology document.
Include trained model artifacts if desired, but retain commands to retrain from the
provided data. Exclude `.venv`, temporary indices, prepared matrices, and raw datasets.
LightGBM is MIT licensed; retain dependency license notices as applicable.
