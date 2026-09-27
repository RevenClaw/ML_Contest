# Run complete training on Ubuntu

## Start

Copy the project folder and the supplied dataset onto the Ubuntu server. Keep the
project structure intact, including `code/business_entity_resolution/` and its tests.
From the project folder, run:

```bash
bash run_training.sh --data /path/to/student_resource
```

You can point `--data` directly at `student_resource/dataset/` instead. If the
following layout is used, no arguments are needed:

```
project/
  run_training.sh
  code/business_entity_resolution/
  student_resource/
    dataset/
      train/train_source1.tsv
      train/train_source2.tsv
      train/train_source3.tsv
      train/train_ground_truth.tsv
      test/...
```

```bash
bash run_training.sh
```

No Windows paths are embedded in the training launcher. `bash` invocation means
there is no need to set the executable bit after copying files from Windows.
The launcher and code use LF line endings.

## NVIDIA A30: use CUDA training

Run this command on your A30 server:

```bash
bash run_training.sh --data /path/to/student_resource --device cuda
```

The A30 has compute capability 8.0, meeting LightGBM's CUDA requirement of 6.0 or
higher. CUDA can accelerate its tree-learning stage. SQLite retrieval, pairwise
feature preparation, the Python macro F0.5 callback/threshold search, and
exported-model prediction still use the CPU. End-to-end speedup depends on those
stages; no speedup multiplier or training duration has been measured here.

The updated launcher defaults to `--device auto`: it selects CUDA when `nvidia-smi`
reports a GPU and GPUs are not explicitly hidden with `CUDA_VISIBLE_DEVICES`.
Explicit `--device cuda` is recommended for this server so a missing NVIDIA driver
cannot result in an accidental CPU run. `--device cpu` retains CPU operation.
Once CUDA is selected, build/preflight/training failures stop the pipeline; there
is no automatic CPU fallback.

CUDA setup performs these additional steps:

1. Checks `nvidia-smi`. A working NVIDIA driver and GPU passthrough must already
   be provided by the server administrator. The launcher does not install drivers
   or reboot the server.
2. Creates a separate `.venv-linux-cuda/` environment.
3. Finds an existing CUDA 11/12 toolkit or attempts to install Ubuntu's
   `nvidia-cuda-toolkit` package. With `BER_SKIP_APT=1`, missing system dependencies
   must instead be installed by the administrator. If the repository lacks that
   package, use the server's preinstalled toolkit and point `CUDACXX` at its `nvcc`.
4. Installs a compatible host compiler (GCC 10 for CUDA 11, GCC 12 for CUDA 12).
   Python build isolation supplies CMake/scikit-build dependencies. Source build
   parallelism defaults to four jobs; `CMAKE_BUILD_PARALLEL_LEVEL` can change it.
5. Builds pinned LightGBM 4.6.0 from source with `USE_CUDA=ON`. The ordinary CPU
   wheel is not used for CUDA training. A verified binary is cached for later runs.
6. Runs a tiny synthetic three-tree CUDA preflight before any contest data
   preprocessing. It checks the selected device, compiled backend, runtime and
   driver, prints `CUDA PREFLIGHT PASSED`, and records the actual loaded binary hash.

LightGBM 4.6.0's build targets architectures removed in CUDA 13, so this launcher
uses CUDA 11/12 for building this pinned release. A newer NVIDIA **driver** can
still run a compatible older toolkit build. For a toolkit installed outside the
usual paths:

```bash
CUDACXX=/usr/local/cuda-12.4/bin/nvcc bash run_training.sh --data /data/student_resource --device cuda
```

The build helper records/restores the toolkit's `lib64` directory for normal
layouts. For a nonstandard runtime location, include it in `LD_LIBRARY_PATH` on
each invocation; a server module/environment may already provide it.

To select a GPU, use `--gpu-device-id N`, where N is the ordinal **after** applying
`CUDA_VISIBLE_DEVICES`. For example, to expose physical GPU 1 as logical device 0:

```bash
CUDA_VISIBLE_DEVICES=1 bash run_training.sh --data /data/student_resource --device cuda --gpu-device-id 0
```

CUDA training uses `device_type=cuda`, one GPU, and `max_bin=63` by default. It omits
CPU-only deterministic/column-wise flags and OpenCL-only precision flags. The macro
F0.5 objective, reference split, negatives, trial comparison, and exact threshold
selection are unchanged. GPU results need not be bit-identical to CPU results;
changing histogram bins can also change scores. To compare with the previous bin
count, use a different run directory:

```bash
bash run_training.sh --data /data/student_resource --device cuda --max-bin 127 --run-dir /mnt/ssd/a30-127bins
```

Default CUDA outputs are in `runs/full-training-cuda/`. `hardware.json` records the
device and preflight details. Heartbeats report GPU utilization and memory alongside
CPU RAM and disk space. These GPU telemetry lines list physical devices as reported
by `nvidia-smi`; the preflight identifies the selected CUDA-visible device. An A30
partition exposed through MIG may have less memory than the full GPU. The code does
not silently shrink training data on a GPU out-of-memory error.

Existing CPU-run artifacts should not be placed in the CUDA run directory. The
fingerprints reject changed backend/configuration/code rather than mixing results.
Use a fresh run directory with this updated project. `BER_REBUILD_CUDA=1` forces a
new source build if the cached binary needs replacement.

References: [NVIDIA compute capabilities](https://developer.nvidia.com/cuda/gpus),
[LightGBM 4.6 CUDA installation](https://lightgbm.readthedocs.io/en/v4.6.0/Installation-Guide.html#build-cuda-version).

## Server setup

Intended targets are Ubuntu 22.04/24.04 on x86_64 or ARM64, with Bash, internet
access for software installation, and root or sudo access for missing OS packages.
No GPU is required for CPU mode. Software is installed as follows:

1. `apt-get`: Python bootstrap support, certificate bundle, GNU OpenMP runtime,
   and standard utilities.
2. A project-local bootstrap virtual environment installs pinned `uv`.
3. uv installs a local Python 3.12 runtime, including on Ubuntu 22.04 whose default
   Python is older than the project's requirements.
4. Pinned dependencies from `requirements.txt` are installed in `.venv-linux/`.
5. Dependency consistency and the synthetic unit/integration suite are checked.

The OS Python installation is not used as the application's package environment.
No third-party apt repository is added. Generic software installation is the only
network activity; business data is not uploaded or looked up online.

If the administrator already installed `python3`, `python3-venv`, `ca-certificates`,
`libgomp1`, and `util-linux`, use:

```bash
BER_SKIP_APT=1 bash run_training.sh --data /path/to/student_resource
```

Full-scale RAM and disk requirements are not benchmarked. Plan for substantial
local SSD space and RAM: at 100 million stored pairs, just the 64 float32 features
consume 25.6 GB, with additional space for indices, ID lists, and training buffers.
Memory mapping does not eliminate LightGBM's allocations. The terminal reports
available memory/free disk and actual preparation throughput. Resource exhaustion
is a failure, not a successful training completion.

To choose a large artifact volume and CPU thread count:

```bash
bash run_training.sh --data /data/student_resource --run-dir /mnt/ssd/ber-run --threads 16
```

The default thread count is up to 16 CPUs from the process's CPU affinity. Explicit
`--threads` overrides it. Preprocessing is currently single-process; the thread
setting applies to LightGBM. Avoid running multiple launchers from the same checkout.

## What happens automatically

1. Validate data presence and compute content fingerprints.
2. Record effective configuration, source-code hashes, dependency versions, and
   a full installed-package version list.
3. Index all three training sources and all ground-truth labels.
4. Prepare features for the 80/10/10 train/tune/audit reference split. No development
   reference cap is allowed by this full-training launcher.
5. Compare the three configured LightGBM trials. Optimize macro F0.5 for early
   stopping and sweep exact score thresholds on the tuning split.
6. Save the selected model and evaluate it on the untouched audit split.
7. Reopen the model, verify its schema and reports, and write `SUCCESS.json`.

The exported model is the selected, audited model trained on the training partition.
The full supplied dataset participates in preparation/evaluation, but the tuning
and audit labels are deliberately not included in fitting. There is no automatic
refit on holdout labels that would invalidate the independent audit score.

## Terminal updates and outputs

The terminal shows package installation, test results, indexing counts, feature
preparation counts/percentages/throughput, estimated remaining preparation time,
model iterations and tuning score, trial summaries, audit results, and final paths.
A heartbeat prints every 30 seconds during long-running pipeline stages. ETA is
based on observed throughput and is an estimate, not a runtime guarantee.

On success you will see messages such as:

```
TRAINING COMPLETE | audit macro F0.5=... | threshold=... | model=...
FULL PIPELINE COMPLETED SUCCESSFULLY | elapsed ... minutes | results: ...
```

Default outputs:

```
logs/training-<timestamp>-<pid>.log       complete terminal output, including setup
runs/full-training/
  effective_config.json
  run_manifest.json                    input/code/config/dependency fingerprints
  environment-freeze.txt
  train.sqlite
  prepared/{train,tune,audit}/          feature arrays, pair/reference manifests
  model/
    model.txt                          selected trained model
    model_metadata.json                selected threshold/configuration
    trial_*.txt                        trained candidates
    trial_*.json                       completed-trial checkpoints
    trials.json
    tune_metrics.json
    audit_metrics.json
    feature_importance.json
    complete.json
  logs/pipeline.log                    orchestration log
  SUCCESS.json                         written after verification
```

On failure the launcher exits nonzero, prints an error, and records
`last_failure.json` when orchestration has started. A previous successful run's
`SUCCESS.json` is moved to a timestamped history file before a new invocation.
Read the timestamp of any historical failure/success report.

## Restart after interruption

Run exactly the same command again. The pipeline checks that the data, configuration,
code, and dependency versions have not changed before reusing artifacts.

- Completed indices are reused. A partial database build is preserved with an
  `.interrupted-<timestamp>` suffix and rebuilt from the beginning. Preserved
  databases consume additional space; remove obsolete copies yourself after review.
- Feature preparation checkpoints every 1,000 references. It truncates uncommitted
  trailing writes to the saved offsets and resumes without duplicate examples.
- Completed model trials are reused. An interrupted trial restarts from its first
  boosting round; training does not resume partway through a tree ensemble.
- A completed model is verified and reused.

Changed code/config/data require a new `--run-dir`. This prevents stale candidates,
features, or thresholds from being silently mixed. Checkpoints are not an automatic
recovery from corrupted files; structural inconsistencies stop the run.

For an SSH session that must survive disconnects, start the command inside an
existing `tmux` or `screen` session. The launcher itself does not daemonize or install
an ongoing service.

## Optional prediction and submission files

To continue automatically from training through test prediction and validation:

```bash
bash run_training.sh --data /data/student_resource --predict
```

This adds a test index and writes `output/matching_results.tsv` and
`output/candidate_pairs.tsv` under the run directory, then validates both. You can
also add `--predict` when rerunning a previously completed training run; the model
is reused. Interrupted prediction restarts that stage, preserving its old partial
output directory.

## Verification status

Synthetic regression/integration tests and Bash syntax/help checks have been run.
Training tests use stub models. No real-data training, full-size benchmark, or live
Ubuntu installation/training run has been executed in the development session.

The dependency-bootstrap design follows the [uv Python installation documentation](https://docs.astral.sh/uv/guides/install-python/).
