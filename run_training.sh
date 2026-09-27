#!/usr/bin/env bash
# Ubuntu launcher. Usage: bash run_training.sh --data /path/to/student_resource
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$ROOT/code/business_entity_resolution"
if [[ ! -d "$PROJECT/src/ber" ]]; then
    printf 'ERROR: Keep run_training.sh beside code/business_entity_resolution/.\n' >&2
    exit 2
fi
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'HELP'
Run the complete Ubuntu CPU or NVIDIA CUDA training pipeline:
  bash run_training.sh --data /path/to/student_resource

Options (forwarded to the pipeline):
  --data PATH        student_resource/ or its dataset/ directory. Auto-detected
                     if dataset/ or student_resource/dataset/ is beside this script.
  --run-dir PATH     Artifacts/logs (default: runs/full-training for CPU,
                    runs/full-training-cuda for CUDA).
  --threads N        LightGBM CPU threads (default: min(available CPUs, 16)).
  --device MODE      auto (default), cuda, or cpu. Use cuda for the NVIDIA A30.
  --gpu-device-id N  CUDA-visible GPU ordinal (default: 0).
  --max-bin N        Histogram bins (default: 63 CUDA, 127 CPU).
  --config PATH      Configuration (default: configs/default.json).
  --predict         Also generate and validate both competition TSVs after training.

Re-run the same command after interruption. Completed stages are reused; feature
preparation resumes at its last checkpoint and completed model trials are reused.
Changed data/config/code require a different --run-dir. An unfinished database
build is preserved under a timestamped name and rebuilt, not resumed row by row.

System packages are installed using apt-get (sudo if not root), then a local
Python 3.12 environment and pinned Python dependencies are installed. Ubuntu
22.04/24.04, x86_64 or ARM64, are intended targets. Internet access is needed only
for software installation. Set BER_SKIP_APT=1 when system packages already exist.
CUDA mode builds LightGBM from source and requires a working NVIDIA driver.
The launcher installs a missing CUDA 11/12 toolkit from Ubuntu packages, but does
not install/change GPU drivers. CUDA build/preflight failure stops the pipeline.
HELP
    exit 0
fi
DEVICE='auto'
GPU_DEVICE_ID=0
FORWARDED=()
while (( $# )); do
    case "$1" in
        --device)
            [[ $# -ge 2 ]] || { printf 'ERROR: --device needs auto, cpu, or cuda.\n' >&2; exit 2; }
            DEVICE="$2"; shift 2 ;;
        --device=*) DEVICE="${1#*=}"; shift ;;
        --gpu-device-id)
            [[ $# -ge 2 ]] || { printf 'ERROR: --gpu-device-id needs an integer.\n' >&2; exit 2; }
            GPU_DEVICE_ID="$2"; shift 2 ;;
        --gpu-device-id=*) GPU_DEVICE_ID="${1#*=}"; shift ;;
        *) FORWARDED+=("$1"); shift ;;
    esac
done
[[ "$DEVICE" =~ ^(auto|cpu|cuda)$ ]] || { printf 'ERROR: Invalid --device.\n' >&2; exit 2; }
[[ "$GPU_DEVICE_ID" =~ ^[0-9]+$ ]] || { printf 'ERROR: Invalid GPU ordinal.\n' >&2; exit 2; }
if [[ "$(uname -s)" != "Linux" ]]; then
    printf 'ERROR: Run this launcher on the Ubuntu server, not Windows.\n' >&2
    exit 2
fi
mkdir -p "$ROOT/logs"
BOOT_LOG="$ROOT/logs/training-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
exec > >(tee -a "$BOOT_LOG") 2>&1
trap 'status=$?; printf "\nTRAINING PIPELINE FAILED (exit %s, line %s). Log: %s\n" "$status" "$LINENO" "$BOOT_LOG" >&2; exit "$status"' ERR
CHILD=''
stop_pipeline() {
    local code="$1"
    if [[ -n "$CHILD" ]] && kill -0 "$CHILD" 2>/dev/null; then
        kill -TERM "$CHILD" 2>/dev/null || true
        wait "$CHILD" || true
    fi
    printf '\nTraining interrupted. Re-run the same command to resume.\n'
    exit "$code"
}
trap 'stop_pipeline 130' INT
trap 'stop_pipeline 143' TERM
printf '\n[%s] Starting automatic training setup\nLog: %s\n' "$(date -u +%FT%TZ)" "$BOOT_LOG"
command -v flock >/dev/null || { printf 'ERROR: flock is required (Ubuntu util-linux package).\n'; exit 2; }
exec 9>"$ROOT/.training.lock"
if ! flock -n 9; then
    printf 'ERROR: Another launcher is already running from this project.\n' >&2
    exit 2
fi
if [[ "$DEVICE" == auto ]]; then
    DEVICE=cpu
    if command -v nvidia-smi >/dev/null && [[ "${CUDA_VISIBLE_DEVICES-unset}" != '-1' && "${CUDA_VISIBLE_DEVICES-unset}" != '' ]]; then
        GPU_LIST="$(nvidia-smi -L)"
        [[ -z "$GPU_LIST" ]] || DEVICE=cuda
    fi
fi
if [[ "$DEVICE" == cuda ]]; then
    command -v nvidia-smi >/dev/null || { printf 'ERROR: CUDA requested but nvidia-smi is unavailable. Provision the NVIDIA driver on the server first.\n' >&2; exit 2; }
    nvidia-smi
fi
printf '\nSelected training backend: %s (GPU ordinal: %s)\n' "$DEVICE" "$GPU_DEVICE_ID"
if [[ "${BER_SKIP_APT:-0}" != "1" ]]; then
    if (( EUID == 0 )); then
        SUDO=()
    elif command -v sudo >/dev/null; then
        SUDO=(sudo)
    else
        printf 'ERROR: Install python3, python3-venv, ca-certificates, libgomp1 and util-linux as root, then set BER_SKIP_APT=1.\n' >&2
        exit 2
    fi
    "${SUDO[@]}" apt-get update
    "${SUDO[@]}" apt-get install -y --no-install-recommends python3 python3-venv ca-certificates libgomp1 util-linux
fi
# Bootstrap uv in an isolated venv; never modify Ubuntu's system Python packages.
if [[ ! -x "$ROOT/.bootstrap/bin/python" ]]; then
    python3 -m venv "$ROOT/.bootstrap"
fi
"$ROOT/.bootstrap/bin/python" -m pip install --disable-pip-version-check 'uv==0.8.22'
UV="$ROOT/.bootstrap/bin/uv"
export UV_PYTHON_INSTALL_DIR="$ROOT/.uv-python"
export UV_CACHE_DIR="$ROOT/.uv-cache"
ENV_DIR="$ROOT/.venv-linux"
[[ "$DEVICE" != cuda ]] || ENV_DIR="$ROOT/.venv-linux-cuda"
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
    "$UV" python install 3.12
    "$UV" venv --python 3.12 "$ENV_DIR"
fi
PYTHON="$ENV_DIR/bin/python"
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=42
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
if [[ "$DEVICE" == cuda ]]; then
    "$UV" pip install --python "$PYTHON" --only-binary :all: -r "$PROJECT/requirements-base.txt"
    source "$PROJECT/scripts/setup_cuda.sh"
    setup_cuda
else
    "$UV" pip install --python "$PYTHON" --only-binary :all: -r "$PROJECT/requirements.txt"
fi
"$UV" pip check --python "$PYTHON"
printf '\n[%s] Checking dependencies and running synthetic tests (no training in tests)\n' "$(date -u +%FT%TZ)"
"$PYTHON" -m unittest discover -s "$PROJECT/tests" -v
"$PYTHON" -u -m ber.server --project-root "$ROOT" --device "$DEVICE" --gpu-device-id "$GPU_DEVICE_ID" "${FORWARDED[@]}" &
CHILD=$!
wait "$CHILD"
CHILD=''
printf '\n[%s] Launcher finished successfully. Complete terminal log: %s\n' "$(date -u +%FT%TZ)" "$BOOT_LOG"
