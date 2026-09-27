#!/usr/bin/env bash
# Sourced by the root launcher so CUDA runtime/library paths remain in its environment.

cuda_major() {
    "$1" --version | sed -n 's/.*release \([0-9][0-9]*\)\..*/\1/p' | head -n 1
}

cuda_apt_install() {
    if [[ "${BER_SKIP_APT:-0}" == 1 ]]; then
        printf 'ERROR: Missing CUDA build packages. Have the administrator install: %s\n' "$*" >&2
        return 2
    fi
    "${SUDO[@]}" apt-get install -y --no-install-recommends "$@"
}

setup_cuda() {
    local stamp="$ENV_DIR/cuda-build-verified.json"
    local cached_lib=''
    if [[ -f "$stamp" ]]; then
        cached_lib="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("cuda_library_path") or "")' "$stamp")"
        if [[ -n "$cached_lib" && -d "$cached_lib" ]]; then
            export LD_LIBRARY_PATH="$cached_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            export BER_CUDA_LIBRARY_PATH="$cached_lib"
        fi
    fi
    if [[ "${BER_REBUILD_CUDA:-0}" != 1 ]] && "$PYTHON" -m ber.cuda --verify-build "$stamp"; then
        printf '\nReusing previously verified CUDA-enabled LightGBM binary.\n'
        return 0
    fi
    printf '\nBuilding LightGBM 4.6.0 with its native CUDA backend. This can take several minutes.\n'
    local candidate='' major='' nvcc='' compiler_major=''
    if [[ -n "${CUDACXX:-}" ]]; then
        nvcc="$(command -v "$CUDACXX")"
        major="$(cuda_major "$nvcc")"
        [[ "$major" == 11 || "$major" == 12 ]] || {
            printf 'ERROR: LightGBM 4.6.0 build requires CUDA 11/12 here. Set CUDACXX to a compatible nvcc (CUDA 13 removes older architectures used by this release).\n' >&2
            return 2
        }
    else
        for candidate in "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" /usr/local/cuda-12*/bin/nvcc /usr/local/cuda-11*/bin/nvcc /usr/bin/nvcc; do
            [[ -x "$candidate" ]] || continue
            major="$(cuda_major "$candidate")"
            if [[ "$major" == 11 || "$major" == 12 ]]; then nvcc="$candidate"; break; fi
        done
        if [[ -z "$nvcc" ]] && command -v nvcc >/dev/null; then
            candidate="$(command -v nvcc)"; major="$(cuda_major "$candidate")"
            [[ "$major" != 11 && "$major" != 12 ]] || nvcc="$candidate"
        fi
        if [[ -z "$nvcc" ]]; then
            cuda_apt_install nvidia-cuda-toolkit
            nvcc=/usr/bin/nvcc
            major="$(cuda_major "$nvcc")"
        fi
    fi
    [[ "$major" == 11 || "$major" == 12 ]] || {
        printf 'ERROR: Could not find a CUDA 11/12 toolkit. Set CUDACXX to its nvcc executable.\n' >&2
        return 2
    }
    compiler_major=12
    [[ "$major" != 11 ]] || compiler_major=10
    if ! command -v make >/dev/null || ! command -v "g++-$compiler_major" >/dev/null || ! command -v "gcc-$compiler_major" >/dev/null; then
        cuda_apt_install build-essential "gcc-$compiler_major" "g++-$compiler_major"
    fi
    export CUDACXX="$nvcc"
    export CC="$(command -v "gcc-$compiler_major")"
    export CXX="$(command -v "g++-$compiler_major")"
    export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-4}"
    local cuda_root
    cuda_root="$(dirname "$(dirname "$(readlink -f "$nvcc")")")"
    if [[ -d "$cuda_root/lib64" ]]; then
        export LD_LIBRARY_PATH="$cuda_root/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        export BER_CUDA_LIBRARY_PATH="$cuda_root/lib64"
    fi
    "$nvcc" --version
    "$CXX" --version
    # Version 4.6.0 has a source distribution with its vendored dependencies.
    # --no-binary and --reinstall-package prevent accidentally keeping a CPU wheel.
    "$UV" pip install --python "$PYTHON" --reinstall-package lightgbm --no-binary lightgbm \
        --config-settings=cmake.define.USE_CUDA=ON \
        --config-settings="cmake.define.CMAKE_CUDA_COMPILER=$CUDACXX" \
        --config-settings="cmake.define.CMAKE_CXX_COMPILER=$CXX" \
        -r "$PROJECT/requirements-model.txt"
    "$PYTHON" -m ber.cuda --gpu-device-id "$GPU_DEVICE_ID" --output "$stamp"
    printf '\nCUDA build and synthetic GPU preflight succeeded.\n'
}
