"""CUDA preflight run on the server, before expensive dataset preparation."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
from .common import write_json


def driver_info(device_id):
    if platform.system() != 'Linux': raise RuntimeError('LightGBM CUDA mode requires Linux')
    if device_id < 0: raise ValueError('GPU ID must be nonnegative')
    try:
        library = ctypes.CDLL('libcuda.so.1')
    except OSError as exc:
        raise RuntimeError('NVIDIA driver library is unavailable. The server administrator must provision a working driver/GPU container passthrough.') from exc
    def call(name,*args):
        result=getattr(library,name)(*args)
        if result != 0: raise RuntimeError(f'{name} failed with CUDA driver error {result}')
    call('cuInit',ctypes.c_uint(0))
    count=ctypes.c_int(); call('cuDeviceGetCount',ctypes.byref(count))
    if device_id>=count.value:
        raise RuntimeError(f'GPU ordinal {device_id} not available; {count.value} CUDA-visible devices. Check CUDA_VISIBLE_DEVICES.')
    device=ctypes.c_int(); call('cuDeviceGet',ctypes.byref(device),ctypes.c_int(device_id))
    name=ctypes.create_string_buffer(256); call('cuDeviceGetName',name,ctypes.c_int(256),device)
    major=ctypes.c_int(); minor=ctypes.c_int()
    call('cuDeviceComputeCapability',ctypes.byref(major),ctypes.byref(minor),device)
    if major.value<6: raise RuntimeError('LightGBM CUDA requires compute capability >= 6.0')
    memory=ctypes.c_size_t(); call('cuDeviceTotalMem_v2',ctypes.byref(memory),device)
    driver=ctypes.c_int(); call('cuDriverGetVersion',ctypes.byref(driver))
    return {'device_id':device_id,'name':name.value.decode('utf-8'),
        'compute_capability':f'{major.value}.{minor.value}','total_memory_bytes':memory.value,
        'cuda_driver_api_version':driver.value,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'visible_device_count':count.value}


def preflight(device_id=0):
    info=driver_info(device_id)
    import lightgbm as lgb
    print(f"CUDA preflight: {info['name']}, compute {info['compute_capability']}, {info['total_memory_bytes']/2**30:.1f} GiB VRAM, visible ordinal {device_id}",flush=True)
    rng=np.random.default_rng(42)
    x=rng.normal(size=(1024,32)).astype(np.float32)
    y=(x[:,0]+.5*x[:,1]>0).astype(np.int32)
    started=time.monotonic()
    # Three tiny synthetic trees verify the compiled backend, runtime, driver and
    # selected device together. This does not use any contest records.
    try:
        model=lgb.train({'objective':'binary','device_type':'cuda','gpu_device_id':device_id,
            'num_gpu':1,'max_bin':63,'num_leaves':7,'min_data_in_leaf':10,
            'num_threads':2,'verbosity':1,'seed':42},lgb.Dataset(x,label=y),num_boost_round=3)
        prediction=model.predict(x)
    except Exception as exc:
        raise RuntimeError('CUDA preflight failed. A CUDA-enabled LightGBM build and a compatible driver/toolkit are required. CPU fallback is disabled.') from exc
    if model.num_trees()<1 or not np.isfinite(prediction).all():
        raise RuntimeError('CUDA preflight produced no valid model/predictions')
    # Hash the actual loaded binary, distinguishing CPU/CUDA builds of one version.
    from lightgbm.basic import _LIB
    binary=Path(_LIB._name)
    info.update(lightgbm_version=lgb.__version__,lightgbm_binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        preflight_seconds=time.monotonic()-started,status='passed',cuda_library_path=os.environ.get('BER_CUDA_LIBRARY_PATH'))
    print('CUDA PREFLIGHT PASSED: LightGBM successfully trained on the selected GPU.',flush=True)
    return info


def telemetry():
    try:
        text=subprocess.check_output(['nvidia-smi',
            '--query-gpu=index,name,utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader,nounits'],text=True,stderr=subprocess.DEVNULL,timeout=5).strip()
        return text.replace('\n','; ')
    except (OSError,subprocess.SubprocessError): return 'GPU telemetry unavailable'


def build_is_verified(path):
    try:
        info=json.loads(Path(path).read_text(encoding='utf-8'))
        import lightgbm as lgb
        from lightgbm.basic import _LIB
        return (info['status']=='passed' and info['lightgbm_version']==lgb.__version__=='4.6.0'
            and info['lightgbm_binary_sha256']==hashlib.sha256(Path(_LIB._name).read_bytes()).hexdigest())
    except (OSError,ValueError,KeyError,ImportError): return False


def main():
    parser=argparse.ArgumentParser(description='Verify LightGBM CUDA with a tiny synthetic fit; no contest data is used')
    parser.add_argument('--gpu-device-id',type=int,default=0)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--verify-build',type=Path)
    args=parser.parse_args()
    if args.verify_build:
        raise SystemExit(0 if build_is_verified(args.verify_build) else 1)
    report=preflight(args.gpu_device_id)
    if args.output: write_json(args.output,report)
    else: print(json.dumps(report,indent=2))


if __name__=='__main__': main()
