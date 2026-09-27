"""One-command server orchestration; invoked by the Ubuntu Bash launcher."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import sys
import time

from .common import fingerprint, load_config, write_json
from .features import FEATURE_NAMES
from .store import check_index, connect
from .backend import configure_backend

LOG = logging.getLogger(__name__)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def find_data(root, requested=None):
    choices = [Path(requested).expanduser()] if requested else [
        root / 'dataset', root / 'student_resource', root.parent / 'student_resource']
    for candidate in choices:
        if (candidate / 'dataset/train/train_source1.tsv').is_file():
            candidate = candidate / 'dataset'
        if (candidate / 'train/train_source1.tsv').is_file():
            return candidate.resolve()
    raise ValueError('Dataset not found. Use --data /path/to/student_resource or place student_resource/ beside run_training.sh')


def data_identity(data, split):
    names = [f'{split}_source{i}.tsv' for i in (1, 2, 3)]
    if split == 'train': names.append('train_ground_truth.tsv')
    result = {}
    for name in names:
        path = data / split / name
        if not path.is_file(): raise FileNotFoundError(path)
        LOG.info('Checking input fingerprint: %s', path)
        result[name] = {'bytes': path.stat().st_size, 'sha256': sha256(path)}
    return result


def preserve_incomplete(path):
    """Preserve partial artifacts; never recursively delete user files."""
    path = Path(path)
    if not path.exists(): return
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination = path.with_name(path.name + '.interrupted-' + stamp)
    path.rename(destination)
    for suffix in ('-wal', '-shm'):
        sidecar = Path(str(path) + suffix)
        if sidecar.exists(): sidecar.rename(Path(str(destination) + suffix))
    LOG.warning('Preserved incomplete artifact at %s', destination)


def resources(run):
    available = None
    if Path('/proc/meminfo').exists():
        info = {line.split(':')[0]: line.split(':')[1].strip() for line in Path('/proc/meminfo').read_text().splitlines()}
        available = int(info['MemAvailable'].split()[0]) * 1024
        # Respect container memory ceilings when provided by cgroup v2.
        try:
            cap = Path('/sys/fs/cgroup/memory.max').read_text().strip()
            used = int(Path('/sys/fs/cgroup/memory.current').read_text().strip())
            if cap != 'max': available = min(available, max(0, int(cap) - used))
        except (FileNotFoundError, ValueError, PermissionError): pass
    disk = shutil.disk_usage(run).free / 2**30
    result=f'free disk={disk:.1f} GiB' + (f', available RAM={available / 2**30:.1f} GiB' if available is not None else '')
    if os.environ.get('BER_ACTIVE_DEVICE')=='cuda':
        from .cuda import telemetry
        result += ' | GPUs [index,name,util%,used MiB,total MiB]: '+telemetry()
    return result


def run_stage(name, args, run):
    started = time.monotonic()
    LOG.info('========== %s STARTED | %s ==========', name, resources(run))
    command = [sys.executable, '-u', '-m', 'ber.cli', *map(str, args)]
    process = subprocess.Popen(command)
    try:
        while True:
            try:
                code = process.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                LOG.info('%s running: elapsed %.1f minutes; %s', name, (time.monotonic()-started)/60, resources(run))
    except BaseException:
        process.terminate()
        try: process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait()
        raise
    if code != 0:
        raise RuntimeError(f'{name} failed with exit code {code}; re-run the same command after fixing the reported problem')
    LOG.info('========== %s FINISHED in %.1f minutes ==========', name, (time.monotonic()-started)/60)


def ensure_index(data, split, path, config_path, config, run):
    ready = False
    if path.exists():
        db = None
        try:
            db = connect(path)
            check_index(db, config, split)
            ready = True
        except (ValueError, sqlite3.DatabaseError, KeyError):
            pass
        finally:
            if db is not None: db.close()
    if ready:
        LOG.info('Reusing completed %s index: %s', split, path)
    else:
        preserve_incomplete(path)
        run_stage(f'INDEX {split.upper()}', ['index','--data',data,'--split',split,'--index',path,'--config',config_path], run)


def verify_model(folder, config):
    required = ['complete.json','model.txt','model_metadata.json','tune_metrics.json',
                'audit_metrics.json','trials.json','feature_importance.json']
    for name in required:
        if not (folder / name).is_file() or not (folder / name).stat().st_size:
            raise ValueError(f'Training artifact missing or empty: {folder/name}')
    if read_json(folder/'complete.json') != {'complete': True, 'trained': True}:
        raise ValueError('Model completion marker is invalid')
    metadata = read_json(folder/'model_metadata.json')
    if metadata['config'] != config or metadata['features'] != FEATURE_NAMES:
        raise ValueError('Saved model configuration/feature schema does not match this run')
    for name in ['tune_metrics.json','audit_metrics.json']:
        metrics = read_json(folder/name)
        if not 0 <= metrics['macro_f05'] <= 1 or metrics['references'] < 1:
            raise ValueError(f'Invalid evaluation report: {name}')
    import lightgbm as lgb
    booster = lgb.Booster(model_file=str(folder/'model.txt'))
    if booster.feature_name() != FEATURE_NAMES:
        raise ValueError('Serialized model does not contain the expected features')
    return read_json(folder/'audit_metrics.json')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Full Ubuntu CPU/CUDA training pipeline with terminal progress and restart support')
    parser.add_argument('--project-root',type=Path,required=True)
    parser.add_argument('--data')
    parser.add_argument('--run-dir',type=Path)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--threads',type=int)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--gpu-device-id',type=int,default=0)
    parser.add_argument('--max-bin',type=int)
    parser.add_argument('--predict',action='store_true')
    args = parser.parse_args(argv)
    def terminate(signum,frame):
        raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM,terminate)
    root = args.project_root.resolve()
    project = root/'code/business_entity_resolution'
    default_run='full-training-cuda' if args.device=='cuda' else 'full-training'
    run = (args.run_dir or root/'runs'/default_run).expanduser().resolve()
    os.environ['BER_ACTIVE_DEVICE']=args.device
    run.mkdir(parents=True,exist_ok=True)
    (run/'logs').mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(),logging.FileHandler(run/'logs/pipeline.log',encoding='utf-8')])
    # Serialize access even when different copies of the shell launcher use one run dir.
    import fcntl
    with (run/'.run.lock').open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError(f'Another pipeline owns {run}')
        started = time.monotonic()
        try:
            if (run/'SUCCESS.json').exists():
                stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                (run/'SUCCESS.json').rename(run/f'previous-success-{stamp}.json')
            data = find_data(root,args.data)
            config = load_config(args.config or project/'configs/default.json')
            configure_backend(config,args.device,args.gpu_device_id,args.max_bin)
            saved = run/'effective_config.json'
            if args.threads is not None:
                if args.threads < 1: raise ValueError('--threads must be positive')
                config['threads'] = args.threads
            elif saved.exists():
                config['threads'] = read_json(saved)['threads']
            else:
                cpus = len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else (os.cpu_count() or 1)
                config['threads'] = min(cpus,16)
            if config['max_references_per_split'] is not None:
                raise ValueError('Full-training launcher requires max_references_per_split=null; use stage commands for development runs')
            from .text import require_fast_text
            require_fast_text()
            hardware={'device_type':args.device}
            if args.device=='cuda':
                from .cuda import preflight
                hardware.update(preflight(args.gpu_device_id))
                LOG.info('CUDA ready: %s | max_bin=%s | original macro F0.5 selection retained',hardware['name'],config['max_bin'])
            packages = {name:importlib.metadata.version(name) for name in ['numpy','lightgbm','rapidfuzz','anyascii','scipy']}
            code = {str(p.relative_to(project)):sha256(p) for p in sorted((project/'src/ber').glob('*.py'))}
            for path in project.glob('requirements*.txt'): code[path.name]=sha256(path)
            from lightgbm.basic import _LIB
            identity = {'train_data':data_identity(data,'train'),'config':config,'code':code,'packages':packages,
                        'lightgbm_binary_sha256':sha256(Path(_LIB._name))}
            manifest = run/'run_manifest.json'
            if manifest.exists() and read_json(manifest) != identity:
                raise ValueError('Data, configuration, dependencies, or code changed. Choose a new --run-dir; existing artifacts are preserved.')
            if not manifest.exists() and any((run/p).exists() for p in ['train.sqlite','prepared','model']):
                raise ValueError('Run directory has artifacts without a run manifest; use a new --run-dir')
            write_json(manifest,identity)
            write_json(saved,config)
            write_json(run/'hardware.json',hardware)
            freeze = subprocess.check_output([sys.executable,'-c',
                'import importlib.metadata as m; print("\\n".join(sorted(d.metadata["Name"]+"=="+d.version for d in m.distributions())))'],text=True)
            (run/'environment-freeze.txt').write_text(freeze,encoding='utf-8')
            LOG.info('Dataset: %s | Artifacts: %s | CPU threads: %s | Training backend: %s',data,run,config['threads'],args.device)
            LOG.info('Resource status: %s. Full-data feature arrays and indices can require tens of GB.',resources(run))
            ensure_index(data,'train',run/'train.sqlite',saved,config,run)
            run_stage('FEATURE PREPARATION',['prepare','--index',run/'train.sqlite','--out',run/'prepared','--config',saved],run)
            if not (run/'model/complete.json').exists():
                run_stage('LIGHTGBM MODEL SELECTION AND AUDIT',['train','--prepared',run/'prepared','--out',run/'model','--config',saved],run)
            else:
                LOG.info('Reusing completed training run')
            audit = verify_model(run/'model',config)
            LOG.info('TRAINING COMPLETE | audit macro F0.5=%.6f | threshold=%.8f | model=%s',audit['macro_f05'],audit['threshold'],run/'model/model.txt')
            if args.predict:
                test_identity = data_identity(data,'test')
                test_manifest = run/'test_data.json'
                if test_manifest.exists() and read_json(test_manifest)!=test_identity:
                    raise ValueError('Test data changed; choose a new run directory')
                write_json(test_manifest,test_identity)
                ensure_index(data,'test',run/'test.sqlite',saved,config,run)
                output = run/'output'
                if not (output/'prediction_summary.json').exists():
                    preserve_incomplete(output)
                    run_stage('TEST PREDICTION',['predict','--index',run/'test.sqlite','--model',run/'model','--out',output],run)
                run_stage('OUTPUT VALIDATION',['validate','--index',run/'test.sqlite','--output',output],run)
            summary = {'status':'complete','model':str(run/'model/model.txt'),'audit_metrics':audit,
                'predictions_requested':args.predict,'elapsed_seconds':time.monotonic()-started,
                'completed_utc':datetime.now(timezone.utc).isoformat()}
            write_json(run/'SUCCESS.json',summary)
            LOG.info('FULL PIPELINE COMPLETED SUCCESSFULLY | elapsed %.1f minutes | results: %s',summary['elapsed_seconds']/60,run)
        except BaseException as exc:
            write_json(run/'last_failure.json',{'error':str(exc),'type':type(exc).__name__,'utc':datetime.now(timezone.utc).isoformat()})
            LOG.exception('Pipeline stopped. No success is claimed for this invocation.')
            raise


if __name__ == '__main__': main()
