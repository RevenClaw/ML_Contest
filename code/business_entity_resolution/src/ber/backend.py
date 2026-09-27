"""Explicit CPU/CUDA settings shared by orchestration, training and tests."""

RESERVED_TRIAL_KEYS = {
    'device', 'device_type', 'gpu_device_id', 'gpu_platform_id', 'num_gpu',
    'gpu_use_dp', 'deterministic', 'force_col_wise', 'force_row_wise',
    'max_bin', 'max_bin_by_feature', 'objective', 'application', 'metric',
}


def configure_backend(config, device, gpu_device_id=0, max_bin=None):
    if device not in ('cpu', 'cuda'):
        raise ValueError('Backend must be cpu or cuda (gpu/OpenCL is not this pipeline)')
    if gpu_device_id < 0:
        raise ValueError('GPU device ID must be a nonnegative CUDA-visible ordinal')
    config['device_type'] = device
    config['gpu_device_id'] = gpu_device_id
    config['max_bin'] = max_bin if max_bin is not None else (63 if device == 'cuda' else config.get('max_bin', 127))
    if config['max_bin'] < 2:
        raise ValueError('max_bin must be at least 2')
    return config


def model_parameters(config, extra):
    invalid = RESERVED_TRIAL_KEYS.intersection(extra)
    if invalid:
        raise ValueError(f'Trial overrides protected settings: {sorted(invalid)}; use top-level configuration')
    device = config.get('device_type', 'cpu')
    if device not in ('cpu', 'cuda'): raise ValueError(f'Unsupported backend: {device}')
    params = {
        'objective':'binary', 'metric':'None', 'verbosity':1 if device=='cuda' else -1,
        'num_threads':config['threads'], 'seed':config['seed'],
        'feature_fraction_seed':config['seed'], 'bagging_seed':config['seed'],
        'data_random_seed':config['seed'], 'feature_pre_filter':False,
        'max_bin':config.get('max_bin',63 if device=='cuda' else 127),
        **extra, 'device_type':device,
    }
    if device == 'cuda':
        params.update(gpu_device_id=config.get('gpu_device_id',0),num_gpu=1)
        # deterministic / force_col_wise are CPU settings. gpu_use_dp is OpenCL-only.
    else:
        params.update(deterministic=True,force_col_wise=True)
    return params
