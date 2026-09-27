import csv
import hashlib
import json
import os
from pathlib import Path

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def read_tsv(path, columns):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != columns:
            raise ValueError(f"{path}: expected columns {columns}, got {reader.fieldnames}")
        for line, row in enumerate(reader, 2):
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f"{path}:{line}: malformed TSV")
            yield row


def stable_number(value, seed=42):
    return int.from_bytes(hashlib.blake2b(f"{seed}:{value}".encode(), digest_size=8).digest(), "big")


def split_for(entity_id, config):
    u = stable_number(entity_id, config["seed"]) / 2**64
    if u < config["audit_fraction"]:
        return "audit"
    if u < config["audit_fraction"] + config["tune_fraction"]:
        return "tune"
    return "train"


def write_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if not 0 < config["tune_fraction"] + config["audit_fraction"] < 1:
        raise ValueError("Holdout fractions must sum to a value between zero and one")
    if min(config["tune_fraction"], config["audit_fraction"]) <= 0:
        raise ValueError("Both tune and audit splits are required")
    if config["train_hard_negative_limit"] > config["train_negative_limit"]:
        raise ValueError("Hard-negative limit exceeds total negative limit")
    if config["train_negative_limit"] < 1:
        raise ValueError("At least one negative per reference must be allowed")
    if any(v < 1 for v in config["retrieval"].values()):
        raise ValueError("Retrieval limits must be positive")
    if not config["trials"] or not config["threshold_grid"]:
        raise ValueError("At least one trial and threshold are required")
    if config['threads'] < 1 or config['num_boost_round'] < 1 or config['early_stopping_rounds'] < 1:
        raise ValueError('Thread and iteration counts must be positive')
    if any(not 0 <= t <= 1 for t in config['threshold_grid']):
        raise ValueError('Threshold grid must be within [0, 1]')
    from .backend import model_parameters
    for trial in config['trials']: model_parameters(config,trial)
    if config.get('max_bin',127)<2 or config.get('gpu_device_id',0)<0:
        raise ValueError('Invalid max_bin or GPU device ID')
    return config


def fingerprint(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
