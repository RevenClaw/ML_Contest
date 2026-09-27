"""Disk-backed records and multi-route full-text retrieval; no network access."""
import json
import logging
import sqlite3
from pathlib import Path
from .common import SOURCE_COLUMNS, fingerprint, read_tsv, split_for
from .text import forms, gram_document

LOG = logging.getLogger(__name__)


def connect(path, readonly=True):
    if readonly:
        db = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    else:
        db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA cache_size=-65536")
    return db


def build_index(data_dir, split, path, config):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}; choose a new run path")
    path.parent.mkdir(parents=True, exist_ok=True)
    db = connect(path, False)
    db.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE refs(id TEXT PRIMARY KEY,name TEXT,address TEXT,country TEXT,fold TEXT);
        CREATE TABLE truth(sid TEXT,tid TEXT UNIQUE,PRIMARY KEY(sid,tid));
        CREATE TABLE records(id TEXT UNIQUE,name TEXT,address TEXT,country TEXT,
            fold TEXT,n TEXT,a TEXT,grams TEXT);
        CREATE VIRTUAL TABLE search USING fts5(n,a,grams,country,fold,
            content='records',content_rowid='rowid',tokenize='unicode61 remove_diacritics 0');
    """)
    db.execute("INSERT INTO meta VALUES('config',?)", (json.dumps(config, sort_keys=True),))
    db.execute("INSERT INTO meta VALUES('fingerprint',?)", (fingerprint(config),))
    db.execute("INSERT INTO meta VALUES('split',?)", (split,))
    sources = []
    def records(source):
        file = Path(data_dir) / split / f"{split}_source{source}.tsv"
        stat = file.stat()
        sources.append({"path": str(file.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        for r in read_tsv(file, SOURCE_COLUMNS):
            if not r['entity_id'].startswith(f"S{source}-") or not r['country'].strip():
                raise ValueError(f"Invalid ID or empty country in {file}: {r['entity_id']}")
            yield r
    try:
        for i, r in enumerate(records(1), 1):
            fold = split_for(r['entity_id'], config) if split == 'train' else 'test'
            db.execute("INSERT INTO refs VALUES(?,?,?,?,?)", (r['entity_id'],r['business_name'],r['business_address'],r['country'],fold))
            if i % 100000 == 0:
                db.commit(); LOG.info("Indexed %s references", i)
        db.commit()
        if split == 'train':
            gt = Path(data_dir) / 'train/train_ground_truth.tsv'
            sources.append({"path": str(gt.resolve()), "bytes": gt.stat().st_size, "mtime_ns": gt.stat().st_mtime_ns})
            db.execute("CREATE TABLE labeled(sid TEXT PRIMARY KEY)")
            for i, r in enumerate(read_tsv(gt, ['source1_entity_id','matched_entity_ids']), 1):
                sid = r['source1_entity_id']
                db.execute("INSERT INTO labeled VALUES(?)", (sid,))
                for tid in r['matched_entity_ids'].split(',') if r['matched_entity_ids'] else []:
                    if not tid.startswith(('S2-', 'S3-')):
                        raise ValueError(f"Invalid truth target {tid}")
                    # A shared target would invalidate entity-disjoint folds. Fail explicitly.
                    db.execute("INSERT INTO truth VALUES(?,?)", (sid,tid))
                if i % 100000 == 0: db.commit()
            if db.execute("SELECT 1 FROM refs LEFT JOIN labeled ON refs.id=labeled.sid WHERE labeled.sid IS NULL LIMIT 1").fetchone() or db.execute("SELECT 1 FROM labeled LEFT JOIN refs ON refs.id=labeled.sid WHERE refs.id IS NULL LIMIT 1").fetchone():
                raise ValueError("Ground-truth rows do not exactly cover Source 1")
        for source in (2,3):
            for i, r in enumerate(records(source), 1):
                _, _, n, a, core = forms(r['business_name'], r['business_address'])
                db.execute("INSERT INTO records VALUES(?,?,?,?,?,?,?,?)", (r['entity_id'],r['business_name'],r['business_address'],r['country'],'unowned' if split=='train' else 'test',n,a,gram_document(core)))
                if i % 100000 == 0:
                    db.commit(); LOG.info("Indexed source %s: %s records", source, i)
            db.commit()
        if split == 'train':
            missing = db.execute("SELECT tid FROM truth LEFT JOIN records ON records.id=truth.tid WHERE records.id IS NULL LIMIT 1").fetchone()
            if missing: raise ValueError(f"Ground truth references missing target {missing[0]}")
            LOG.info("Assigning target owners to prevent holdout leakage")
            db.execute("UPDATE records SET fold=COALESCE((SELECT refs.fold FROM truth JOIN refs ON refs.id=truth.sid WHERE truth.tid=records.id),'unowned')")
        LOG.info("Building text index (may require substantial disk space and time)")
        db.execute("INSERT INTO search(search) VALUES('rebuild')")
        db.execute("CREATE INDEX exact_name ON records(country,n)")
        db.execute("CREATE INDEX exact_address ON records(country,a)")
        db.execute("CREATE INDEX refs_fold ON refs(fold,id)")
        db.execute("CREATE VIRTUAL TABLE vocabulary USING fts5vocab(search,'col')")
        db.execute("INSERT INTO meta VALUES('sources',?)", (json.dumps(sources),))
        db.execute("INSERT INTO meta VALUES('complete','yes')")
        db.commit()
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        db.close()


def check_index(db, config, expected=None):
    meta = dict(db.execute("SELECT key,value FROM meta"))
    if meta.get('complete') != 'yes': raise ValueError("Index build incomplete; rebuild into a new path")
    if meta.get('fingerprint') != fingerprint(config):
        # Retrieval/model settings may change without invalidating the stored split/normalization.
        original = json.loads(meta['config'])
        for key in ('seed','tune_fraction','audit_fraction'):
            if original[key] != config[key]: raise ValueError(f"Index split configuration differs: {key}")
    if expected and meta['split'] != expected: raise ValueError(f"Expected {expected} index")
    return meta
