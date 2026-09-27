import csv
import json
import logging
import os
import time
from contextlib import ExitStack
from pathlib import Path
import numpy as np
from .common import fingerprint, stable_number, write_json
from .features import FEATURE_NAMES, pair_features
from .retrieval import Retriever
from .store import check_index, connect

LOG=logging.getLogger(__name__)
CHECKPOINT_EVERY=1000


def preparation_signature(config):
    return fingerprint({k:config[k] for k in ['seed','tune_fraction','audit_fraction','retrieval','train_negative_limit','train_hard_negative_limit','max_references_per_split']})


def prepare(index, destination, config):
    destination=Path(destination)
    db=connect(index)
    index_meta=check_index(db,config,'train')
    expected={'signature':preparation_signature(config),'features':FEATURE_NAMES,
              'index_sources':json.loads(index_meta['sources'])}
    destination.mkdir(parents=True,exist_ok=True)
    state=destination/'state.json'
    if state.exists():
        if json.loads(state.read_text(encoding='utf-8'))!=expected:
            db.close(); raise ValueError('Preparation configuration or input index changed; use a new destination')
    elif any(destination.iterdir()):
        db.close(); raise ValueError('Preparation destination contains unrecognized files; use a new destination')
    else: write_json(state,expected)
    if (destination/'manifest.json').exists():
        for fold in ['train','tune','audit']:
            validate_prepared_files(destination/fold)
        LOG.info('Reusing completed feature preparation: %s',destination)
        db.close(); return
    retriever=Retriever(db,config['retrieval'])
    try:
        for fold in ['train','tune','audit']:
            folder=destination/fold; folder.mkdir(exist_ok=True)
            if (folder/'manifest.json').exists():
                validate_prepared_files(folder)
                LOG.info('Reusing completed %s feature split',fold)
                continue
            checkpoint_path=folder/'checkpoint.json'
            checkpoint=json.loads(checkpoint_path.read_text(encoding='utf-8')) if checkpoint_path.exists() else None
            pair_count=checkpoint['pairs'] if checkpoint else 0
            group_count=checkpoint['references'] if checkpoint else 0
            positive_count=checkpoint['retrieved_true_pairs'] if checkpoint else 0
            true_count=checkpoint['true_pairs'] if checkpoint else 0
            last_id=checkpoint['last_id'] if checkpoint else ''
            total=db.execute('SELECT COUNT(*) FROM refs WHERE fold=?',(fold,)).fetchone()[0]
            if config['max_references_per_split'] is not None: total=min(total,config['max_references_per_split'])
            LOG.info('%s preparation: %s/%s references complete; resuming after %s',fold,group_count,total,last_id or '(start)')
            initial_count=group_count; started=time.monotonic()
            with ExitStack() as stack:
                names=['features.f32','labels.u8','groups.i32','references.tsv','pairs.tsv']
                for name in names:
                    path=folder/name
                    if checkpoint:
                        offset=checkpoint['offsets'][name]
                        if not path.exists() or path.stat().st_size<offset:
                            raise ValueError(f'Checkpoint data truncated: {path}')
                        with path.open('r+b') as handle: handle.truncate(offset)
                mode='ab' if checkpoint else 'wb'
                x=stack.enter_context((folder/names[0]).open(mode))
                y=stack.enter_context((folder/names[1]).open(mode))
                g=stack.enter_context((folder/names[2]).open(mode))
                mode='a' if checkpoint else 'w'
                gh=stack.enter_context((folder/names[3]).open(mode,encoding='utf-8',newline=''))
                ph=stack.enter_context((folder/names[4]).open(mode,encoding='utf-8',newline=''))
                gw=csv.writer(gh,delimiter='\t',lineterminator='\n')
                pw=csv.writer(ph,delimiter='\t',lineterminator='\n')
                if not checkpoint:
                    gw.writerow(['source1_entity_id','country','true_count','candidate_count','retrieved_true_count'])
                    pw.writerow(['source1_entity_id','target_entity_id'])
                xb=[]; yb=[]; gb=[]
                def flush():
                    if xb:
                        np.asarray(xb,dtype=np.float32).tofile(x)
                        np.asarray(yb,dtype=np.uint8).tofile(y)
                        np.asarray(gb,dtype=np.int32).tofile(g)
                        xb.clear(); yb.clear(); gb.clear()
                def save_checkpoint():
                    flush()
                    for handle in [x,y,g,gh,ph]:
                        handle.flush(); os.fsync(handle.fileno())
                    write_json(checkpoint_path,{'pairs':pair_count,'references':group_count,
                        'retrieved_true_pairs':positive_count,'true_pairs':true_count,'last_id':last_id,
                        'offsets':{name:(folder/name).stat().st_size for name in names}})
                # ID order is independent of input-file geography/order; cap is only for development.
                limit=config['max_references_per_split']
                query='SELECT * FROM refs WHERE fold=? AND id>? ORDER BY id'
                args=[fold,last_id]
                if limit is not None: query+=' LIMIT ?'; args.append(max(0,int(limit)-group_count))
                for refrow in db.execute(query,args):
                    ref=dict(refrow)
                    truth={r[0] for r in db.execute('SELECT tid FROM truth WHERE sid=?',(ref['id'],))}
                    candidates=retriever.get(ref,training=(fold=='train'))
                    hits=sum(t['id'] in truth for t,_ in candidates)
                    gw.writerow([ref['id'],ref['country'],len(truth),len(candidates),hits])
                    true_count+=len(truth); positive_count+=hits
                    selected=candidates
                    if fold=='train':
                        positives=[c for c in candidates if c[0]['id'] in truth]
                        negatives=[c for c in candidates if c[0]['id'] not in truth]
                        hard=config['train_hard_negative_limit']
                        remainder=sorted(negatives[hard:],key=lambda c:stable_number(ref['id']+c[0]['id'],config['seed']))
                        selected=positives+negatives[:hard]+remainder[:config['train_negative_limit']-hard]
                    for target,route in selected:
                        xb.append(pair_features(ref,target,route,len(candidates),retriever))
                        yb.append(target['id'] in truth); gb.append(group_count)
                        pw.writerow([ref['id'],target['id']])
                        if len(xb)>=8192: flush()
                    pair_count+=len(selected); group_count+=1
                    last_id=ref['id']
                    if group_count%CHECKPOINT_EVERY==0:
                        save_checkpoint()
                        rate=(group_count-initial_count)/max(time.monotonic()-started,.001)
                        LOG.info('%s: %s/%s references (%.2f%%), %s pairs, %.2f refs/s, estimated %.1f hours left',fold,group_count,total,100*group_count/max(total,1),pair_count,rate,(total-group_count)/max(rate,.001)/3600)
                save_checkpoint()
            write_json(folder/'manifest.json',{'references':group_count,'pairs':pair_count,'features':FEATURE_NAMES,'true_pairs':true_count,'retrieved_true_pairs':positive_count,'negative_sampling':fold=='train'})
            LOG.info('%s features complete: %s references, %s pairs, blocking pair recall=%s',fold,group_count,pair_count,positive_count/true_count if true_count else 'no positives')
        write_json(destination/'manifest.json',{'complete':True,'signature':preparation_signature(config),'config':config,'index':str(Path(index).resolve()),'features':FEATURE_NAMES})
    finally: db.close()


def validate_prepared_files(folder):
    folder=Path(folder)
    meta=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    if meta['features']!=FEATURE_NAMES: raise ValueError('Feature schema changed; regenerate prepared data')
    for name,width in [('features.f32',4*len(FEATURE_NAMES)),('labels.u8',1),('groups.i32',4)]:
        if (folder/name).stat().st_size!=meta['pairs']*width:
            raise ValueError(f'Prepared array size mismatch: {folder/name}')
    return meta


class Prepared:
    def __init__(self, folder):
        self.folder=Path(folder)
        meta=validate_prepared_files(self.folder)
        self.size=meta['pairs']
        self.x=np.memmap(self.folder/'features.f32',mode='r',dtype=np.float32,shape=(self.size,len(FEATURE_NAMES))) if self.size else np.empty((0,len(FEATURE_NAMES)),dtype=np.float32)
        self.y=np.memmap(self.folder/'labels.u8',mode='r',dtype=np.uint8,shape=(self.size,)) if self.size else np.empty(0,dtype=np.uint8)
        self.groups=np.memmap(self.folder/'groups.i32',mode='r',dtype=np.int32,shape=(self.size,)) if self.size else np.empty(0,dtype=np.int32)
        self.ids=[]; self.countries=[]; truth=[]
        with (self.folder/'references.tsv').open(encoding='utf-8',newline='') as f:
            for r in csv.DictReader(f,delimiter='\t'):
                self.ids.append(r['source1_entity_id']); self.countries.append(r['country']); truth.append(int(r['true_count']))
        self.truth=np.asarray(truth,dtype=np.int64)
        if len(self.truth)!=meta['references']: raise ValueError('Reference manifest is truncated')

    def close(self):
        for value in (self.x,self.y,self.groups):
            if isinstance(value,np.memmap) and not value._mmap.closed: value._mmap.close()
