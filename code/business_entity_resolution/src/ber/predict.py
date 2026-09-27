import csv
import json
import logging
from pathlib import Path
import numpy as np
from .common import write_json
from .features import FEATURE_NAMES,pair_features
from .retrieval import Retriever
from .store import check_index,connect


def predict(index, model_dir, output):
    import lightgbm as lgb
    model_dir=Path(model_dir); output=Path(output)
    if not (model_dir/'complete.json').exists(): raise ValueError('Training did not complete')
    meta=json.loads((model_dir/'model_metadata.json').read_text(encoding='utf-8'))
    if meta['features']!=FEATURE_NAMES: raise ValueError('Model feature schema mismatch')
    config=meta['config']; threshold=meta['selection']['threshold']
    output.mkdir(parents=True,exist_ok=True)
    for name in ['matching_results.tsv','candidate_pairs.tsv','matching_results.tsv.partial','candidate_pairs.tsv.partial']:
        if (output/name).exists(): raise FileExistsError(f'Refusing to overwrite {output/name}')
    db=connect(index); check_index(db,config,'test')
    retriever=Retriever(db,config['retrieval'])
    model=lgb.Booster(model_file=str(model_dir/'model.txt'))
    paths=[output/'matching_results.tsv.partial',output/'candidate_pairs.tsv.partial']
    n=total=matches=0
    try:
        with paths[0].open('w',encoding='utf-8',newline='') as mh,paths[1].open('w',encoding='utf-8',newline='') as ch:
            mw=csv.writer(mh,delimiter='\t',lineterminator='\n'); cw=csv.writer(ch,delimiter='\t',lineterminator='\n')
            mw.writerow(['source1_entity_id','matched_entity_ids']); cw.writerow(['source1_entity_id','candidate_entity_ids'])
            pending=[]; features=[]
            def flush():
                nonlocal n,total,matches
                scores=model.predict(np.asarray(features,dtype=np.float32),num_threads=config['threads']) if features else np.empty(0)
                offset=0
                for sid,ids in pending:
                    local=scores[offset:offset+len(ids)]; offset+=len(ids)
                    accepted=sorted(tid for tid,s in zip(ids,local) if s>=threshold)
                    mw.writerow([sid,','.join(accepted)])
                    cw.writerow([sid,','.join(sorted(ids))])
                    n+=1; total+=len(ids); matches+=len(accepted)
                pending.clear(); features.clear()
            for row in db.execute('SELECT * FROM refs ORDER BY id'):
                ref=dict(row); candidates=retriever.get(ref)
                ids=[]
                for target,route in candidates:
                    ids.append(target['id']); features.append(pair_features(ref,target,route,len(candidates),retriever))
                pending.append((ref['id'],ids))
                if len(features)>=50000 or len(pending)>=1000:
                    flush(); logging.info('Predicted %s references',n)
            flush()
        paths[0].rename(output/'matching_results.tsv'); paths[1].rename(output/'candidate_pairs.tsv')
        write_json(output/'prediction_summary.json',{'references':n,'candidate_pairs':total,'matched_pairs':matches,'threshold':threshold})
    finally: db.close()

