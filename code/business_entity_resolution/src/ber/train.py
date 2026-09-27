"""Explicitly invoked training; never called by index, prepare, or tests."""
import json
import logging
import gc
import shutil
import time
from pathlib import Path
import numpy as np
from .common import fingerprint, write_json
from .features import FEATURE_NAMES
from .metrics import best_grid_f05, optimize_threshold, report
from .prepare import Prepared, preparation_signature
from .backend import model_parameters


def train(prepared, destination, config):
    import lightgbm as lgb
    prepared=Path(prepared); destination=Path(destination)
    manifest=json.loads((prepared/'manifest.json').read_text(encoding='utf-8'))
    if not manifest.get('complete') or manifest['signature']!=preparation_signature(config):
        raise ValueError('Prepared data incomplete or preprocessing configuration changed')
    state={'config':config,'prepared_manifest':manifest,'features':FEATURE_NAMES}
    destination.mkdir(parents=True,exist_ok=True)
    state_path=destination/'state.json'
    if state_path.exists():
        if json.loads(state_path.read_text(encoding='utf-8'))!=state:
            raise ValueError('Training configuration or prepared data changed; use a new destination')
    elif any(destination.iterdir()):
        raise ValueError('Model directory contains unrecognized artifacts; use a new destination')
    else: write_json(state_path,state)
    if (destination/'complete.json').exists():
        logging.info('Training already complete: %s',destination)
        return
    logging.info('Opening training and tuning matrices | backend=%s | max_bin=%s',config.get('device_type','cpu'),config.get('max_bin',127))
    training,tune=Prepared(prepared/'train'),Prepared(prepared/'tune')
    if not training.size or not tune.size or not len(tune.truth): raise ValueError('Training/tuning data are empty')
    if len(np.unique(training.y))!=2: raise ValueError('Training needs positive and negative examples')
    # Equal total pair weight per reference approximates the macro objective.
    counts=np.bincount(training.groups,minlength=len(training.truth))
    weights=1/np.maximum(1,counts[training.groups])
    weights=weights/weights.mean()
    logging.info('Training: %s pairs; tuning: %s pairs; %s features. Building LightGBM datasets.',training.size,tune.size,len(FEATURE_NAMES))
    train_set=lgb.Dataset(training.x,label=training.y,weight=weights,feature_name=FEATURE_NAMES,free_raw_data=False)
    valid_set=lgb.Dataset(tune.x,label=tune.y,reference=train_set,feature_name=FEATURE_NAMES,free_raw_data=False)
    def metric(scores,dataset):
        value=best_grid_f05(tune.y,scores,tune.groups,tune.truth,config['threshold_grid'])
        return 'macro_f05_threshold_grid',value,True
    trials=[]; best=None
    for number,extra in enumerate(config['trials']):
        trial_record=destination/f'trial_{number}.json'
        trial_model=destination/f'trial_{number}.txt'
        if trial_record.exists() and trial_model.exists():
            item=json.loads(trial_record.read_text(encoding='utf-8'))
            trials.append(item)
            logging.info('Reusing completed trial %s/%s: macro F0.5=%.6f',number+1,len(config['trials']),item['macro_f05'])
            if best is None or item['macro_f05']>best['macro_f05']:
                best=item
                shutil.copyfile(trial_model,destination/'model.txt')
                write_json(destination/'tune_metrics.json',item['tune_report'])
            continue
        logging.info('Trial %s/%s starting: %s',number+1,len(config['trials']),extra)
        began=time.monotonic()
        params=model_parameters(config,extra)
        model=lgb.train(params,train_set,num_boost_round=config['num_boost_round'],valid_sets=[valid_set],valid_names=['tune'],feval=metric,
            callbacks=[lgb.early_stopping(config['early_stopping_rounds'],first_metric_only=True),lgb.log_evaluation(25)])
        scores=model.predict(tune.x,num_iteration=model.best_iteration,num_threads=config['threads'])
        selection=optimize_threshold(tune.y,scores,tune.groups,tune.truth)
        evaluation=report(tune.y,scores,tune.groups,tune.truth,tune.countries,selection['threshold'])
        item={'trial':number,'parameters':params,'best_iteration':model.best_iteration,'tune_report':evaluation,**selection}
        trials.append(item)
        temporary=destination/f'trial_{number}.txt.tmp'
        model.save_model(str(temporary),num_iteration=model.best_iteration)
        temporary.replace(trial_model)
        write_json(trial_record,item)
        if best is None or selection['macro_f05']>best['macro_f05']:
            best=item
            shutil.copyfile(trial_model,destination/'model.txt')
            write_json(destination/'tune_metrics.json',evaluation)
        logging.info('Trial %s complete in %.1f minutes: iteration=%s, macro F0.5=%.6f, threshold=%.8f',number+1,(time.monotonic()-began)/60,item['best_iteration'],item['macro_f05'],item['threshold'])
        del model,scores
    write_json(destination/'trials.json',trials)
    write_json(destination/'model_metadata.json',{'config':config,'features':FEATURE_NAMES,'selection':best,'prepared_manifest':manifest,'lightgbm_version':lgb.__version__})
    del train_set,valid_set,weights,counts
    training.close(); tune.close()
    del training,tune
    gc.collect()
    # The audit fold is opened only after all model/threshold selection is finished.
    logging.info('Model selection finished. Evaluating the untouched audit split.')
    audit=Prepared(prepared/'audit')
    if not len(audit.truth): raise ValueError('Audit set is empty; cannot report an independent score')
    winner=lgb.Booster(model_file=str(destination/'model.txt'))
    scores=winner.predict(audit.x,num_threads=config['threads']) if audit.size else np.empty(0)
    write_json(destination/'audit_metrics.json',report(audit.y,scores,audit.groups,audit.truth,audit.countries,best['threshold']))
    importance=dict(zip(FEATURE_NAMES,map(float,winner.feature_importance(importance_type='gain'))))
    write_json(destination/'feature_importance.json',dict(sorted(importance.items(),key=lambda x:-x[1])))
    audit.close()
    write_json(destination/'complete.json',{'complete':True,'trained':True})
    logging.info('TRAINING COMPLETE: model and audit reports saved to %s',destination.resolve())
