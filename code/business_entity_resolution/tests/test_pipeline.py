import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from ber.common import SOURCE_COLUMNS, load_config, split_for, write_json
from ber.features import FEATURE_NAMES, pair_features
from ber.metrics import best_grid_f05, entity_f05, macro_f05, optimize_threshold, report
from ber.prepare import Prepared, prepare
from ber.retrieval import Retriever
from ber.store import build_index, connect
from ber.text import forms, normalize
from ber.validate import validate

CONFIG=Path(__file__).resolve().parents[1]/'configs/default.json'


def tsv(path,header,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f,delimiter='\t',lineterminator='\n'); w.writerow(header); w.writerows(rows)


class MetricTests(unittest.TestCase):
    def test_readme_example(self):
        self.assertAlmostEqual(entity_f05(2,3,2),5/7)

    def test_singletons_and_missing_candidates(self):
        truth=np.array([0,2,1]); groups=np.array([0,1]); y=np.array([0,1]); p=np.array([.8,.9])
        self.assertAlmostEqual(macro_f05(y,p,groups,truth,.85),(1+1.25/1.5)/3)
        self.assertEqual(entity_f05(0,1,0),0)

    def test_macro_not_micro(self):
        self.assertAlmostEqual(macro_f05(np.ones(10),np.ones(10),np.zeros(10,dtype=int),np.array([10,1]),.5),.5)

    def test_no_candidates(self):
        self.assertEqual(macro_f05(np.array([]),np.array([]),np.array([],dtype=int),np.array([0,1]),.5),.5)

    def test_exact_sweep_matches_bruteforce(self):
        rng=np.random.default_rng(42)
        for _ in range(50):
            group=rng.integers(0,12,100); labels=rng.integers(0,2,100)
            truth=np.bincount(group,weights=labels,minlength=12).astype(int)+rng.integers(0,3,12)
            scores=np.round(rng.random(100),1)
            result=optimize_threshold(labels,scores,group,truth)
            best=max(macro_f05(labels,scores,group,truth,t) for t in [np.nextafter(float(scores.max()),np.inf),*set(scores)])
            self.assertAlmostEqual(result['macro_f05'],best)
            self.assertAlmostEqual(macro_f05(labels,scores,group,truth,result['threshold']),best)

    def test_reject_all_threshold_and_ties(self):
        result=optimize_threshold(np.array([0,0]),np.array([1.,1.]),np.array([0,0]),np.array([0]))
        self.assertGreater(result['threshold'],1.)
        self.assertEqual(result['macro_f05'],1.)

    def test_blocking_ceiling(self):
        r=report(np.array([1]),np.array([.9]),np.array([0]),np.array([2,0]),['India','France'],.5)
        self.assertEqual(r['blocking_pair_recall'],.5)
        self.assertAlmostEqual(r['blocking_macro_f05_ceiling'],(1+1.25/1.5)/2)

    def test_vectorized_grid_matches_independent_scorer(self):
        rng=np.random.default_rng(10)
        for _ in range(30):
            groups=rng.integers(0,20,100); y=rng.integers(0,2,100)
            truth=np.bincount(groups,weights=y,minlength=22).astype(int)+rng.integers(0,2,22)
            scores=np.round(rng.random(100),1)
            thresholds=[0,.2,.5,.8,1]
            self.assertAlmostEqual(best_grid_f05(y,scores,groups,truth,thresholds),
                max(macro_f05(y,scores,groups,truth,t) for t in thresholds))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        self.config=load_config(CONFIG)
        self.ids={}
        for i in range(1000):
            sid=f'S1-{i:05d}'; fold=split_for(sid,self.config)
            self.ids.setdefault(fold,sid)
            if len(self.ids)==3: break
        self.refrows=[[self.ids[f],'Alpha Services Ltd','10 Lake Road, Pune','India'] for f in ['train','tune','audit']]
        self.targetrows=[['S2-1','Alpha Services Limited','10 Lake Rd Pune','India'],['S2-2','Alpha Services Ltd','10 Lake Road Pune','India'],['S2-3','Alpha Services Ltd','10 Lake Rd Pune','India'],['S2-4','Alpha Services Ltd','999 River Road Delhi','India']]
        for split in ['train','test']:
            tsv(self.root/split/f'{split}_source1.tsv',SOURCE_COLUMNS,self.refrows)
            tsv(self.root/split/f'{split}_source2.tsv',SOURCE_COLUMNS,self.targetrows)
            tsv(self.root/split/f'{split}_source3.tsv',SOURCE_COLUMNS,[['S3-1','Autre Entreprise','8 rue Victor Hugo','France']])
        tsv(self.root/'train/train_ground_truth.tsv',['source1_entity_id','matched_entity_ids'],[[self.ids[f],f'S2-{i}'] for i,f in enumerate(['train','tune','audit'],1)])
        self.index=self.root/'train.sqlite'
        build_index(self.root,'train',self.index,self.config)

    def tearDown(self):
        self.tmp.cleanup()

    def test_holdout_targets_excluded_from_training(self):
        db=connect(self.index)
        try:
            retriever=Retriever(db,self.config['retrieval'])
            ref=dict(db.execute("SELECT * FROM refs WHERE fold='train'").fetchone())
            candidates=retriever.get(ref,training=True)
            ids={t['id'] for t,_ in candidates}
            self.assertIn('S2-1',ids); self.assertIn('S2-4',ids)
            self.assertNotIn('S2-2',ids); self.assertNotIn('S2-3',ids)
            self.assertTrue({'S2-1','S2-2','S2-3'}<={t['id'] for t,_ in retriever.get(ref)})
        finally: db.close()

    def test_features_finite_missing_not_agreement(self):
        db=connect(self.index)
        try:
            retriever=Retriever(db,self.config['retrieval']); ref=dict(db.execute('SELECT * FROM refs LIMIT 1').fetchone())
            target,route=retriever.get(ref)[0]
            ref['address']=''; target['address']=''
            x=pair_features(ref,target,route,1,retriever)
            self.assertEqual(x.shape,(len(FEATURE_NAMES),)); self.assertTrue(np.isfinite(x).all())
            self.assertEqual(x[FEATURE_NAMES.index('address_native_exact')],0)
            self.assertEqual(x[FEATURE_NAMES.index('address_missing_right')],1)
        finally: db.close()

    def test_prepare_no_training_and_manifest(self):
        out=self.root/'prepared'; prepare(self.index,out,self.config)
        for fold in ['train','tune','audit']:
            d=Prepared(out/fold)
            self.assertEqual(len(d.truth),1); self.assertEqual(int(d.y.sum()),1)
            self.assertTrue(np.isfinite(d.x).all())
            # Close memory maps so Windows can remove temporary fixture files.
            for a in (d.x,d.y,d.groups):
                if isinstance(a,np.memmap): a._mmap.close()
        self.assertFalse(list(self.root.rglob('model.txt')))

    def test_unseen_country_and_strict_output(self):
        test=self.root/'test.sqlite'
        extra=['S1-france','Autre Entreprise','8 rue Victor Hugo','France']
        tsv(self.root/'test/test_source1.tsv',SOURCE_COLUMNS,self.refrows+[extra])
        build_index(self.root,'test',test,self.config)
        db=connect(test)
        try:
            ref=dict(db.execute("SELECT * FROM refs WHERE country='France'").fetchone())
            self.assertIn('S3-1',{t['id'] for t,_ in Retriever(db,self.config['retrieval']).get(ref)})
        finally: db.close()
        out=self.root/'output'; refs=sorted([r[0] for r in self.refrows]+['S1-france'])
        rows=[[sid,'S3-1' if sid=='S1-france' else ''] for sid in refs]
        tsv(out/'matching_results.tsv',['source1_entity_id','matched_entity_ids'],rows)
        tsv(out/'candidate_pairs.tsv',['source1_entity_id','candidate_entity_ids'],rows)
        self.assertEqual(validate(test,out)['references'],4)
        rows[-1][1]='S3-nonexistent'
        tsv(out/'candidate_pairs.tsv',['source1_entity_id','candidate_entity_ids'],rows)
        with self.assertRaises(ValueError): validate(test,out)

    def test_predict_with_stub_no_model_training(self):
        from ber.predict import predict
        import types
        test=self.root/'test.sqlite'; build_index(self.root,'test',test,self.config)
        model=self.root/'fake_model'; model.mkdir()
        write_json(model/'complete.json',{'complete':True})
        write_json(model/'model_metadata.json',{'features':FEATURE_NAMES,'config':self.config,'selection':{'threshold':.9}})
        class FakeBooster:
            def __init__(self,**kwargs): pass
            def predict(self,x,**kwargs): return np.ones(len(x))*.95
        with patch.dict('sys.modules',{'lightgbm':types.SimpleNamespace(Booster=FakeBooster)}):
            predict(test,model,self.root/'predictions')
        self.assertEqual(validate(test,self.root/'predictions')['references'],3)

    def test_training_selection_with_stub_only(self):
        """Verify the training orchestrator's API flow without fitting any model."""
        from ber.train import train
        import types
        out=self.root/'prepared'; prepare(self.index,out,self.config)
        opened=[]; datasets=[]; evaluations=[]
        class FakeDataset:
            def __init__(self,x,**kwargs):
                self.x=x; self.kwargs=kwargs; datasets.append(self)
        class FakeBooster:
            best_iteration=4
            def __init__(self,**kwargs): pass
            def predict(self,x,**kwargs): return np.full(len(x),.75)
            def save_model(self,path,**kwargs): Path(path).write_text('STUB ONLY',encoding='utf-8')
            def feature_importance(self,**kwargs): return np.ones(len(FEATURE_NAMES))
        def fake_train(params,data,**kwargs):
            self.assertEqual(params['objective'],'binary')
            self.assertEqual(params['metric'],'None')
            metric=kwargs['feval'](np.full(len(kwargs['valid_sets'][0].x),.75),kwargs['valid_sets'][0])
            self.assertTrue(metric[2]); evaluations.append(metric)
            return FakeBooster()
        def tracked_prepared(path):
            opened.append(Path(path).name)
            return Prepared(path)
        fake=types.SimpleNamespace(Dataset=FakeDataset,Booster=FakeBooster,train=fake_train,
            early_stopping=lambda *a,**k:None,log_evaluation=lambda *a:None,__version__='stub')
        with patch.dict('sys.modules',{'lightgbm':fake}),patch('ber.train.Prepared',side_effect=tracked_prepared):
            train(out,self.root/'selected',self.config)
        self.assertEqual(opened,['train','tune','audit'])
        self.assertEqual(len(evaluations),len(self.config['trials']))
        self.assertTrue((self.root/'selected/audit_metrics.json').exists())
        # Simulate interruption after all trials, before the completion marker.
        (self.root/'selected/complete.json').rename(self.root/'selected/previous-complete.json')
        with patch.dict('sys.modules',{'lightgbm':fake}):
            train(out,self.root/'selected',self.config)
        self.assertEqual(len(evaluations),len(self.config['trials']))
        # Release maps held by fake datasets under Windows.
        for dataset in datasets:
            for obj in [dataset.x,dataset.kwargs.get('label')]:
                if isinstance(obj,np.memmap): obj._mmap.close()

    def test_prepare_resumes_without_duplicate_or_partial_rows(self):
        from ber.features import pair_features as original
        more=[]
        for i in range(1000,1100):
            sid=f'S1-{i:05d}'
            if split_for(sid,self.config)=='train':
                more.append([sid,'Alpha Services Ltd','10 Lake Road Pune','India'])
            if len(more)==2: break
        tsv(self.root/'train/train_source1.tsv',SOURCE_COLUMNS,self.refrows+more)
        tsv(self.root/'train/train_ground_truth.tsv',['source1_entity_id','matched_entity_ids'],
            [[self.ids[f],f'S2-{i}'] for i,f in enumerate(['train','tune','audit'],1)]+[[r[0],''] for r in more])
        idx=self.root/'resume.sqlite'; build_index(self.root,'train',idx,self.config)
        target=max([self.ids['train']]+[r[0] for r in more])
        def fail_on_third(ref,*args):
            if ref['id']==target: raise RuntimeError('simulated interruption')
            return original(ref,*args)
        resumed=self.root/'resumed'; fresh=self.root/'fresh'
        with patch('ber.prepare.CHECKPOINT_EVERY',1),patch('ber.prepare.pair_features',side_effect=fail_on_third):
            with self.assertRaisesRegex(RuntimeError,'simulated'): prepare(idx,resumed,self.config)
        checkpoint=json.loads((resumed/'train/checkpoint.json').read_text())
        self.assertEqual(checkpoint['references'],2)
        prepare(idx,resumed,self.config)
        prepare(idx,fresh,self.config)
        prepare(idx,resumed,self.config)  # Completed data is reusable, not overwritten.
        for fold in ['train','tune','audit']:
            for name in ['features.f32','labels.u8','groups.i32','references.tsv','pairs.tsv']:
                self.assertEqual((resumed/fold/name).read_bytes(),(fresh/fold/name).read_bytes())

    def test_resume_rejects_changed_configuration(self):
        out=self.root/'prepared'; prepare(self.index,out,self.config)
        other=copy.deepcopy(self.config); other['retrieval']['name_k']+=1
        with self.assertRaises(ValueError): prepare(self.index,out,other)

    def test_truncated_array_rejected(self):
        out=self.root/'prepared'; prepare(self.index,out,self.config)
        with (out/'train/features.f32').open('r+b') as handle: handle.truncate(4)
        with self.assertRaises(ValueError): Prepared(out/'train')

    def test_zero_candidate_reference_is_preserved(self):
        # A distinct-country reference has a ground-truth singleton and no targets.
        sid=next(f'S1-empty-{i}' for i in range(100) if split_for(f'S1-empty-{i}',self.config)=='audit')
        tsv(self.root/'train/train_source1.tsv',SOURCE_COLUMNS,self.refrows+[[sid,'Isolated','Remote','NewCountry']])
        tsv(self.root/'train/train_ground_truth.tsv',['source1_entity_id','matched_entity_ids'],
            [[self.ids[f],f'S2-{i}'] for i,f in enumerate(['train','tune','audit'],1)]+[[sid,'']])
        idx=self.root/'singletons.sqlite'; build_index(self.root,'train',idx,self.config)
        dest=self.root/'with_singletons'; prepare(idx,dest,self.config)
        d=Prepared(dest/'audit')
        self.assertIn(sid,d.ids)
        group=d.ids.index(sid)
        self.assertEqual(d.truth[group],0)
        self.assertFalse(np.any(d.groups==group))
        for a in (d.x,d.y,d.groups):
            if isinstance(a,np.memmap): a._mmap.close()

    def test_shared_target_rejected(self):
        tsv(self.root/'train/train_ground_truth.tsv',['source1_entity_id','matched_entity_ids'],
            [[self.ids[f],'S2-1'] for f in ['train','tune','audit']])
        import sqlite3
        with self.assertRaises(sqlite3.IntegrityError):
            build_index(self.root,'train',self.root/'conflict.sqlite',self.config)

    def test_no_overwrite(self):
        with self.assertRaises(FileExistsError): build_index(self.root,'train',self.index,self.config)

    def test_unicode_marks_preserved(self):
        self.assertEqual(normalize('बाबा'), 'बाबा')
        self.assertEqual(normalize('ABC & Co.'),'abc and co')


if __name__=='__main__': unittest.main()
