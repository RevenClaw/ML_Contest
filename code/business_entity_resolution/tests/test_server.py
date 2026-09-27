import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ber.common import write_json
from ber.server import find_data, preserve_incomplete, run_stage, sha256, verify_model


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)

    def tearDown(self): self.tmp.cleanup()

    def test_dataset_path_discovery(self):
        dataset=self.root/'student_resource/dataset/train'
        dataset.mkdir(parents=True)
        (dataset/'train_source1.tsv').write_text('fixture')
        self.assertEqual(find_data(self.root),dataset.parent)
        self.assertEqual(find_data(self.root,str(dataset.parent.parent)),dataset.parent)
        with self.assertRaises(ValueError): find_data(self.root,str(self.root/'missing'))

    def test_preserve_partial_database_and_sidecars(self):
        path=self.root/'train.sqlite'
        for suffix in ['', '-wal','-shm']: Path(str(path)+suffix).write_text('preserved')
        preserve_incomplete(path)
        self.assertFalse(path.exists())
        archived=list(self.root.glob('train.sqlite.interrupted-*'))
        self.assertEqual(len(archived),3)
        self.assertTrue(all(p.read_text()=='preserved' for p in archived))

    def test_failed_stage_cannot_report_completion(self):
        with patch('ber.server.subprocess.Popen') as popen:
            popen.return_value.wait.return_value=7
            with self.assertRaisesRegex(RuntimeError,'exit code 7'):
                run_stage('TEST',['validate'],self.root)

    def test_missing_model_artifact_rejected(self):
        model=self.root/'model'; model.mkdir()
        write_json(model/'complete.json',{'complete':True,'trained':True})
        with self.assertRaisesRegex(ValueError,'missing or empty'): verify_model(model,{})

    def test_atomic_metadata_and_hash(self):
        path=self.root/'state.json'; write_json(path,{'a':1})
        before=sha256(path); write_json(path,{'a':2})
        self.assertNotEqual(before,sha256(path))
        self.assertEqual(json.loads(path.read_text()),{'a':2})
        self.assertFalse(path.with_name('state.json.tmp').exists())


if __name__=='__main__': unittest.main()
