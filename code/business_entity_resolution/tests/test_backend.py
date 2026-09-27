import copy
import json
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

from ber.backend import configure_backend, model_parameters
from ber.common import load_config
from ber.cuda import build_is_verified, driver_info, preflight, telemetry

CONFIG=Path(__file__).resolve().parents[1]/'configs/default.json'


class BackendTests(unittest.TestCase):
    def setUp(self): self.config=load_config(CONFIG)

    def test_cpu_settings_retained(self):
        p=model_parameters(self.config,self.config['trials'][0])
        self.assertEqual(p['device_type'],'cpu')
        self.assertEqual(p['max_bin'],127)
        self.assertTrue(p['deterministic'])
        self.assertTrue(p['force_col_wise'])
        self.assertNotIn('num_gpu',p)

    def test_cuda_settings_never_include_cpu_or_opencl_flags(self):
        c=configure_backend(self.config,'cuda',2)
        for trial in c['trials']:
            p=model_parameters(c,trial)
            self.assertEqual(p['device_type'],'cuda')
            self.assertEqual(p['gpu_device_id'],2)
            self.assertEqual(p['num_gpu'],1)
            self.assertEqual(p['max_bin'],63)
            self.assertEqual(p['objective'],'binary')
            for key in ['force_col_wise','force_row_wise','deterministic','gpu_use_dp','gpu_platform_id']:
                self.assertNotIn(key,p)

    def test_explicit_bin_override(self):
        c=configure_backend(self.config,'cuda',0,127)
        self.assertEqual(model_parameters(c,{})['max_bin'],127)

    def test_trial_cannot_silently_change_device_or_objective(self):
        configure_backend(self.config,'cuda')
        for override in [{'device':'cpu'},{'device_type':'cpu'},{'objective':'regression'},{'max_bin':255}]:
            with self.assertRaises(ValueError): model_parameters(self.config,override)

    def test_invalid_device_rejected(self):
        with self.assertRaises(ValueError): configure_backend(self.config,'gpu')
        with self.assertRaises(ValueError): configure_backend(self.config,'cuda',-1)
        with self.assertRaises(ValueError): configure_backend(self.config,'cuda',0,1)

    def test_gpu_preflight_failure_never_retries_on_cpu(self):
        calls=[]
        def failed_train(params,*args,**kwargs):
            calls.append(params['device_type'])
            raise RuntimeError('CUDA Tree Learner was not enabled in this build')
        fake=types.SimpleNamespace(train=failed_train,Dataset=lambda *a,**k:None)
        info={'name':'test A30','compute_capability':'8.0','total_memory_bytes':24*2**30}
        with patch('ber.cuda.driver_info',return_value=info),patch.dict('sys.modules',{'lightgbm':fake}):
            with self.assertRaisesRegex(RuntimeError,'CPU fallback is disabled'): preflight()
        self.assertEqual(calls,['cuda'])

    def test_driver_errors_are_actionable(self):
        with patch('ber.cuda.platform.system',return_value='Linux'),patch('ber.cuda.ctypes.CDLL',side_effect=OSError('missing')):
            with self.assertRaisesRegex(RuntimeError,'driver library is unavailable'): driver_info(0)

    def test_telemetry_failure_does_not_stop_training(self):
        with patch('ber.cuda.subprocess.check_output',side_effect=subprocess.TimeoutExpired('nvidia-smi',5)):
            self.assertEqual(telemetry(),'GPU telemetry unavailable')

    def test_missing_build_marker_is_not_verified(self):
        self.assertFalse(build_is_verified(Path(__file__).parent/'does-not-exist.json'))

    def test_modified_cuda_binary_invalidates_marker(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            binary=Path(directory)/'lib_lightgbm.so'; binary.write_bytes(b'CUDA fixture')
            marker=Path(directory)/'verified.json'
            marker.write_text(json.dumps({'status':'passed','lightgbm_version':'4.6.0',
                'lightgbm_binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest()}))
            fake=types.SimpleNamespace(__version__='4.6.0')
            basic=types.SimpleNamespace(_LIB=types.SimpleNamespace(_name=str(binary)))
            with patch.dict('sys.modules',{'lightgbm':fake,'lightgbm.basic':basic}):
                self.assertTrue(build_is_verified(marker))
                binary.write_bytes(b'CPU fixture')
                self.assertFalse(build_is_verified(marker))


if __name__=='__main__': unittest.main()
