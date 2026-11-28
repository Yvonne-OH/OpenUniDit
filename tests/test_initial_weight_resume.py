"""A new backend inherits the source once; later restarts must prefer own weights."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import torch
from test_stage1_training_runtime import helper_namespace

class InitialResumeTests(unittest.TestCase):
    def test_own_checkpoint_wins_after_first_launch(self):
        ns=helper_namespace();loaded=[]
        engine=SimpleNamespace(module=SimpleNamespace(),load_full_weights=lambda state:(loaded.append(state['w'].item()) or (1,0,0)))
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'step-00040000.pt'
            torch.save({'w':torch.tensor(1.)},source)
            source.with_suffix('.json').write_text(json.dumps({'step':40000,'samples_seen':100,'data_cursor':{'batch':7}}))
            out=root/'new'
            self.assertEqual(ns['load_pure_weights'](engine,out,fallback_path=source),40000)
            self.assertEqual(engine.giga_samples_seen,100);self.assertEqual(engine.giga_data_cursor,{'batch':7})
            own=out/'checkpoints/weights/step-00040010.pt';own.parent.mkdir(parents=True)
            torch.save({'w':torch.tensor(2.)},own)
            own.with_suffix('.json').write_text(json.dumps({'step':40010,'samples_seen':120,'data_cursor':{'batch':27}}))
            self.assertEqual(ns['load_pure_weights'](engine,out,fallback_path=source),40010)
            self.assertEqual(loaded,[1.,2.]);self.assertEqual(engine.giga_samples_seen,120)
            self.assertEqual(engine.giga_data_cursor,{'batch':27})
    def test_absent_default_stays_fresh(self):
        ns=helper_namespace()
        with tempfile.TemporaryDirectory() as d:self.assertEqual(ns['load_pure_weights'](SimpleNamespace(),Path(d)),0)
if __name__=='__main__':unittest.main()
