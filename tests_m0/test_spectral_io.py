"""Train/Dev identity isolation in the frozen-feature input contract."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scripts.vad_research.m0.run_spectral_ridge import checked_features


class SpectralIOTests(unittest.TestCase):
    def test_feature_ids_and_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            arrays={
                'lb_x':np.ones((500,2),dtype=np.float32),
                'ulb_x':np.ones((7562,2),dtype=np.float32),
                'dev_x':np.ones((1000,2),dtype=np.float32),
                'lb_y':np.zeros((500,3),dtype=np.float32),
                'dev_y':np.zeros((1000,3),dtype=np.float32),
                'lb_ids':np.array([f'lb_{i}' for i in range(500)]),
                'ulb_ids':np.array([f'ulb_{i}' for i in range(7562)]),
                'dev_ids':np.array([f'dev_{i}' for i in range(1000)]),
            }
            (folder/'feature_metadata.json').write_text(json.dumps({
                'budget':500,'seed':0,'model':'frozen pretrained RoBERTa-base, mask-aware mean pooling'}))
            np.savez_compressed(folder/'pretrained_features.npz',**arrays)
            checked,_=checked_features(folder,500,0)
            self.assertEqual(checked['dev_x'].shape,(1000,2))
            arrays['dev_ids'][0]=arrays['lb_ids'][0]
            np.savez_compressed(folder/'pretrained_features.npz',**arrays)
            with self.assertRaisesRegex(ValueError,'overlap'):
                checked_features(folder,500,0)


if __name__=='__main__':
    unittest.main()
