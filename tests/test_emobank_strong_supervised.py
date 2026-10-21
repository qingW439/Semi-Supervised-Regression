"""CPU tests: python -m unittest discover -s tests -p test_emobank_strong_supervised.py."""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.emobank import train_emobank_strong_supervised as strong  # noqa: E402


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = nn.Module()
        self.bert.encoder = nn.Module()
        self.bert.encoder.layer = nn.ModuleList([nn.Linear(4, 4) for _ in range(4)])
        self.head = nn.Linear(4, 3)

    def forward(self, x):
        for layer in self.bert.encoder.layer:
            x = layer(x).tanh()
        return self.head(x)


class StrongSupervisedTests(unittest.TestCase):
    def test_ridge_fold_and_prediction_equivalence(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(50, 512)).astype(np.float32)
        y = rng.normal(size=(50, 3)).astype(np.float32)
        dev = rng.normal(size=(13, 512)).astype(np.float32)
        probe, weight, bias = strong.fit_fixed_ridge({"x_train": x, "y_train": y})
        expected = make_pipeline(StandardScaler(), Ridge(alpha=1000.0)).fit(x, y).predict(dev)
        np.testing.assert_allclose(probe.predict(dev), expected, atol=1e-7)
        np.testing.assert_allclose(dev.astype(np.float64) @ weight.T + bias, expected, atol=2e-6)
        module = nn.Module()
        module.head = nn.Linear(512, 3)
        strong.install_linear_head(module, weight, bias)
        module.eval()
        with torch.no_grad():
            actual = module.head(torch.from_numpy(dev)).numpy()
        self.assertLess(strong.assert_prediction_equivalence(
            expected, actual, name="unit folded head", atol=5e-6), 5e-6)

    def test_frozen_layer_gradients(self):
        for stage, layers in (("S1", [3]), ("S2", [0, 1, 2, 3])):
            with self.subTest(stage=stage):
                torch.manual_seed(0)
                model = TinyModel()
                enc, head = strong.freeze_for_stage(model, stage)
                self.assertEqual(len(enc), 2 * len(layers))
                self.assertEqual(len(head), 2)
                model(torch.randn(5, 4)).abs().mean().backward()
                strong.assert_trainable_gradients(model, enc, head)
                for index, layer in enumerate(model.bert.encoder.layer):
                    for param in layer.parameters():
                        self.assertEqual(param.requires_grad, index in layers)
                        self.assertEqual(param.grad is not None, index in layers)

    def test_common_initialization(self):
        torch.manual_seed(4)
        reference = TinyModel().state_dict()
        a, b = TinyModel(), TinyModel()
        a.load_state_dict(copy.deepcopy(reference), strict=True)
        b.load_state_dict(copy.deepcopy(reference), strict=True)
        self.assertEqual(strong.state_fingerprint(a), strong.state_fingerprint(b))
        strong.freeze_for_stage(a, "S1")
        strong.freeze_for_stage(b, "S2")
        self.assertTrue(all(torch.equal(a.state_dict()[name], b.state_dict()[name]) for name in reference))

    def test_dev_isolation_and_epoch_zero(self):
        strong.assert_disjoint_ids(["tr-a", "tr-b"], ["dv-c"])
        with self.assertRaisesRegex(ValueError, "leakage"):
            strong.assert_disjoint_ids(["tr-a", "tr-b"], ["dv-c", "tr-b"])
        report = {"macro_mae_original_scale": 0.18,
                  **{dim: {metric: 0.1 for metric in (
                      "mae", "rmse", "pearson", "ccc", "pred_std", "target_std")}
                     for dim in strong.DIMENSIONS}}
        self.assertEqual(strong.history_row(0, None, report)["epoch"], 0)
        self.assertFalse(strong.improves(0.19, 0.18))
        self.assertFalse(strong.improves(0.18, 0.18))
        self.assertTrue(strong.improves(0.17, 0.18))
        self.assertEqual(strong.best_history_epoch([
            {"epoch": 0, "dev_macro_mae_original_scale": 0.18},
            {"epoch": 1, "dev_macro_mae_original_scale": 0.19}]), 0)

    def test_fixed_configs(self):
        paths = [REPO / "config/emobank/strong_supervised_s1_lb500_s0.json",
                 REPO / "config/emobank/strong_supervised_s2_lb500_s0.json"]
        configs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        outputs = [strong.validate_config(config) for config in configs]
        self.assertEqual(len(set(outputs)), 2)
        configs[0]["encoder_lr"] = 1e-4
        with self.assertRaisesRegex(ValueError, "Protocol violation"):
            strong.validate_config(configs[0])

    def test_ridge_package_alpha_and_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_ids = [f"tr{i}" for i in range(500)]
            dev_ids = [f"dv{i}" for i in range(1000)]
            train = pd.DataFrame({"id": train_ids, **{name: np.full(500, 0.5)
                for name in strong.NORM_COLUMNS}})
            dev = pd.DataFrame({"id": dev_ids, **{name: np.full(1000, 0.5)
                for name in strong.NORM_COLUMNS}})
            np.savez_compressed(root / "base_pretrained_features.npz",
                x_train=np.zeros((500, 512), dtype=np.float32),
                y_train=np.full((500, 3), 0.5, dtype=np.float32),
                x_dev=np.zeros((1000, 512), dtype=np.float32),
                y_dev=np.full((1000, 3), 0.5, dtype=np.float32),
                train_ids=np.asarray(train_ids), dev_ids=np.asarray(dev_ids))
            audit = {"selected_alpha": 1000.0,
                     "dev_metrics": {"macro_mae_original_scale": strong.EXPECTED_S0_MAE}}
            (root / "ridge_base_audit.json").write_text(json.dumps(audit))
            pd.DataFrame({"id": dev_ids, **{f"pred_{dim}": np.full(1000, 3.0)
                for dim in strong.DIMENSIONS}}).to_csv(root / "dev_predictions.csv", index=False)
            _, _, historical = strong.load_ridge_package(root, train, dev)
            self.assertEqual(historical.shape, (1000, 3))
            bad_dev = dev.copy()
            bad_dev.loc[0, "id"] = "wrong"
            with self.assertRaisesRegex(ValueError, "IDs"):
                strong.load_ridge_package(root, train, bad_dev)
            audit["selected_alpha"] = 100.0
            (root / "ridge_base_audit.json").write_text(json.dumps(audit))
            with self.assertRaisesRegex(ValueError, "alpha"):
                strong.load_ridge_package(root, train, dev)


if __name__ == "__main__":
    unittest.main()
