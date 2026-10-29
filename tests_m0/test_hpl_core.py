"""CPU-only mechanism tests; temporary checkpoints, no data, GPU, or Test split."""
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import torch
from torch import nn
from scripts.vad_research.m0.hpl_core import (
    VADUncertainty, bilevel_head_step, component_gradient_norm,
    gradient_statistics, sample_statistics, save_last_checkpoint, weighted_pseudo_mse,
)


class HPLCoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)

    def test_shape_and_detached_inputs(self):
        weak = torch.randn(4, 3, requires_grad=True)
        strong = torch.randn(4, 3, requires_grad=True)
        uncertainty = VADUncertainty(hidden=8)
        logvar = uncertainty(strong - weak, strong)
        self.assertEqual(tuple(logvar.shape), (4, 3))
        loss = weighted_pseudo_mse(strong, weak, logvar)
        loss.backward()
        self.assertIsNone(weak.grad)
        self.assertIsNotNone(strong.grad)
        self.assertGreater(float(strong.grad.abs().sum()), 0)
        self.assertGreater(float(sum(p.grad.abs().sum() for p in uncertainty.parameters())), 0)

    def test_exact_weighting_and_zero_difference(self):
        strong = torch.tensor([[1., 2., 3.]], requires_grad=True)
        weak = torch.tensor([[0., 2., 1.]])
        s = torch.zeros_like(strong)
        loss = weighted_pseudo_mse(strong, weak, s)
        self.assertAlmostEqual(float(loss), 5.0 / 6.0, places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(strong.grad).all())
        exact = strong.detach().clone().requires_grad_(True)
        zero = weighted_pseudo_mse(exact, exact.detach(), s)
        zero.backward()
        self.assertEqual(float(zero), 0.0)
        self.assertEqual(float(exact.grad.abs().sum()), 0.0)

    def test_reject_bad_shapes(self):
        u = VADUncertainty(hidden=4)
        with self.assertRaises(ValueError):
            u(torch.ones(2, 1), torch.ones(2, 1))
        with self.assertRaises(ValueError):
            weighted_pseudo_mse(torch.ones(2, 3), torch.ones(2, 3), torch.ones(2, 1))

    def test_gradient_statistics_and_component_norms(self):
        first = (torch.tensor([3., 4.]), None)
        second = (torch.tensor([4., -3.]), torch.tensor([5.]))
        first_norm, second_norm, cosine = gradient_statistics(first, second)
        self.assertAlmostEqual(first_norm, 5.)
        self.assertAlmostEqual(second_norm, math.sqrt(50.))
        self.assertAlmostEqual(cosine, 0.)
        self.assertAlmostEqual(gradient_statistics((first[0],), (-first[0],))[2], -1.)
        self.assertEqual(gradient_statistics((None,), (None,)), (0., 0., 0.))
        with self.assertRaises(FloatingPointError):
            gradient_statistics((torch.tensor([float('nan')]),), (None,))
        parameter = nn.Parameter(torch.tensor(2.))
        mse = (parameter - 1.).square()
        regularization = -3. * parameter
        self.assertAlmostEqual(component_gradient_norm(mse, (parameter,)), 2.)
        self.assertAlmostEqual(component_gradient_norm(regularization, (parameter,)), 3.)
        self.assertIsNone(parameter.grad)
        (mse + regularization).backward()
        self.assertAlmostEqual(float(parameter.grad), -1.)

    def test_sample_statistics_are_finite(self):
        stats = sample_statistics(torch.tensor([1., 2., 3., 4., 5.]))
        self.assertAlmostEqual(stats['mean'], 3.)
        self.assertAlmostEqual(stats['std'], math.sqrt(2.))
        self.assertAlmostEqual(stats['p10'], 1.4, places=5)
        self.assertAlmostEqual(stats['p50'], 3.)
        self.assertAlmostEqual(stats['p90'], 4.6, places=5)
        self.assertEqual(sample_statistics(torch.tensor([2.]))['std'], 0.)
        self.assertTrue(all(math.isfinite(value) for value in stats.values()))

    def test_last_checkpoint_contains_final_model_and_uncertainty(self):
        model = nn.Linear(2, 3)
        uncertainty = VADUncertainty(hidden=4)
        before = model.weight.detach().clone()
        with torch.no_grad():
            model.weight.add_(1.)
            next(uncertainty.parameters()).add_(2.)
        with TemporaryDirectory() as directory:
            best = Path(directory) / 'model_best.pth'
            last = Path(directory) / 'model_last.pth'
            best.write_bytes(b'existing best checkpoint')
            save_last_checkpoint(model, uncertainty, 160, {'steps': 160}, last)
            payload = torch.load(last, map_location='cpu', weights_only=False)
            self.assertEqual(payload['step'], 160)
            self.assertEqual(payload['run_config'], {'steps': 160})
            self.assertFalse(torch.equal(payload['model']['weight'], before))
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(payload['model'][name], value))
            for name, value in uncertainty.state_dict().items():
                self.assertTrue(torch.equal(payload['uncertainty'][name], value))
            self.assertEqual(best.read_bytes(), b'existing best checkpoint')

    def test_bilevel_meta_gradient_and_head_unchanged(self):
        head = nn.Linear(5, 3)
        uncertainty = VADUncertainty(hidden=8)
        head_before = [p.detach().clone() for p in head.parameters()]
        unc_before = [p.detach().clone() for p in uncertainty.parameters()]
        head_opt = torch.optim.Adam(head.parameters(), lr=1e-3)
        unc_opt = torch.optim.Adam(uncertainty.parameters(), lr=1e-3)
        feats = torch.randn(4, 5)
        targets = torch.rand(4, 3)
        stats = bilevel_head_step(
            head, head_opt, uncertainty, unc_opt,
            feats, targets, torch.randn(5, 5), torch.randn(5, 5),
            torch.randn(4, 5), torch.rand(4, 3),
            lambda_u=0.2, lambda2=1.0,
        )
        self.assertTrue(all(torch.equal(p, q) for p, q in zip(head.parameters(), head_before)))
        self.assertTrue(any(not torch.equal(p, q) for p, q in zip(uncertainty.parameters(), unc_before)))
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in stats.values()))
        self.assertAlmostEqual(stats['meta_loss'],
                               stats['meta_mse'] + stats['meta_regularization_term'], places=6)
        self.assertGreaterEqual(stats['meta_mse_uncertainty_grad_norm'], 0.)
        self.assertGreaterEqual(stats['meta_regularization_uncertainty_grad_norm'], 0.)


    def test_unlabeled_term_drives_meta_gradient_without_regularizer(self):
        import copy
        head_a = nn.Linear(5, 3)
        unc_a = VADUncertainty(hidden=8)
        head_b = copy.deepcopy(head_a)
        unc_b = copy.deepcopy(unc_a)
        feats = torch.randn(4, 5)
        targets = torch.rand(4, 3)
        weak = torch.randn(5, 5)
        strong = torch.randn(5, 5)
        meta = torch.randn(4, 5)
        meta_y = torch.rand(4, 3)
        before_a = [p.detach().clone() for p in unc_a.parameters()]
        before_b = [p.detach().clone() for p in unc_b.parameters()]
        stats_a = bilevel_head_step(head_a, torch.optim.Adam(head_a.parameters(), lr=0.01),
                          unc_a, torch.optim.SGD(unc_a.parameters(), lr=0.01),
                          feats, targets, weak, strong, meta, meta_y,
                          lambda_u=0.2, lambda2=0.0)
        stats_b = bilevel_head_step(head_b, torch.optim.Adam(head_b.parameters(), lr=0.01),
                          unc_b, torch.optim.SGD(unc_b.parameters(), lr=0.01),
                          feats, targets, weak, strong, meta, meta_y,
                          lambda_u=0.0, lambda2=0.0)
        self.assertTrue(any(not torch.equal(p, q) for p, q in zip(unc_a.parameters(), before_a)))
        self.assertTrue(all(torch.equal(p, q) for p, q in zip(unc_b.parameters(), before_b)))
        self.assertGreater(stats_a['meta_mse_uncertainty_grad_norm'], 0.)
        self.assertEqual(stats_b['meta_mse_uncertainty_grad_norm'], 0.)
        self.assertEqual(stats_a['meta_regularization_uncertainty_grad_norm'], 0.)
        self.assertEqual(stats_b['meta_regularization_uncertainty_grad_norm'], 0.)


if __name__ == '__main__':
    unittest.main()