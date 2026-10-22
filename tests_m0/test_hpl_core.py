"""CPU-only mechanism tests; no checkpoint, data, GPU, or Test split."""
import unittest
import torch
from torch import nn
from scripts.vad_research.m0.hpl_core import VADUncertainty, weighted_pseudo_mse, bilevel_head_step


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
        bilevel_head_step(head_a, torch.optim.Adam(head_a.parameters(), lr=0.01),
                          unc_a, torch.optim.SGD(unc_a.parameters(), lr=0.01),
                          feats, targets, weak, strong, meta, meta_y,
                          lambda_u=0.2, lambda2=0.0)
        bilevel_head_step(head_b, torch.optim.Adam(head_b.parameters(), lr=0.01),
                          unc_b, torch.optim.SGD(unc_b.parameters(), lr=0.01),
                          feats, targets, weak, strong, meta, meta_y,
                          lambda_u=0.0, lambda2=0.0)
        self.assertTrue(any(not torch.equal(p, q) for p, q in zip(unc_a.parameters(), before_a)))
        self.assertTrue(all(torch.equal(p, q) for p, q in zip(unc_b.parameters(), before_b)))


if __name__ == '__main__':
    unittest.main()
