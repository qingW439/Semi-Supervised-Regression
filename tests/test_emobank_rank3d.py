"""CPU tests for the three-dimensional auxiliary ranking objective."""

import unittest
from copy import deepcopy
from types import SimpleNamespace

import torch
from torch import nn

from semilearn.nets.bert.bert_vad_rank import (
    BertVADRank3D, pseudo_rank_loss, supervised_rank_loss,
)
from scripts.emobank.train_emobank_rank3d import train_step


class TinyBackbone(nn.Module):
    num_features = 4

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(4, 3)

    def encode(self, **tokens):
        return tokens["features"]

    def regress(self, features):
        return self.head(features)


class CountingDropoutBackbone(TinyBackbone):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.3)
        self.forward_calls = 0

    def encode(self, **tokens):
        self.forward_calls += 1
        return self.dropout(tokens["features"])


class Rank3DTests(unittest.TestCase):
    def test_shapes_swap_and_backward(self):
        model = BertVADRank3D(TinyBackbone())
        outputs = model(features=torch.randn(4, 4))
        self.assertEqual(outputs["regression"].shape, (4, 3))
        self.assertEqual(outputs["rank_logits"].shape, (4, 3, 2))
        p = model.pair_probability(outputs["rank_logits"])
        self.assertEqual(p.shape, (4, 4, 3))
        torch.testing.assert_close(p + p.transpose(0, 1), torch.ones_like(p), atol=1e-6, rtol=0)
        labels = torch.tensor([[0.8, 0.3, 0.5], [0.1, 0.7, 0.5],
                               [0.4, 0.6, 0.2], [0.3, 0.4, 0.8]])
        rank_loss, tie_counts = supervised_rank_loss(outputs["rank_logits"], labels)
        self.assertEqual(tie_counts.tolist(), [0, 0, 1])
        (outputs["regression"].abs().mean() + rank_loss).backward()
        self.assertTrue(torch.isfinite(model.rank_head.weight.grad).all())
        self.assertGreater(model.rank_head.weight.grad.abs().sum().item(), 0)

    def test_gold_direction_and_tie_target(self):
        # class 0 denotes first sample's label is larger; V is increasing,
        # A is decreasing, D ties and must target probability 0.5.
        scores = torch.tensor([[[4., 0.], [0., 4.], [0., 0.]],
                               [[0., 0.], [4., 0.], [0., 0.]]])
        labels = torch.tensor([[0.9, 0.1, 0.5], [0.1, 0.9, 0.5]])
        good, ties = supervised_rank_loss(scores, labels)
        bad, _ = supervised_rank_loss(-scores, labels)
        self.assertLess(good.item(), bad.item())
        self.assertEqual(ties.tolist(), [0, 0, 1])

    def test_unique_pseudopairs_and_empty_gradient(self):
        teacher = torch.zeros(3, 3, 2)
        teacher[:, :, 0] = torch.tensor([8., 0., -8.])[:, None]
        student = torch.randn(3, 3, 2, requires_grad=True)
        loss, accepted, possible = pseudo_rank_loss(teacher, student, 0.95)
        self.assertEqual(possible.tolist(), [3, 3, 3])
        self.assertEqual(accepted.tolist(), [3, 3, 3])
        loss.backward()
        self.assertTrue(torch.isfinite(student.grad).all())

        empty_student = torch.randn(3, 3, 2, requires_grad=True)
        no_pairs, accepted, possible = pseudo_rank_loss(
            torch.zeros_like(teacher), empty_student, 0.95)
        self.assertEqual(accepted.tolist(), [0, 0, 0])
        self.assertEqual(possible.tolist(), [3, 3, 3])
        self.assertTrue(torch.isfinite(no_pairs))
        no_pairs.backward()
        self.assertTrue(torch.equal(empty_student.grad, torch.zeros_like(empty_student)))

    def test_singleton_batch_has_no_rank_gradient(self):
        rank = torch.randn(1, 3, 2, requires_grad=True)
        loss, ties = supervised_rank_loss(rank, torch.tensor([[0.2, 0.3, 0.4]]))
        self.assertEqual(ties.tolist(), [0, 0, 0])
        loss.backward()
        self.assertTrue(torch.equal(rank.grad, torch.zeros_like(rank)))

    def test_three_controls_use_expected_signals_and_same_labeled_rng(self):
        base = BertVADRank3D(CountingDropoutBackbone())
        labeled = ({"features": torch.randn(4, 4)}, torch.rand(4, 3), list(range(4)))
        unlabeled = ({"features": torch.randn(4, 4)}, list(range(4)))
        rng_states = {}
        for mode, expected_calls in (("supervised_control", 1),
                                     ("supervised_arc", 1), ("rank3d", 3)):
            model = deepcopy(base)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            args = SimpleNamespace(mode=mode, arc_weight=0.2,
                                   unlabeled_rank_weight=0 if mode == "supervised_arc" else 1,
                                   confidence_cutoff=0.95, clip_grad_norm=1.0)
            torch.manual_seed(2026)
            report = train_step(model, optimizer, labeled,
                                unlabeled if mode == "rank3d" else None,
                                args, torch.device("cpu"))
            rng_states[mode] = torch.get_rng_state().clone()
            self.assertEqual(model.backbone.forward_calls, expected_calls)
            self.assertTrue(torch.isfinite(torch.tensor(report["loss"])))
            if mode == "supervised_control":
                self.assertEqual(report["supervised_rank"], 0)
            else:
                self.assertGreater(report["supervised_rank"], 0)
            if mode != "rank3d":
                self.assertEqual(report["unsupervised_rank"], 0)
                self.assertEqual(report["possible"], [0, 0, 0])
        self.assertTrue(torch.equal(rng_states["supervised_control"],
                                    rng_states["supervised_arc"]))
        self.assertTrue(torch.equal(rng_states["supervised_arc"], rng_states["rank3d"]))


if __name__ == "__main__":
    unittest.main()
