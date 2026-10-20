"""CPU tests for the three-dimensional auxiliary ranking objective."""

import unittest

import torch
from torch import nn

from semilearn.nets.bert.bert_vad_rank import (
    BertVADRank3D, pseudo_rank_loss, supervised_rank_loss,
)


class TinyBackbone(nn.Module):
    num_features = 4

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(4, 3)

    def encode(self, **tokens):
        return tokens["features"]

    def regress(self, features):
        return self.head(features)


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


if __name__ == "__main__":
    unittest.main()
