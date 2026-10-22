"""The HPL meta head receives P1's mask-aware mean-pooled features."""
import unittest
from types import SimpleNamespace
import torch
from torch import nn
from scripts.vad_research.m0.train_hpl_vad import pooled_features


class DummyEncoder(nn.Module):
    def forward(self, **tokens):
        return SimpleNamespace(last_hidden_state=tokens['states'])


class PoolingTests(unittest.TestCase):
    def test_padding_is_excluded(self):
        model=SimpleNamespace(encoder=DummyEncoder(),dropout=nn.Identity())
        tokens={'states':torch.tensor([[[2.,4.],[4.,6.],[100.,200.]]]),
                'attention_mask':torch.tensor([[1,1,0]])}
        result=pooled_features(model,tokens)
        torch.testing.assert_close(result,torch.tensor([[3.,5.]]))


if __name__=='__main__':
    unittest.main()
