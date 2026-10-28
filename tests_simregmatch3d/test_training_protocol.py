from types import SimpleNamespace
import torch
import pytest
from scripts.vad_research.simregmatch3d.train_simregmatch3d import MODES,mode_flags,make_unlabeled_loss,FeatureRegressor
from scripts.vad_research.train_roberta_vad_supervised import RobertaVADRegressor

class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout=torch.nn.Dropout(.5)
        self.head=torch.nn.Linear(2,3)
    def forward_with_features(self,**kw):
        x=kw['x'];return self.head(self.dropout(x)),x
    def forward(self,**kw):return self.forward_with_features(**kw)[0]

def args():return SimpleNamespace(mc_passes=5,temperature=1.0,beta=.6,filter_percentile=95.0)

def test_mode_ablation_flags():
    assert [mode_flags(m) for m in MODES]==[(False,False,False),(True,False,False),(True,True,False),(True,False,True),(True,True,True)]

def test_all_modes_mc_and_finite_gradients():
    torch.manual_seed(2)
    model=TinyModel().train();x=torch.randn(12,2)
    y=torch.rand(4,3)
    for mode in MODES[1:]:
        model.zero_grad(set_to_none=True)
        loss,stats,_=make_unlabeled_loss(model,x[:4],y,{'x':x},mode,args(),10.0)
        assert torch.isfinite(loss) and stats['mc_draw_difference']>0
        assert stats['filter_pass_fraction']==1.0
        assert (stats['calibration_magnitude']>0)==(mode in ('calibration_only','simregmatch_full'))
        loss.backward()
        assert torch.isfinite(model.head.weight.grad).all()

def test_empty_filter_does_not_break_backward():
    model=TinyModel().train();x=torch.randn(10,2)
    loss,stats,_=make_unlabeled_loss(model,x[:4],torch.rand(4,3),{'x':x},'simregmatch_full',args(),0.0)
    assert loss.item()==0 and stats['filter_pass_count']==0
    loss.backward()
    assert torch.equal(model.head.weight.grad,torch.zeros_like(model.head.weight.grad))

def test_feature_regressor_preserves_p1_forward_math():
    class Encoder(torch.nn.Module):
        def forward(self,input_ids,attention_mask,return_dict):
            return SimpleNamespace(last_hidden_state=input_ids.float().unsqueeze(-1).repeat(1,1,2))
    a=RobertaVADRegressor.__new__(RobertaVADRegressor)
    torch.nn.Module.__init__(a)
    a.encoder=Encoder();a.dropout=torch.nn.Dropout(0);a.head=torch.nn.Linear(2,3)
    b=FeatureRegressor.__new__(FeatureRegressor)
    torch.nn.Module.__init__(b)
    b.encoder=Encoder();b.dropout=torch.nn.Dropout(0);b.head=torch.nn.Linear(2,3)
    b.head.load_state_dict(a.head.state_dict())
    tok={'input_ids':torch.tensor([[1,2,3],[2,3,4]]),'attention_mask':torch.tensor([[1,1,0],[1,1,1]])}
    assert torch.equal(a(**tok),b(**tok))
