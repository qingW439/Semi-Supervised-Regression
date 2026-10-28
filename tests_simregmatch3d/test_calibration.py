import torch
import pytest
from scripts.vad_research.simregmatch3d.calibration import mc_statistics,similarity_labels,calibrated_targets

def test_three_dimensional_mc_statistics():
    draws=torch.tensor([[[.2,.3,.4]],[[.4,.5,.6]],[[.6,.7,.8]]])
    mean,var,total=mc_statistics(draws)
    assert mean.shape==(1,3) and var.shape==(1,3) and total.shape==(1,)
    assert torch.allclose(mean,torch.tensor([[.4,.5,.6]]))
    assert torch.allclose(var,torch.full((1,3),.04))
    assert torch.allclose(total,torch.tensor([.12]))

def test_similarity_softmax_and_neighbor_labels():
    u=torch.tensor([[2.,0.],[0.,3.]])
    l=torch.tensor([[1.,0.],[0.,1.]])
    y=torch.tensor([[.1,.2,.3],[.8,.7,.6]])
    neighbor,weights=similarity_labels(u,l,y,1.0)
    assert neighbor.shape==(2,3) and torch.allclose(weights.sum(1),torch.ones(2))
    assert weights[0,0]>weights[0,1] and weights[1,1]>weights[1,0]
    assert torch.all(neighbor>=0) and torch.all(neighbor<=1)

def test_calibrated_targets_detach_and_range():
    model=torch.tensor([[1.2,-.1,.4]],requires_grad=True)
    neighbor=torch.tensor([[.2,.3,.6]],requires_grad=True)
    result=calibrated_targets(model,neighbor,.6)
    assert not result.requires_grad and torch.all(result>=0) and torch.all(result<=1)
    assert torch.allclose(result,torch.tensor([[.8,.06,.48]]))

@pytest.mark.parametrize('bad',[(1,2),(2,2,2)])
def test_mc_shape_rejected(bad):
    with pytest.raises(ValueError):mc_statistics(torch.zeros(bad))
