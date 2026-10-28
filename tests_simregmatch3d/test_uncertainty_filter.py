import torch
from scripts.vad_research.simregmatch3d.uncertainty_filter import filter_mask,next_threshold,masked_mse

def test_previous_threshold_and_95th_percentile():
    scores=torch.arange(100,dtype=torch.float)
    old=50.0
    mask=filter_mask(scores,old)
    assert int(mask.sum())==50
    assert abs(next_threshold(scores,95.0)-94.05)<1e-4
    assert old==50.0

def test_empty_mask_is_zero_and_differentiable():
    pred=torch.ones(2,3,requires_grad=True)
    target=torch.zeros(2,3)
    loss=masked_mse(pred,target,torch.zeros(2,dtype=torch.bool))
    assert loss.item()==0.0 and torch.isfinite(loss)
    loss.backward()
    assert torch.equal(pred.grad,torch.zeros_like(pred))

def test_selected_sample_loss_sums_three_dimensions():
    pred=torch.ones(2,3,requires_grad=True)
    loss=masked_mse(pred,torch.zeros_like(pred),torch.tensor([True,False]))
    assert loss.item()==3.0
    loss.backward()
    assert torch.equal(pred.grad[1],torch.zeros(3))
