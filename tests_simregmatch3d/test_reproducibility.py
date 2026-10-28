import argparse
import torch
from scripts.vad_research.simregmatch3d.train_simregmatch3d import parse_args,check_inputs

def test_default_protocol_fixed():
    a=parse_args(['--mode','simregmatch_full'])
    assert (a.steps,a.batch_size,a.ulb_batch_size,a.eval_every,a.max_length)==(160,16,64,32,128)
    assert (a.mc_passes,a.beta,a.temperature,a.lambda_u,a.filter_percentile)==(5,.6,1.0,.01,95.)

def test_forked_mc_randomness_restores_supervised_rng():
    torch.manual_seed(41)
    reference=torch.rand(4)
    torch.manual_seed(41)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(12345)
        _=torch.rand(100)
    assert torch.equal(torch.rand(4),reference)

def test_split_identity_and_checkpoint_hash_when_local_data_available():
    a=parse_args(['--mode','supervised_only'])
    if not a.p1_dir.exists() or not a.data_root.exists():return
    lb,dev,ulb,_,hashes=check_inputs(a)
    assert not (set(lb.id)&set(ulb.id))
    assert not (set(dev.id)&(set(lb.id)|set(ulb.id)))
    assert len(hashes)==3
