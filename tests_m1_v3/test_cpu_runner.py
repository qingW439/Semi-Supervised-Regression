"""CPU device safety and benchmark guard tests; no full audit is launched."""
from pathlib import Path
import numpy as np
import pytest
import torch
from scripts.vad_research.m1_v3 import cpu_resources
from scripts.vad_research.m1_v3.build_audit import supervised_update


def test_container_memory_preflight(monkeypatch):
    monkeypatch.setattr(cpu_resources,'available_memory_bytes',lambda: 2*cpu_resources.GIB)
    with pytest.raises(RuntimeError,match='2.00 GiB'):
        cpu_resources.require_cpu_memory()
    monkeypatch.setattr(cpu_resources,'available_memory_bytes',lambda: 16*cpu_resources.GIB)
    assert cpu_resources.require_cpu_memory()==16*cpu_resources.GIB


def test_real_update_helper_operates_on_cpu():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.head=torch.nn.Linear(4,3)
        def forward(self,input_ids,attention_mask):
            return self.head(input_ids.float()*attention_mask.float())
    torch.manual_seed(7)
    model=Tiny().cpu(); optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01)
    before={k:v.detach().clone() for k,v in model.state_dict().items()}
    tokens={'input_ids':torch.ones(2,4),'attention_mask':torch.ones(2,4)}
    target=torch.full((2,3),.7)
    supervised_update(model,optimizer,tokens,target,torch.device('cpu'))
    assert any(not torch.equal(before[k],v) for k,v in model.state_dict().items())
    assert all(torch.isfinite(v).all() for v in model.state_dict().values())
    assert all(state['step'].item()==1 for state in optimizer.state.values())


def test_cpu_launcher_is_explicit_and_fixed():
    script=(Path(__file__).resolve().parents[1]/'scripts/vad_research/m1_v3/run_cpu_audit.sh').read_text()
    for item in ('OMP_NUM_THREADS=8','MKL_NUM_THREADS=8','CUDA_VISIBLE_DEVICES=""',
                 '--device cpu','--check-only','exit_status.txt','build.log','audit.log'):
        assert item in script
    assert 'benchmark_cpu.py' not in script
