#!/usr/bin/env python3
"""Short real RoBERTa CPU updates for timing ONLY; never a GO/NO-GO audit."""
import argparse, json, os, statistics, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO))
from semilearn.datasets.nlp_datasets.emobank import GoldData, TextCollator, read_gold
from scripts.vad_research.train_roberta_vad_supervised import RobertaVADRegressor, sha256_file
from scripts.vad_research.m1_v3.build_audit import partition, seed_all, supervised_update
from scripts.vad_research.m1_v3.cpu_resources import require_cpu_memory

def forever(loader):
    while True:
        yield from loader

def phase(model, optimizer, batches, device, warmup, timed):
    durations=[]
    for index in range(warmup+timed):
        start=time.perf_counter()
        tokens, labels, _=next(batches)
        supervised_update(model,optimizer,tokens,labels,device)
        elapsed=time.perf_counter()-start
        if index>=warmup: durations.append(elapsed)
    return {'mean_seconds':statistics.mean(durations),
            'median_seconds':statistics.median(durations),
            'min_seconds':min(durations),'max_seconds':max(durations),
            'samples_seconds':durations}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',type=Path,default=REPO.parent/'data/emobank_processed/lb500_s0/train_lb.parquet')
    p.add_argument('--pretrain-path',type=Path,default=REPO.parent/'pretrained/roberta-base')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--warmup-steps',type=int,default=2)
    p.add_argument('--timed-steps',type=int,default=3)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists(): raise FileExistsError(f'Output exists: {a.out}')
    if a.warmup_steps<1 or a.timed_steps<1: p.error('Step counts must be positive')
    memory_bytes=require_cpu_memory()
    frame=read_gold(a.train,expected_n=500).reset_index(drop=True)
    fit,audit,_=partition(frame['id'].to_numpy(),a.seed)
    assert not (set(frame.iloc[fit]['id'])&set(frame.iloc[audit]['id']))
    fit_frame=frame.iloc[fit].reset_index(drop=True)
    seed_all(a.seed+200); device=torch.device('cpu')
    tokenizer=AutoTokenizer.from_pretrained(a.pretrain_path,local_files_only=True,use_fast=True)
    collator=TextCollator(tokenizer,128)
    loader=DataLoader(GoldData(fit_frame),batch_size=16,shuffle=True,collate_fn=collator,
                      generator=torch.Generator().manual_seed(a.seed+217))
    batches=forever(loader)
    start=time.perf_counter()
    model=RobertaVADRegressor(str(a.pretrain_path),dropout=0.1,offline=True).to(device)
    model_load_seconds=time.perf_counter()-start
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=0.01)
    p1=phase(model,optimizer,batches,device,a.warmup_steps,a.timed_steps)
    optimizer=torch.optim.AdamW(model.parameters(),lr=5e-6,weight_decay=0.01)
    p2=phase(model,optimizer,batches,device,a.warmup_steps,a.timed_steps)
    # 5 folds x 320 examples x 20 epochs + full 400 x 20 epochs, batch 16.
    p1_full_steps=5*20*(320//16)+20*(400//16)
    p2_full_steps=6*160
    estimate=p1_full_steps*p1['mean_seconds']+p2_full_steps*p2['mean_seconds']+6*model_load_seconds
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=REPO,capture_output=True,text=True,check=True).stdout.strip()
    report={'kind':'BENCHMARK_ONLY','go_no_go_eligible':False,'device':'cpu',
      'seed':a.seed,'git_commit':commit,'train_sha256':sha256_file(a.train),
      'fit_count':400,'audit_count':100,'batch_size':16,'max_length':128,
      'warmup_steps_per_phase':a.warmup_steps,'timed_steps_per_phase':a.timed_steps,
      'omp_num_threads':os.environ.get('OMP_NUM_THREADS'),
      'mkl_num_threads':os.environ.get('MKL_NUM_THREADS'),
      'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
      'torch_num_threads':torch.get_num_threads(),
      'container_memory_gib':memory_bytes/(1024**3),
      'model_load_seconds':model_load_seconds,'p1_step':p1,'p2_step':p2,
      'full_p1_steps':p1_full_steps,'full_p2_steps':p2_full_steps,
      'full_training_steps':p1_full_steps+p2_full_steps,
      'estimated_training_and_model_loading_seconds':estimate,
      'estimate_excludes':'feature extraction, inference, checkpoint I/O, data preprocessing'}
    a.out.mkdir(parents=True)
    (a.out/'benchmark.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__': main()
