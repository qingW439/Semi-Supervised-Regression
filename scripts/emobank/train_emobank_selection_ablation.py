#!/usr/bin/env python3
"""E1-500 fixed-teacher semi-supervised regression selection ablation.

Four modes: control (labeled-only continued training), all (all teacher pseudo
coordinates), mean50 (50% lowest distance to train mean per V/A/D), risk50
(50% lowest OOF-predicted large-error risk per V/A/D). One common MC teacher
pseudo-label cache is used in all modes; the risk model uses TRAIN OOF ONLY.

This is a controlled diagnostic, NOT GaussianMatch, RankUp or a new paper model.
The same BERT checkpoint initializes the frozen teacher and each student.
No TEST access; DEV is used only for model selection/evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from semilearn.nets.bert.bert_vad import BertVADRegressor
from semilearn.datasets.nlp_datasets.emobank import (GoldData, UnlabeledData, TextCollator,
                                                       read_gold, read_unlabeled)
from scripts.emobank.train_emobank_supervised import (set_seed, predict, metric_report)

DIMS = ('V','A','D')


class UCollator:
    def __init__(self, tok, max_len):
        self.tok, self.max_len = tok, max_len

    def __call__(self, rows):
        texts, ids = zip(*rows)
        return (self.tok(list(texts), truncation=True, max_length=self.max_len,
                         padding=True, return_tensors='pt'), list(ids))


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('prepare','control','all','mean50','risk50'), required=True)
    p.add_argument('--budget', type=int, choices=(250,500,1000), default=500)
    p.add_argument('--seed', type=int, choices=(0,1,2), default=0)
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--unlabeled-batch-size', type=int, default=16)
    p.add_argument('--eval-batch-size', type=int, default=16)
    p.add_argument('--mc-passes', type=int, default=8)
    p.add_argument('--max-length', type=int, default=128)
    p.add_argument('--lr', type=float, default=5e-6)
    p.add_argument('--weight-decay', type=float, default=5e-4)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--unsup-weight', type=float, default=0.2)
    p.add_argument('--clip-grad-norm', type=float, default=1.0)
    p.add_argument('--pretrain-path', default='google/bert_uncased_L-4_H-512_A-8')
    p.add_argument('--data-root', type=Path, default=ROOT.parent/'data'/'emobank_processed')
    p.add_argument('--teacher-checkpoint', type=Path)
    p.add_argument('--oof-dir', type=Path)
    p.add_argument('--cache', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--offline', action='store_true')
    p.add_argument('--device', choices=('auto','cuda','cpu'), default='auto')
    a = p.parse_args()
    if min(a.epochs,a.batch_size,a.unlabeled_batch_size,a.eval_batch_size,a.mc_passes,a.max_length) < 1:
        p.error('epochs, batch sizes, views and max-length must be >=1')
    if a.unsup_weight<0 or a.lr<=0:
        p.error('unsup-weight must be non-negative; lr positive')
    if a.teacher_checkpoint is None:
        a.teacher_checkpoint=ROOT/'saved_models'/'emobank'/f'supervised_lb{a.budget}_s{a.seed}'/'model_best.pth'
    if a.oof_dir is None:
        a.oof_dir=ROOT/'saved_models'/'emobank'/f'm1_oof_diag_lb{a.budget}_s{a.seed}'
    if a.cache is None:
        a.cache=ROOT/'saved_models'/'emobank'/f'selection_pilot_cache_lb{a.budget}_s{a.seed}_mc{a.mc_passes}.npz'
    if a.out is None and a.mode!='prepare':
        a.out=ROOT/'saved_models'/'emobank'/f'selection_pilot_{a.mode}_lb{a.budget}_s{a.seed}'
    if not a.teacher_checkpoint.is_file():
        p.error(f'Teacher checkpoint missing: {a.teacher_checkpoint}')
    if a.mode!='prepare' and a.out.exists() and any(a.out.iterdir()):
        p.error(f'Output directory not empty (refuse overwrite): {a.out}')
    return a


def on_device(toks, device):
    return {k:v.to(device) for k,v in toks.items()}


def sha256_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def risk_features(df, dim):
    return df[[f'mc_std_{dim}',f'pred_{dim}',f'mean_dist_{dim}', 'text_chars']].to_numpy(dtype=np.float64)


def make_risk_model(oof, d):
    error=np.abs(oof[f'pred_{d}']-oof[f'gold_{d}'])
    target=(error>0.25).astype(int)
    if target.nunique()<2:
        raise RuntimeError(f'Only one OOF error class for {d}')
    model=make_pipeline(StandardScaler(),LogisticRegression(C=0.3,max_iter=1000,random_state=0))
    model.fit(risk_features(oof,d),target)
    return model


def load_model(a, device):
    model=BertVADRegressor(a.pretrain_path,dropout=a.dropout,offline=a.offline).to(device)
    state=torch.load(a.teacher_checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(state['model'],strict=True)
    return model


def validate_splits(a):
    split=a.data_root/f'lb{a.budget}_s{a.seed}'
    lb=read_gold(split/'train_lb.parquet', expected_n=a.budget)
    ub=read_unlabeled(split/'train_ulb.parquet',labeled_ids=lb['id'])
    if len(ub)!=8062-a.budget:
        raise RuntimeError(f'Unlabeled count wrong: {len(ub)}')
    return lb, ub


def prepare(a, lb, ub, tok, device, teacher_hash):
    if a.cache.exists():
        raise FileExistsError(f'Cache exists, will not overwrite: {a.cache}')
    path=a.oof_dir/'oof_predictions.csv'
    if not path.is_file():
        raise FileNotFoundError(f'OOF predictions not found: {path}')
    oof=pd.read_csv(path)
    if len(oof)!=a.budget or set(oof['id'].astype(str))!=set(lb['id'].astype(str)):
        raise RuntimeError('OOF IDs must equal exactly the labeled TRAIN IDs')
    risk_models={d:make_risk_model(oof,d) for d in DIMS}
    teacher=load_model(a,device)
    teacher.requires_grad_(False)
    loader=DataLoader(UnlabeledData(ub),batch_size=a.eval_batch_size,shuffle=False,
                      collate_fn=UCollator(tok,a.max_length))
    mu=lb[list(DIMS)].mean().to_numpy(dtype=np.float64)
    pred_orig, std_orig, seen_ids=[],[],[]
    # Shared seed ensures the pseudo-cache is stable across all four modes.
    set_seed(a.seed+30000)
    with torch.inference_mode():
        for bi,(toks,ids) in enumerate(loader):
            toks=on_device(toks,device)
            teacher.train()  # stochastic Dropout only, no BERT batchnorm running stats
            draws=torch.stack([teacher(**toks) for _ in range(a.mc_passes)],0)
            pred_orig.append((draws.mean(0)*4+1).cpu().numpy())
            std_orig.append((draws.std(0,unbiased=False)*4).cpu().numpy())
            seen_ids.extend(str(x) for x in ids)
            if (bi+1)%100==0:
                print(f'PREPARE: {len(seen_ids)}/{len(ub)} unlabeled predicted',flush=True)
    if seen_ids!=list(map(str,ub['id'].tolist())):
        raise RuntimeError('Unlabeled ID order mismatch in precompute')
    pred_orig=np.concatenate(pred_orig).astype(np.float32)
    std_orig=np.concatenate(std_orig).astype(np.float32)
    df=pd.DataFrame({'text_chars':ub['text'].str.len().to_numpy()})
    risk=np.empty_like(pred_orig)
    mean_distance=np.empty_like(pred_orig)
    for j,d in enumerate(DIMS):
        df[f'pred_{d}']=pred_orig[:,j]
        df[f'mc_std_{d}']=std_orig[:,j]
        df[f'mean_dist_{d}']=np.abs(pred_orig[:,j]-mu[j])
        risk[:,j]=risk_models[d].predict_proba(risk_features(df,d))[:,1]
        mean_distance[:,j]=df[f'mean_dist_{d}'].to_numpy()
    n=len(ub)
    masks={}
    for key,score in [('mean50',mean_distance),('risk50',risk)]:
        sel=np.zeros_like(score,dtype=np.uint8)
        for d in range(3):
            idx=np.argsort(score[:,d],kind='stable')[:n//2]
            sel[idx,d]=1
        masks[key]=sel
    meta={'budget':a.budget,'seed':a.seed,'mc_passes':a.mc_passes,
          'teacher_sha256':teacher_hash,'oof_file_sha256':sha256_file(path),
          'unlabeled_n':n,'normalized_pseudo_clipped_to_0_1':True,
          'risk_fitted_with_dev_labels':False,'test_accessed':False,
          'teacher_type':'frozen original E1 L1 checkpoint',
          'note':'pilot uses MC pseudo-label mean; mean50 and risk50 each keep exactly 50% of unlabeled examples per dimension'}
    a.cache.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.cache,meta=np.array(json.dumps(meta)),ids=np.asarray(seen_ids,dtype=str),
                        pseudo_norm=np.clip((pred_orig-1)/4,0,1).astype(np.float32),
                        risk=risk.astype(np.float32),mean_distance=mean_distance.astype(np.float32),
                        mask_mean50=masks['mean50'],mask_risk50=masks['risk50'])
    print('PREPARE DONE:',a.cache, 'n=',n,flush=True)
    print('per-dim selected for mean50/risk50:',n//2, '/',n,flush=True)


def train(a, lb, ub, tok, device, teacher_hash):
    if not a.cache.is_file():
        raise FileNotFoundError(f'Pseudo-cache missing; first run --mode prepare: {a.cache}')
    oof_file=a.oof_dir/'oof_predictions.csv'
    with np.load(a.cache,allow_pickle=False) as z:
        meta=json.loads(z['meta'].item())
        if (meta['budget'],meta['seed'],meta['mc_passes'],meta['teacher_sha256']) != (a.budget,a.seed,a.mc_passes,teacher_hash):
            raise RuntimeError('Cache parameters/checkpoint do not match this run')
        if meta['oof_file_sha256']!=sha256_file(oof_file):
            raise RuntimeError('OOF training predictions have changed since caching')
        ids=z['ids'].astype(str).tolist()
        if ids!=list(map(str,ub['id'].tolist())):
            raise RuntimeError('Cache unlabeled IDs are not identical to dataset order')
        pseudo=torch.from_numpy(z['pseudo_norm'].copy())
        mask=torch.ones_like(pseudo) if a.mode=='all' else (
            torch.from_numpy(z['mask_mean50'].copy()).float() if a.mode=='mean50' else (
            torch.from_numpy(z['mask_risk50'].copy()).float() if a.mode=='risk50' else torch.zeros_like(pseudo)))
    set_seed(a.seed+40000)
    model=load_model(a,device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    id_to_idx={x:i for i,x in enumerate(ids)}
    lb_dl=DataLoader(GoldData(lb), batch_size=a.batch_size,shuffle=True,
                     generator=torch.Generator().manual_seed(a.seed+41000),
                     collate_fn=TextCollator(tok,a.max_length))
    ub_dl=DataLoader(UnlabeledData(ub),batch_size=a.unlabeled_batch_size,shuffle=True,
                     generator=torch.Generator().manual_seed(a.seed+42000),
                     collate_fn=UCollator(tok,a.max_length))
    dev=read_gold(a.data_root/'dev_gold.parquet',expected_n=1000)
    if set(dev['id'].astype(str)) & (set(lb['id'].astype(str))|set(ub['id'].astype(str))):
        raise RuntimeError('DEV intersects labeled or unlabeled train')
    dev_dl=DataLoader(GoldData(dev),batch_size=a.eval_batch_size,shuffle=False,
                      collate_fn=TextCollator(tok,a.max_length))
    a.out.mkdir(parents=True,exist_ok=True)
    cfg={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    cfg.update({'teacher_sha256':teacher_hash,'pseudo_cache_meta':meta,
                'student_initialized_from':'frozen E1 teacher checkpoint',
                'training_scheme':'supervised continuation + optional fixed teacher pseudo-loss',
                'test_accessed':False})
    (a.out/'run_config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n')
    ub_it=iter(ub_dl)
    best=float('inf');best_epoch=0;history=[]
    total_updates=0
    for epoch in range(1,a.epochs+1):
        model.train();s_l=0.;u_l=0.;used=0;possible=0;steps=0
        for toks,ys,_ in lb_dl:
            try:
                ut,ui=next(ub_it)
            except StopIteration:
                ub_it=iter(ub_dl)
                ut,ui=next(ub_it)
            optimizer.zero_grad(set_to_none=True)
            sup=F.l1_loss(model(**on_device(toks,device)),ys.to(device))
            unsup=sup*0.
            if a.mode!='control':
                idx=[id_to_idx[str(x)] for x in ui]
                tt=model(**on_device(ut,device))
                target=pseudo[idx].to(device)
                weights=mask[idx].to(device)
                used+=int(weights.sum().item())
                possible+=weights.numel()
                unsup=((tt-target).abs()*weights).sum()/weights.sum().clamp_min(1)
            loss=sup+(a.unsup_weight*unsup if a.mode!='control' else 0.)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            loss.backward()
            if a.clip_grad_norm>0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),a.clip_grad_norm)
            optimizer.step()
            s_l+=float(sup.detach());u_l+=float(unsup.detach());steps+=1
            total_updates+=1
        gold,pred,dev_ids=predict(model,dev_dl,device)
        report=metric_report(gold,pred)
        mae=report['macro_mae_original_scale']
        coverage=used/max(1,possible)
        row={'epoch':epoch,'train_steps':total_updates,'train_sup_loss_norm':s_l/steps,
             'train_unsup_loss_norm':u_l/steps,'pseudo_coord_coverage':coverage,
             'dev_macro_mae_original':mae,
             **{f'dev_{d}_mae':report[d]['mae'] for d in DIMS},
             **{f'dev_{d}_ccc':report[d]['ccc'] for d in DIMS}}
        history.append(row)
        pd.DataFrame(history).to_csv(a.out/'training_history.csv',index=False)
        print(f'{a.mode} epoch={epoch:02d} updates={total_updates} MAE={mae:.5f} '
              f'CCC=({report["V"]["ccc"]:.3f},{report["A"]["ccc"]:.3f},{report["D"]["ccc"]:.3f}) '
              f'coverage={coverage:.3f}',flush=True)
        if mae<best-1e-4:
            best=mae;best_epoch=epoch
            torch.save({'model':model.state_dict(),'epoch':epoch,'dev_metrics':report,'run_config':cfg},
                       a.out/'model_best.pth')
            (a.out/'dev_metrics_best.json').write_text(json.dumps({'epoch':epoch,**report},indent=2)+'\n')
    print(f'FINISHED mode={a.mode} best_epoch={best_epoch} best_dev_MAE={best:.5f} '
          f'updates={total_updates}; Test never opened',flush=True)


def main():
    a=arguments()
    device=torch.device('cuda' if a.device=='cuda' or (a.device=='auto' and torch.cuda.is_available()) else 'cpu')
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if a.mode=='prepare' and a.cache.exists():
        raise FileExistsError(f'Will not overwrite existing cache: {a.cache}')
    lb,ub=validate_splits(a)
    tok=AutoTokenizer.from_pretrained(a.pretrain_path,use_fast=True,local_files_only=a.offline)
    teacher_hash=sha256_file(a.teacher_checkpoint)
    if a.mode=='prepare':prepare(a,lb,ub,tok,device,teacher_hash)
    else:train(a,lb,ub,tok,device,teacher_hash)

if __name__=='__main__':
    main()
