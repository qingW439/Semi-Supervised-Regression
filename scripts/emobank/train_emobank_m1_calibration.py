#!/usr/bin/env python3
"""EmoBank M1-P1: conservative OOF regression calibration -> pseudo-label ablation.

THIS IS A DIAGNOSTIC/BASELINE, NOT A CLAIMED PAPER CONTRIBUTION.

- diagnose: requires existing 500-label 5-fold OOF CSV, existing full-teacher
  DEV predictions, and EXISTING unlabeled MC pseudo-label cache. No new BERT
  inference or training. Fits correction and risk on TRAIN OOF labels only.
  Reports DEV metrics for diagnosis; never fits on DEV and never reads TEST.
- corrected: fixed-teacher self-training with *corrected* continuous pseudo labels.
- weighted: same corrected pseudo labels, weighted by calibrated TRAIN-only
  probability of NOT exceeding the pre-set 0.25 absolute error limit.
- control/all: matching ablations, using the same training implementation.

The correction is a guarded, per-dimension affine map fit to OOF predictions:
  calibrated(p) = clip(p + clip(mean_y + slope*(p-mean_p)-p, +/-max_adjust), 1,5)
  slope is ridge-fitted then bounded to [min_slope, max_slope].
OOF teachers each see 400 labels whereas the final teacher sees 500; mapping
transfer is a hypothesis that is TESTED, not assumed. Compare median baselines,
CCC and error distribution; a lower MAE from shrinkage alone isn't enough.
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
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.emobank.train_emobank_supervised import set_seed, predict, metric_report
from scripts.emobank.train_emobank_selection_ablation import (
    UCollator, load_model, on_device, sha256_file, validate_splits
)
from semilearn.datasets.nlp_datasets.emobank import (
    GoldData, UnlabeledData, TextCollator, read_gold
)

DIMS = ('V', 'A', 'D')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', required=True,
                   choices=('diagnose', 'control', 'all', 'corrected', 'weighted'))
    p.add_argument('--budget', type=int, choices=(250,500,1000), default=500)
    p.add_argument('--seed', type=int, choices=(0,1,2), default=0)
    p.add_argument('--mc-passes', type=int, default=8)
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--unlabeled-batch-size', type=int, default=16)
    p.add_argument('--eval-batch-size', type=int, default=16)
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
    p.add_argument('--teacher-cache', type=Path)
    p.add_argument('--calibration-dir', type=Path)
    p.add_argument('--out', type=Path)
    # Pre-specified pilot hyperparameters, NOT optimized using Dev data.
    p.add_argument('--ridge-alpha', type=float, default=2.0)
    p.add_argument('--min-slope', type=float, default=0.5)
    p.add_argument('--max-slope', type=float, default=1.5)
    p.add_argument('--max-adjust', type=float, default=0.12,
                   help='Maximum absolute VAD correction, original 1-5 scale')
    p.add_argument('--error-limit', type=float, default=0.25)
    p.add_argument('--offline', action='store_true')
    p.add_argument('--device', choices=('auto','cuda','cpu'), default='auto')
    a=p.parse_args()
    if min(a.epochs,a.batch_size,a.unlabeled_batch_size,a.eval_batch_size,a.mc_passes,a.max_length) < 1:
        p.error('epochs, batch sizes, mc-passes and max-length must be >= 1')
    if a.lr<=0 or a.unsup_weight<0 or a.ridge_alpha<0 or a.error_limit<=0:
        p.error('invalid nonpositive parameter')
    if a.max_adjust<=0 or not (0 < a.min_slope <= a.max_slope):
        p.error('max-adjust and slope bounds must be positive and ordered')
    base=ROOT/'saved_models'/'emobank'
    if a.teacher_checkpoint is None:
        a.teacher_checkpoint=base/f'supervised_lb{a.budget}_s{a.seed}'/'model_best.pth'
    if a.oof_dir is None:
        a.oof_dir=base/f'm1_oof_diag_lb{a.budget}_s{a.seed}'
    if a.teacher_cache is None:
        a.teacher_cache=base/f'selection_pilot_cache_lb{a.budget}_s{a.seed}_mc{a.mc_passes}.npz'
    if a.calibration_dir is None:
        a.calibration_dir=base/f'm1_calibration_lb{a.budget}_s{a.seed}'
    if a.out is None and a.mode!='diagnose':
        a.out=base/f'm1_{a.mode}_lb{a.budget}_s{a.seed}'
    for path in (a.teacher_checkpoint,a.teacher_cache):
        if not path.is_file(): p.error(f'Required existing file missing: {path}')
    if a.mode == 'diagnose':
        if a.calibration_dir.exists() and any(a.calibration_dir.iterdir()):
            p.error(f'Diagnostic output already exists; refusing overwrite: {a.calibration_dir}')
    else:
        if not (a.calibration_dir/'calibrated_cache.npz').is_file():
            p.error('Run --mode diagnose first to generate calibrated_cache.npz')
        if a.out.exists() and any(a.out.iterdir()):
            p.error(f'Training output exists; refusing overwrite: {a.out}')
    return a


def parse_cache(path, ids, teacher_sha, budget, seed, passes):
    with np.load(path,allow_pickle=False) as z:
        meta=json.loads(z['meta'].item())
        keys=('budget','seed','mc_passes','teacher_sha256')
        vals=(budget,seed,passes,teacher_sha)
        if tuple(meta[k] for k in keys)!=vals:
            raise RuntimeError('Unlabeled pseudo cache does not match E1 teacher or split')
        cache_ids=z['ids'].astype(str).tolist()
        if cache_ids!=list(map(str,ids)):
            raise RuntimeError('Unlabeled IDs/order differ from cached pseudo labels')
        pseudo=z['pseudo_norm'].copy()
        if pseudo.shape!=(len(ids),3) or not np.isfinite(pseudo).all():
            raise RuntimeError('Invalid cached teacher pseudo labels')
    return pseudo,meta


def affine_params(pred,gold,alpha,min_slope,max_slope):
    x=np.asarray(pred,dtype=np.float64)
    y=np.asarray(gold,dtype=np.float64)
    if len(x)<30 or not(np.isfinite(x).all() and np.isfinite(y).all()):
        raise RuntimeError('Invalid or too-small OOF calibration training sample')
    fit=Ridge(alpha=alpha,fit_intercept=True)
    fit.fit(x[:,None],y)
    raw_slope=float(fit.coef_[0])
    slope=float(np.clip(raw_slope,min_slope,max_slope))
    return dict(train_pred_mean=float(x.mean()),train_gold_mean=float(y.mean()),
                slope=slope,unclipped_slope=raw_slope,n_train=int(len(x)))


def adjust(pred,model,max_adjust):
    p=np.asarray(pred,dtype=np.float64)
    calibrated=(model['train_gold_mean']+model['slope']*(p-model['train_pred_mean']))
    corrected=p+np.clip(calibrated-p,-max_adjust,max_adjust)
    return np.clip(corrected,1.0,5.0)


def risk_features(pred_corr, raw_pred, text_chars, mean):
    """Only features computable for the OLD unlabeled cache (no MCstd there)."""
    p=np.asarray(pred_corr,dtype=np.float64)
    raw=np.asarray(raw_pred,dtype=np.float64)
    return np.column_stack((p, np.abs(p-mean),
                            np.abs(p-raw), np.asarray(text_chars,dtype=np.float64)))


def numeric_report(y,p):
    y=np.asarray(y,dtype=np.float64);p=np.asarray(p,dtype=np.float64)
    if not(len(y)>0 and np.isfinite(y).all() and np.isfinite(p).all()):
        raise RuntimeError('Invalid numeric report arrays')
    var=np.var(y)
    cov=np.mean((y-y.mean())*(p-p.mean()))
    denom=var+np.var(p)+(y.mean()-p.mean())**2
    corr=(cov/np.sqrt(var*np.var(p))) if var>1e-14 and np.var(p)>1e-14 else None
    return dict(mae=float(np.mean(np.abs(y-p))),
                rmse=float(np.sqrt(np.mean((y-p)**2))),
                ccc=float(2*cov/denom) if denom>1e-14 else None,
                pearson=float(corr) if corr is not None else None,
                r2=float(1-np.mean((y-p)**2)/var) if var>1e-14 else None,
                std=float(np.std(p)),bias=float(np.mean(p-y)))


def diagnose(a,lb,ub,teacher_hash):
    oof_path=a.oof_dir/'oof_predictions.csv'
    dev_path=a.oof_dir/'dev_diagnostic_predictions.csv'
    for path in (oof_path,dev_path):
        if not path.is_file():raise FileNotFoundError(path)
    oof=pd.read_csv(oof_path);dev=pd.read_csv(dev_path)
    if len(oof)!=a.budget or oof['id'].astype(str).nunique()!=a.budget:
        raise RuntimeError('OOF must have exactly one prediction per labeled train ID')
    if set(oof['id'].astype(str))!=set(lb['id'].astype(str)):
        raise RuntimeError('OOF ID set does not match labeled TRAIN set')
    if 'fold' not in oof or oof['fold'].nunique()!=5:
        raise RuntimeError('Expected existing complete five-fold OOF predictions')
    if len(dev)!=1000 or dev['id'].astype(str).nunique()!=1000:
        raise RuntimeError('DEV diagnostic needs exactly 1000 independent predictions')
    if set(dev['id'].astype(str)) & set(oof['id'].astype(str)):
        raise RuntimeError('OOF labeled and DEV IDs overlap')
    dev_gold=read_gold(a.data_root/'dev_gold.parquet',expected_n=1000)
    if set(dev_gold['id'].astype(str))!=set(dev['id'].astype(str)):
        raise RuntimeError('DEV prediction IDs and DEV gold IDs differ')
    aligned=dev_gold.set_index(dev_gold['id'].astype(str)).loc[dev['id'].astype(str)]
    for d in DIMS:
        if not np.allclose(dev[f'gold_{d}'],aligned[d],atol=1e-5):
            raise RuntimeError(f'DEV gold alignment failed for {d}')
    raw_ulb,base_meta=parse_cache(a.teacher_cache,ub['id'].tolist(),teacher_hash,
                                  a.budget,a.seed,a.mc_passes)
    raw_ulb_orig=raw_ulb.astype(np.float64)*4+1
    corrected=np.empty_like(raw_ulb_orig)
    unlab_risk=np.empty_like(raw_ulb_orig)
    report={'purpose':'M1 calibration pilot, numerical pseudo-target quality (NOT a proposed novelty)',
            'source_teacher_checkpoint':str(a.teacher_checkpoint),
            'oof_file_sha256':sha256_file(oof_path),
            'teacher_cache_sha256':sha256_file(a.teacher_cache),
            'teacher_sha256':teacher_hash,
            'train_gold_used_for_fitting':True,
            'dev_labels_used_for_fitting':False, 'test_accessed':False,
            'oof_teacher_training_labels':a.budget*(5-1)//5,
            'full_teacher_training_labels':a.budget,
            'warning':'OOF teacher trained with 400 labels; full teacher has 500. Calibration transfer and error-probability accuracy unverified.',
            'per_dimension':{}}
    train_mean=lb[list(DIMS)].mean().to_numpy(dtype=np.float64)
    train_median=lb[list(DIMS)].median().to_numpy(dtype=np.float64)
    for idx,d in enumerate(DIMS):
        # OOF calibration validation is CROSS-FIT by fold. No fold's labels
        # are used to fit the calibrator that predicts that same fold.
        cross_corr=np.full(len(oof),np.nan,dtype=np.float64)
        fold_models=[]
        for fold in sorted(oof['fold'].unique()):
            held=(oof['fold']==fold).to_numpy()
            model=affine_params(oof.loc[~held,f'pred_{d}'],oof.loc[~held,f'gold_{d}'],
                                a.ridge_alpha,a.min_slope,a.max_slope)
            cross_corr[held]=adjust(oof.loc[held,f'pred_{d}'],model,a.max_adjust)
            fold_models.append(model)
        if np.isnan(cross_corr).any():raise RuntimeError(f'Incomplete crossfit for {d}')
        final_model=affine_params(oof[f'pred_{d}'],oof[f'gold_{d}'],
                                  a.ridge_alpha,a.min_slope,a.max_slope)
        dev_raw=np.clip(dev[f'pred_{d}'].to_numpy(dtype=np.float64),1,5)
        dev_corr=adjust(dev[f'pred_{d}'],final_model,a.max_adjust)
        ydev=dev[f'gold_{d}'].to_numpy(dtype=np.float64)
        y_oof=oof[f'gold_{d}'].to_numpy(dtype=np.float64)
        oof_raw=np.clip(oof[f'pred_{d}'].to_numpy(dtype=np.float64),1,5)
        # Model post-correction error risk from calibrated OOF predictions
        # that were generated without using held-fold calibration labels.
        errs=np.abs(cross_corr-y_oof)
        labels=(errs>a.error_limit).astype(int)
        if len(np.unique(labels))<2:
            raise RuntimeError(f'OOF corrected error threshold has only one class for {d}')
        rmodel=make_pipeline(StandardScaler(),LogisticRegression(C=0.3,max_iter=1000,random_state=0))
        mean_ref=float(train_mean[idx])
        rmodel.fit(risk_features(cross_corr,oof_raw,oof['text_chars'],mean_ref),labels)
        dv_risk=rmodel.predict_proba(risk_features(dev_corr,dev_raw,dev['text_chars'],mean_ref))[:,1]
        gt=(np.abs(dev_corr-ydev)>a.error_limit).astype(int)
        dv_auc=float(roc_auc_score(gt,dv_risk)) if len(np.unique(gt))>1 else None
        if not(0<=np.min(dv_risk) and np.max(dv_risk)<=1):raise RuntimeError('Invalid risk')
        # Fit on TRAIN cross-fit corrected errors, apply to unlabeled using
        # features available both in OOF and the old unlabeled cache.
        corrected[:,idx]=adjust(raw_ulb_orig[:,idx],final_model,a.max_adjust)
        unlab_risk[:,idx]=rmodel.predict_proba(
            risk_features(corrected[:,idx],raw_ulb_orig[:,idx],
                          ub['text'].str.len().to_numpy(),mean_ref))[:,1]
        report['per_dimension'][d]={
            'affine_fit':final_model,
            'oof_crossfit_raw':numeric_report(y_oof,oof_raw),
            'oof_crossfit_corrected':numeric_report(y_oof,cross_corr),
            'dev_raw':numeric_report(ydev,dev_raw),
            'dev_corrected':numeric_report(ydev,dev_corr),
            'dev_train_median':numeric_report(ydev,np.full_like(ydev,train_median[idx])),
            'dev_corrected_error_auc':dv_auc,
            'dev_corrected_gt_error_rate':float(gt.mean()),
            'dev_uncalibrated_prediction_mean':float(dev_raw.mean()),
            'dev_mean_absolute_correction':float(np.abs(dev_corr-dev_raw).mean()),
            'oof_corrected_error_rate':float(labels.mean()),
        }
        print(f'{d}: OOF_MAE {np.abs(y_oof-oof_raw).mean():.4f} -> {np.abs(y_oof-cross_corr).mean():.4f}; '
              f'DEV_MAE {np.abs(ydev-dev_raw).mean():.4f} -> {np.abs(ydev-dev_corr).mean():.4f}; '
              f'DEV_CCC {numeric_report(ydev,dev_raw)["ccc"]:.3f} -> {numeric_report(ydev,dev_corr)["ccc"]:.3f}; '
              f'DEV_median_MAE {np.abs(ydev-train_median[idx]).mean():.4f}',flush=True)
    # These risks target CORRECTED pseudo-label errors, but calibration may
    # still shift when moving from 400-label OOF teachers to the 500-label
    # full teacher; "probability" does not imply a coverage guarantee.
    if not np.isfinite(unlab_risk).all() or (unlab_risk<0).any() or (unlab_risk>1).any():
        raise RuntimeError('Invalid corrected error-risk estimates')
    weights=np.clip(1-unlab_risk,0.05,1.0)
    items=report['per_dimension']
    report['summary']={
        'dev_raw_macro_mae':float(np.mean([items[d]['dev_raw']['mae'] for d in DIMS])),
        'dev_corrected_macro_mae':float(np.mean([items[d]['dev_corrected']['mae'] for d in DIMS])),
        'dev_median_macro_mae':float(np.mean([items[d]['dev_train_median']['mae'] for d in DIMS])),
        'dev_raw_macro_ccc':float(np.mean([items[d]['dev_raw']['ccc'] for d in DIMS])),
        'dev_corrected_macro_ccc':float(np.mean([items[d]['dev_corrected']['ccc'] for d in DIMS])),
        'unlabeled_error_weight_mean_by_dim':dict(zip(DIMS,[float(x) for x in weights.mean(axis=0)])),
    }
    m=report['summary']
    print(f'DEV MACRO: MAE raw={m["dev_raw_macro_mae"]:.5f} '
          f'corrected={m["dev_corrected_macro_mae"]:.5f} '
          f'constant_median={m["dev_median_macro_mae"]:.5f}; '
          f'CCC raw={m["dev_raw_macro_ccc"]:.4f} '
          f'corrected={m["dev_corrected_macro_ccc"]:.4f}',flush=True)
    a.calibration_dir.mkdir(parents=True,exist_ok=True)
    meta={'base_teacher_cache_sha256':sha256_file(a.teacher_cache),
          'oof_sha256':sha256_file(oof_path),
          'teacher_sha256':teacher_hash,'budget':a.budget,'seed':a.seed,
          'mc_passes':a.mc_passes, 'ridge_alpha':a.ridge_alpha,
          'max_adjust':a.max_adjust,'slope_bounds':[a.min_slope,a.max_slope],
          'weights_source':'OOF cross-fit corrected pseudo error risk (held-out fold correction), applied to full-teacher unlabeled outputs',
          'test_accessed':False}
    np.savez_compressed(a.calibration_dir/'calibrated_cache.npz',
                        ids=np.asarray(list(map(str,ub['id'])),dtype=str),
                        pseudo_original_norm=raw_ulb.astype(np.float32),
                        pseudo_corrected_norm=((corrected-1)/4).astype(np.float32),
                        predicted_corrected_error_risk=unlab_risk.astype(np.float32),
                        error_weight=weights.astype(np.float32),
                        meta=np.array(json.dumps(meta)))
    (a.calibration_dir/'calibration_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('DIAGNOSE FINISHED:',a.calibration_dir/'calibration_report.json')
    print('No new BERT training; DEV evaluation only; TEST never accessed.')


def train(a,lb,ub,teacher_hash):
    with np.load(a.calibration_dir/'calibrated_cache.npz',allow_pickle=False) as z:
        meta=json.loads(z['meta'].item())
        ids=z['ids'].astype(str).tolist()
        raw=z['pseudo_original_norm'].copy()
        corr=z['pseudo_corrected_norm'].copy()
        heuristic=z['error_weight'].copy()
    with np.load(a.teacher_cache,allow_pickle=False) as z:
        oldmeta=json.loads(z['meta'].item())
    if ids!=list(map(str,ub['id'])):raise RuntimeError('Unlabeled IDs/order differ')
    if (meta['base_teacher_cache_sha256']!=sha256_file(a.teacher_cache)
        or meta['teacher_sha256']!=teacher_hash
        or meta['oof_sha256']!=sha256_file(a.oof_dir/'oof_predictions.csv')
        or meta['budget']!=a.budget or meta['seed']!=a.seed or meta['mc_passes']!=a.mc_passes):
        raise RuntimeError('Calibration cache is not consistent with base pseudo labels / split')
    if any(v.shape!=(len(ub),3) for v in (raw,corr,heuristic)):
        raise RuntimeError('Invalid correction cache arrays')
    target=torch.tensor(raw if a.mode in ('control','all') else corr,dtype=torch.float32)
    weights=torch.tensor(heuristic if a.mode=='weighted' else np.ones_like(corr),dtype=torch.float32)
    if not torch.isfinite(target).all() or not torch.isfinite(weights).all():
        raise RuntimeError('NaN or inf in pseudo targets/weights')
    set_seed(a.seed+40000)
    device=torch.device('cuda' if a.device=='cuda' or (a.device=='auto' and torch.cuda.is_available()) else 'cpu')
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    model=load_model(a,device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    tok=AutoTokenizer.from_pretrained(a.pretrain_path,use_fast=True,local_files_only=a.offline)
    lb_dl=DataLoader(GoldData(lb),batch_size=a.batch_size,shuffle=True,
                     generator=torch.Generator().manual_seed(a.seed+41000),
                     collate_fn=TextCollator(tok,a.max_length))
    ub_dl=DataLoader(UnlabeledData(ub),batch_size=a.unlabeled_batch_size,shuffle=True,
                     generator=torch.Generator().manual_seed(a.seed+42000),
                     collate_fn=UCollator(tok,a.max_length))
    dev=read_gold(a.data_root/'dev_gold.parquet',expected_n=1000)
    if set(dev['id'].astype(str)) & (set(lb['id'].astype(str))|set(ub['id'].astype(str))):
        raise RuntimeError('DEV overlaps with training samples')
    dev_dl=DataLoader(GoldData(dev),batch_size=a.eval_batch_size,shuffle=False,
                      collate_fn=TextCollator(tok,a.max_length))
    id_to_idx={x:i for i,x in enumerate(ids)}
    cfg={k:(str(v) if isinstance(v,Path) else v) for k,v in vars(a).items()}
    cfg.update({'teacher_sha256':teacher_hash,'calibration_metadata':meta,
                'student_start':'same fixed original E1 checkpoint for all modes',
                'weight_warning':'corrected risk is TRAIN OOF-derived; full teacher 500-vs-400 transfer can be miscalibrated',
                'test_accessed':False})
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'run_config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n')
    ub_it=iter(ub_dl);best=float('inf');best_epoch=0;total_updates=0;history=[]
    for epoch in range(1,a.epochs+1):
        model.train();steps=0;lsup=0;lu=0
        for toks,ys,_ in lb_dl:
            try:ut,ui=next(ub_it)
            except StopIteration:
                ub_it=iter(ub_dl)
                ut,ui=next(ub_it)
            optimizer.zero_grad(set_to_none=True)
            sup=F.l1_loss(model(**on_device(toks,device)),ys.to(device))
            unsup=sup*0
            if a.mode!='control':
                ix=[id_to_idx[str(x)] for x in ui]
                logits=model(**on_device(ut,device))
                pseudo=target[ix].to(device)
                w=weights[ix].to(device)
                unsup=(torch.abs(logits-pseudo)*w).sum()/w.sum().clamp_min(1)
            loss=sup+(a.unsup_weight*unsup if a.mode!='control' else 0.)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
            loss.backward()
            if a.clip_grad_norm>0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),a.clip_grad_norm)
            optimizer.step()
            steps+=1;total_updates+=1;lsup+=float(sup.detach());lu+=float(unsup.detach())
        gold,preds,dev_ids=predict(model,dev_dl,device)
        rep=metric_report(gold,preds)
        mae=rep['macro_mae_original_scale']
        row={'epoch':epoch,'total_updates':total_updates,
             'train_sup_loss_norm':lsup/steps,'train_unsup_loss_norm':lu/steps,
             'dev_macro_mae_original':mae,
             'dev_macro_ccc':float(np.mean([rep[d]['ccc'] for d in DIMS]))}
        history.append(row)
        pd.DataFrame(history).to_csv(a.out/'training_history.csv',index=False)
        print(f'{a.mode} epoch={epoch:02d} updates={total_updates} '
              f'DEV_MAE={mae:.5f} DEV_CCC={row["dev_macro_ccc"]:.4f}',flush=True)
        if mae<best-1e-4:
            best=mae;best_epoch=epoch
            torch.save({'model':model.state_dict(),'epoch':epoch,'dev_metrics':rep,
                        'run_config':cfg},a.out/'model_best.pth')
            (a.out/'dev_metrics_best.json').write_text(json.dumps({'epoch':epoch,**rep},indent=2)+'\n')
    print(f'FINISHED {a.mode}: best_epoch={best_epoch} best_DEV_MAE={best:.5f} '
          f'updates={total_updates}. TEST never accessed.')


def main():
    a=parse_args()
    if a.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but unavailable')
    lb,ub=validate_splits(a)
    teacher_hash=sha256_file(a.teacher_checkpoint)
    if a.mode=='diagnose':diagnose(a,lb,ub,teacher_hash)
    else:train(a,lb,ub,teacher_hash)


if __name__=='__main__':main()
