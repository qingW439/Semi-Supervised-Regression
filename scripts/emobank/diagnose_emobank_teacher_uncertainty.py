#!/usr/bin/env python3
"""Dev-only pilot diagnostic: does MC-dropout uncertainty predict V/A/D regression errors?

Does not train any model, update weights, or access TEST. Does not calibrate thresholds.
Requires current emobank-vad repository, existing E1 checkpoint, HF tokenizer cache.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr, rankdata
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from semilearn.nets.bert.bert_vad import BertVADRegressor
from semilearn.datasets.nlp_datasets.emobank import GoldData, TextCollator, read_gold
from scripts.emobank.train_emobank_supervised import set_seed

DIMS = ('V','A','D')


def auc_for_high_error(score: np.ndarray, large_error: np.ndarray):
    """Mann-Whitney AUROC for uncertainty as detector of errors above a fixed cutoff."""
    npos=int(large_error.sum()); nneg=int((~large_error).sum())
    if npos==0 or nneg==0:
        return None
    ranks=rankdata(score,method='average')
    auc=(ranks[large_error].sum()-npos*(npos+1)/2)/(npos*nneg)
    return float(auc)


def evaluate_uncertainty(unc: np.ndarray, errors: np.ndarray, large_error_limit: float):
    if unc.ndim!=1 or errors.shape!=unc.shape or len(unc)==0:
        raise ValueError('expected paired nonempty 1D score/error arrays')
    corr=spearmanr(unc,errors).statistic if np.std(unc)>1e-12 and np.std(errors)>1e-12 else None
    ascending=np.argsort(unc,kind='stable')
    coverage=[]
    for fraction in (0.10,0.25,0.50,0.75,1.0):
        n=max(1,int(round(len(unc)*fraction)))
        idx=ascending[:n]
        coverage.append({'fraction':fraction,'retained':n,
                         'mean_abs_error_1to5':float(errors[idx].mean()),
                         'large_error_rate':float((errors[idx]>large_error_limit).mean()),
                         'mc_std_cutoff_1to5':float(unc[idx].max())})
    return {'mc_std_mean_1to5':float(unc.mean()),
            'mae_mc_pseudolabel_1to5':float(errors.mean()),
            'spearman_std_vs_abs_error':float(corr) if corr is not None and np.isfinite(corr) else None,
            'auc_detect_error_gt_fixed_limit':auc_for_high_error(unc,errors>large_error_limit),
            'overall_large_error_rate':float((errors>large_error_limit).mean()),
            'error_by_uncertainty_coverage':coverage}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--budget',type=int,choices=(250,500,1000),default=500)
    p.add_argument('--seed',type=int,choices=(0,1,2),default=0)
    p.add_argument('--checkpoint',type=Path,default=None)
    p.add_argument('--data-root',type=Path,default=REPO.parent/'data'/'emobank_processed')
    p.add_argument('--pretrain-path',default='google/bert_uncased_L-4_H-512_A-8')
    p.add_argument('--max-length',type=int,default=128)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--mc-passes',type=int,default=8)
    p.add_argument('--large-error-limit',type=float,default=0.25,
                   help='Diagnostic large error cutoff on original [1,5] scale; not a calibrated training threshold')
    p.add_argument('--offline',action='store_true')
    p.add_argument('--out',type=Path,default=None)
    a=p.parse_args()
    if a.batch_size<1 or a.mc_passes<2 or a.large_error_limit<=0:
        p.error('invalid diagnostic arguments')
    set_seed(a.seed+18010)
    checkpoint=a.checkpoint or REPO/'saved_models'/'emobank'/f'supervised_lb{a.budget}_s{a.seed}'/'model_best.pth'
    if not checkpoint.is_file(): p.error(f'missing checkpoint: {checkpoint}')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok=AutoTokenizer.from_pretrained(a.pretrain_path,use_fast=True,local_files_only=a.offline)
    dev=read_gold(a.data_root/'dev_gold.parquet',expected_n=1000)
    loader=DataLoader(GoldData(dev),batch_size=a.batch_size,shuffle=False,
                      num_workers=0,collate_fn=TextCollator(tok,a.max_length))
    model=BertVADRegressor(a.pretrain_path,offline=a.offline).to(device)
    obj=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(obj['model'],strict=True)
    mc_pred, mc_unc, targets, deterministic=[] ,[],[],[]
    for tokens,y,_ in loader:
        x={k:v.to(device) for k,v in tokens.items()}
        with torch.inference_mode():
            model.eval()
            baseline=model(**x).cpu().numpy()
            model.train() # dropout MC: identical tokenized text, no semantic augmentation
            draws=torch.stack([model(**x) for _ in range(a.mc_passes)],dim=0)
            mean=draws.mean(0).cpu().numpy()
            std=draws.std(0,unbiased=False).cpu().numpy()
        deterministic.append(baseline); mc_pred.append(mean); mc_unc.append(std); targets.append(y.numpy())
    gold=(np.concatenate(targets,axis=0)*4+1).astype(np.float64)
    pseudo=(np.concatenate(mc_pred,axis=0)*4+1).astype(np.float64)
    deterministic=(np.concatenate(deterministic,axis=0)*4+1).astype(np.float64)
    uncertainty=np.concatenate(mc_unc,axis=0).astype(np.float64)*4
    error=np.abs(pseudo-gold)
    report={'checkpoint':str(checkpoint),'budget':a.budget,'seed':a.seed,
            'mc_passes':a.mc_passes,'views':'same tokens, independent dropout masks',
            'n_dev':len(gold),'test_accessed':False,
            'uses_dev_labels_for_training':False,
            'fixed_error_limit_1to5':a.large_error_limit,
            'interpretation':'diagnostic only; do not train reliability calibrator on dev; use train OOF for fitting',
            'dimensions':{}}
    for k,d in enumerate(DIMS):
        r=evaluate_uncertainty(uncertainty[:,k],error[:,k],a.large_error_limit)
        r['deterministic_e1_mae_1to5']=float(np.abs(deterministic[:,k]-gold[:,k]).mean())
        r['mc_pred_std_across_dev_1to5']=float(pseudo[:,k].std())
        r['gold_std_across_dev_1to5']=float(gold[:,k].std())
        report['dimensions'][d]=r
        x10=r['error_by_uncertainty_coverage'][0]['mean_abs_error_1to5']
        x50=r['error_by_uncertainty_coverage'][2]['mean_abs_error_1to5']
        print(f'{d}: MC_MAE={r["mae_mc_pseudolabel_1to5"]:.4f}, E1_eval_MAE={r["deterministic_e1_mae_1to5"]:.4f}, '
              f'Spearman(unc,error)={r["spearman_std_vs_abs_error"]}, AUC={r["auc_detect_error_gt_fixed_limit"]}, '
              f'low-uncertainty MAE@10%={x10:.4f}, @50%={x50:.4f}',flush=True)
    print('Dev-only reliability diagnostic; no training, no Test access.',flush=True)
    if a.out:
        if a.out.exists():p.error(f'refusing overwrite: {a.out}')
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print('Saved:',a.out,flush=True)

if __name__=='__main__':
    main()
