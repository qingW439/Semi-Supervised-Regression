#!/usr/bin/env python3
"""B0/B1/B2 on frozen pretrained RoBERTa features; TRAIN-only CV, DEV evaluation.

B1/B2 use the same Nyström approximation. B1 fits its basis on labeled TRAIN
covariates; B2 adds unlabeled TRAIN covariates. Neither fits a basis on DEV.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from scripts.emobank.train_emobank_supervised import metric_report
from scripts.vad_research.m0.spectral_core import fit_nystrom_spectral
from semilearn.datasets.nlp_datasets.emobank import DIMENSIONS

ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budget', type=int, choices=(500,), default=500)
    parser.add_argument('--seed', type=int, choices=(0, 1, 2), default=0)
    parser.add_argument('--feature-dir', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--rank', type=int, default=64)
    parser.add_argument('--landmarks', type=int, default=256)
    parser.add_argument('--gamma', type=float, default=1.0)
    args = parser.parse_args()
    if args.feature_dir is None:
        args.feature_dir = REPO/'saved_models'/'vad_research'/'m0'/f'pretrained_features_lb{args.budget}_s{args.seed}'
    if args.out is None:
        args.out = REPO/'saved_models'/'vad_research'/'m0'/f'spectral_lb{args.budget}_s{args.seed}'
    if not (1<=args.rank<=args.landmarks<=args.budget) or args.gamma<=0:
        parser.error('invalid rank, landmark count, or gamma')
    return args


def checked_features(feature_dir, budget, seed):
    metadata = json.loads((feature_dir/'feature_metadata.json').read_text(encoding='utf-8'))
    if metadata['budget']!=budget or metadata['seed']!=seed:
        raise ValueError('feature budget/seed mismatch')
    if not metadata['model'].startswith('frozen pretrained RoBERTa-base'):
        raise ValueError('feature source is not the frozen pretrained encoder')
    data = np.load(feature_dir/'pretrained_features.npz',allow_pickle=False)
    arrays = {key:data[key] for key in ('lb_x','ulb_x','dev_x','lb_y','dev_y',
                                         'lb_ids','ulb_ids','dev_ids')}
    counts = (len(arrays['lb_x']),len(arrays['ulb_x']),len(arrays['dev_x']))
    if counts!=(budget,8062-budget,1000):
        raise ValueError(f'unexpected feature counts: {counts}')
    if any(len(set(arrays[f'{name}_ids'].tolist()))!=counts[index]
           for index,name in enumerate(('lb','ulb','dev'))):
        raise ValueError('duplicate IDs')
    if set(arrays['lb_ids']) & set(arrays['ulb_ids']) or set(arrays['dev_ids']) & (set(arrays['lb_ids'])|set(arrays['ulb_ids'])):
        raise ValueError('TRAIN/DEV ID overlap')
    if arrays['lb_y'].shape!=(budget,3) or arrays['dev_y'].shape!=(1000,3):
        raise ValueError('target shape mismatch')
    width=arrays['lb_x'].shape[1]
    if any(arrays[f'{name}_x'].ndim!=2 or arrays[f'{name}_x'].shape[1]!=width
           for name in ('lb','ulb','dev')):
        raise ValueError('feature dimension mismatch')
    if any(not np.isfinite(arrays[key]).all() for key in ('lb_x','ulb_x','dev_x','lb_y','dev_y')):
        raise ValueError('nonfinite feature or target')
    return arrays, metadata


def spectral_pair(mode, x_train, x_valid, x_unlabeled, *, gamma, rank, landmarks, seed):
    if mode=='b0':
        return x_train, x_valid
    basis_covariates=x_train if mode=='b1' else np.concatenate((x_train,x_unlabeled))
    basis=fit_nystrom_spectral(basis_covariates,gamma=gamma,rank=rank,
                               landmarks=landmarks,seed=seed)
    return basis.transform(x_train),basis.transform(x_valid)


def select_alpha(mode, x_train, y_train, x_unlabeled, *, gamma, rank, landmarks, seed):
    folds=KFold(n_splits=5,shuffle=True,random_state=seed)
    fold_features=[]
    for fold,(fit_idx,val_idx) in enumerate(folds.split(x_train)):
        fit_x,val_x=spectral_pair(mode,x_train[fit_idx],x_train[val_idx],x_unlabeled,
                                  gamma=gamma,rank=rank,landmarks=landmarks,seed=seed+fold)
        fold_features.append((fit_x,val_x,y_train[fit_idx],y_train[val_idx]))
    scores={}
    for alpha in ALPHAS:
        values=[]
        for fit_x,val_x,fit_y,val_y in fold_features:
            model=make_pipeline(StandardScaler(),Ridge(alpha=alpha))
            model.fit(fit_x,fit_y)
            values.append(4*mean_absolute_error(val_y,model.predict(val_x)))
        scores[str(alpha)]=float(np.mean(values))
    selected=min(ALPHAS,key=lambda alpha:scores[str(alpha)])
    return selected,scores


def main():
    args=parse_args()
    arrays,metadata=checked_features(args.feature_dir,args.budget,args.seed)
    print(json.dumps({'counts':metadata['counts'],'pretrain_sha256':metadata['pretrain_sha256'],
                      'split_sha256':metadata['split_sha256'],'output':str(args.out)},sort_keys=True))
    if args.check_only:
        print('PASS: frozen pretrained features and IDs verified; no fitting or output writes')
        return
    if args.out.exists():
        raise FileExistsError(f'refusing to overwrite output: {args.out}')
    x_train=normalize(arrays['lb_x'].astype(np.float64))
    x_unlabeled=normalize(arrays['ulb_x'].astype(np.float64))
    x_dev=normalize(arrays['dev_x'].astype(np.float64))
    y_train=arrays['lb_y'].astype(np.float64)
    y_dev=arrays['dev_y'].astype(np.float64)
    results={}
    predictions={'id':arrays['dev_ids']}
    for index,dim in enumerate(DIMENSIONS):
        predictions[f'gold_{dim}']=y_dev[:,index]*4+1
    for mode in ('b0','b1','b2'):
        alpha,cv_scores=select_alpha(mode,x_train,y_train,x_unlabeled,
                                     gamma=args.gamma,rank=args.rank,
                                     landmarks=args.landmarks,seed=args.seed)
        fit_x,dev_x=spectral_pair(mode,x_train,x_dev,x_unlabeled,
                                  gamma=args.gamma,rank=args.rank,
                                  landmarks=args.landmarks,seed=args.seed)
        model=make_pipeline(StandardScaler(),Ridge(alpha=alpha))
        model.fit(fit_x,y_train)
        pred=model.predict(dev_x)
        metrics=metric_report(y_dev,pred)
        results[mode]={'alpha':alpha,'cv_mae_original_scale':cv_scores,
                       'dev_metrics':metrics,'feature_dim':int(fit_x.shape[1])}
        for index,dim in enumerate(DIMENSIONS):
            predictions[f'{mode}_pred_{dim}']=pred[:,index]*4+1
        print(f'{mode}: alpha={alpha} Dev Macro-MAE={metrics["macro_mae_original_scale"]:.6f}',flush=True)
    args.out.mkdir(parents=True,exist_ok=False)
    manifest={'protocol':'pretrained frozen RoBERTa; L2 normalize; train-only 5-fold CV',
              'spectral_approximation':'Nyström, not full kernel eigendecomposition',
              'rank':args.rank,'landmarks':args.landmarks,'gamma':args.gamma,
              'seed':args.seed,'alphas':ALPHAS,'feature_metadata':metadata,
              'results':results}
    (args.out/'comparison.json').write_text(json.dumps(manifest,indent=2)+'\n')
    pd.DataFrame(predictions).to_csv(args.out/'dev_predictions.csv',index=False)


if __name__=='__main__':
    main()
