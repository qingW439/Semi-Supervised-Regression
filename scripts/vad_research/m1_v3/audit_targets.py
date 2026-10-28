#!/usr/bin/env python3
"""One-shot held-out TRAIN audit; never tune against these 100 labels."""
import argparse, json, sys
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO))
from scripts.vad_research.m1_v3.build_audit import partition
from scripts.vad_research.m1_v3.calibration import NeighborLibrary, MODES, calibrate, check_targets
from scripts.vad_research.train_roberta_vad_supervised import sha256_file

def verify_provenance(data,manifest):
    fit=list(map(str,data['fit_ids'])); audit=list(map(str,data['audit_ids']))
    if len(fit)!=400 or len(audit)!=100 or len(set(fit+audit))!=500:
        raise ValueError('Invalid 400/100 ID partition')
    ids=fit+audit; fit_pos,audit_pos,folds=partition(ids,int(manifest['seed']))
    if [ids[i] for i in fit_pos]!=fit or [ids[i] for i in audit_pos]!=audit:
        raise ValueError('Split differs from predetermined seed')
    if manifest['reference_train_ids']!=fit or manifest['audit_ids']!=audit:
        raise ValueError('Reference or audit ID provenance mismatch')
    if manifest['feature_encoder']!='unfitted pretrained RoBERTa, label-blind':
        raise ValueError('Feature provenance mismatch')
    if len(manifest['folds'])!=5: raise ValueError('Expected five OOF folds')
    for j,held in enumerate(folds):
        record=manifest['folds'][j]
        held_ids=[fit[i] for i in held]
        train_ids=[item for item in fit if item not in set(held_ids)]
        if record['held_out_ids']!=held_ids or record['train_ids']!=train_ids:
            raise ValueError(f'Fold {j} contains leakage or missing IDs')
    if check_targets(data['fit_labels'],'fit labels').shape!=(400,3): raise ValueError('Bad fit labels')
    if check_targets(data['audit_labels'],'audit labels').shape!=(100,3): raise ValueError('Bad audit labels')
    check_targets(data['oof_predictions'],'OOF'); check_targets(data['reference_audit'],'reference')
    return fit,audit

def metrics(y,base,target,correction):
    error=np.abs(target-y); base_error=np.abs(base-y)
    truth_direction=np.sign(y-base); applied_direction=np.sign(correction)
    changed=np.abs(correction)>1e-12
    direction=np.where(changed,applied_direction==truth_direction,np.nan)
    return {'mae':error.mean(axis=0).tolist(), 'macro_mae':float(error.mean()),
      'gain_vs_reference':float(base_error.mean()-error.mean()),
      'direction_accuracy':[float(np.mean(direction[changed[:,j],j])) if changed[:,j].any() else None for j in range(3)],
      'mean_abs_correction':np.abs(correction).mean(axis=0).tolist(),
      'prediction_variance':target.var(axis=0).tolist(),
      'changed_fraction':changed.mean(axis=0).tolist()}

def paired_lower_bound(y,base,target,seed):
    # Three candidates, two-sided Bonferroni familywise 95%: lower 0.8333 percentile.
    per_item=np.abs(base-y).mean(1)-np.abs(target-y).mean(1)
    rng=np.random.default_rng(seed+9001)
    indices=rng.integers(0,len(y),size=(10000,len(y)))
    return float(np.quantile(per_item[indices].mean(1),0.05/(2*3)))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--train',type=Path,default=REPO.parent/'data/emobank_processed/lb500_s0/train_lb.parquet')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists(): raise FileExistsError(f'Output exists: {a.out}')
    manifest=json.loads((a.inputs/'manifest.json').read_text(encoding='utf-8'))
    if sha256_file(a.train)!=manifest['train_sha256']:
        raise ValueError('TRAIN split hash mismatch')
    if sha256_file(a.inputs/'reference_400.pth')!=manifest['reference_checkpoint_sha256']:
        raise ValueError('Reference checkpoint hash mismatch')
    with np.load(a.inputs/'audit_inputs.npz',allow_pickle=False) as data:
        fit_ids,audit_ids=verify_provenance(data,manifest)
        library=NeighborLibrary(tuple(fit_ids),data['fit_features'],data['fit_labels'],data['oof_predictions'])
        y=data['audit_labels'].copy(); base=data['reference_audit'].copy()
        query=data['audit_features'].copy()
    results={}; arrays={}
    for mode in MODES:
        target,details=calibrate(base,query,audit_ids,library,mode)
        correction=details['correction']
        row=metrics(y,base,target,correction)
        if mode!='reference_target':
            row['bonferroni_bootstrap_gain_lower']=paired_lower_bound(y,base,target,int(manifest['seed']))
            row['passes_precommitted_gate']=bool(row['bonferroni_bootstrap_gain_lower']>0)
        results[mode]=row; arrays[mode]=target
    go=any(results[m].get('passes_precommitted_gate',False) for m in MODES)
    report={'decision':'GO' if go else 'NO-GO','protocol':'one fixed 400/100 audit; no search',
            'units':'normalized [0,1] VAD','dimensions':['V','A','D'],
            'precommitted_gate':'paired 10000-bootstrap gain lower bound > 0, Bonferroni 3 arms, two-sided familywise 95%',
            'results':results,'input_manifest_sha256':sha256_file(a.inputs/'manifest.json')}
    a.out.mkdir(parents=True)
    (a.out/'audit_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    np.savez_compressed(a.out/'audit_predictions.npz',ids=np.asarray(audit_ids),gold=y,
                        reference=base,**arrays)
    print(json.dumps(report,indent=2,allow_nan=False))
if __name__=='__main__': main()
