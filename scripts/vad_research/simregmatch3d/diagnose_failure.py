#!/usr/bin/env python3
"""Read-only, fixed-batch post-hoc diagnosis. No optimizer, update, or TEST read."""
from __future__ import annotations
import argparse,hashlib,json,math,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO))
from semilearn.datasets.nlp_datasets.emobank import GoldData,TextCollator,UnlabeledData
from scripts.emobank.train_emobank_supervised import metric_report
from scripts.vad_research.train_roberta_vad_supervised import load_checked_splits,sha256_file
from scripts.vad_research.train_roberta_vad_p2 import UnlabeledCollator
from scripts.vad_research.simregmatch3d.train_simregmatch3d import FeatureRegressor
from scripts.vad_research.simregmatch3d.calibration import mc_statistics,similarity_labels,calibrated_targets
from scripts.vad_research.simregmatch3d.uncertainty_filter import filter_mask,next_threshold,masked_mse

FIXED_SEED=20261028
N_BATCHES=8

def summaries(values):
    a=np.asarray(values,dtype=np.float64)
    return {'mean':float(a.mean()),'std':float(a.std()),'p10':float(np.quantile(a,.1)),
            'p50':float(np.quantile(a,.5)),'p90':float(np.quantile(a,.9)),
            'min':float(a.min()),'max':float(a.max())}

def tensor_flat(grads):
    return torch.cat([g.detach().reshape(-1) for g in grads])

def make_model(pretrain,checkpoint,device):
    model=FeatureRegressor(str(pretrain),dropout=.1,offline=True).to(device)
    obj=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(obj['model'],strict=True)
    return model,obj['step']

@torch.inference_mode()
def dev_stats(model,loader,device):
    model.eval();gold=[];pred=[]
    for token,y,_ in loader:
        z=model(**{k:v.to(device) for k,v in token.items()})
        gold.append(y.numpy());pred.append(z.cpu().numpy())
    y=np.concatenate(gold);p=np.concatenate(pred)
    metrics=metric_report(y,p)
    stats={d:{'gold_std':float(y[:,i].std()*4),'pred_std':float(p[:,i].std()*4),
              'pred_gold_std_ratio':float(p[:,i].std()/y[:,i].std()),
              'mae':metrics[d]['mae'],'ccc':metrics[d]['ccc']} for i,d in enumerate('VAD')}
    return {'step':None,'macro_mae':metrics['macro_mae_original_scale'],'dims':stats}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--pretrain-path',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    device=torch.device(a.device)
    lb,dev,ulb,paths=load_checked_splits(argparse.Namespace(data_root=a.data_root,budget=500,seed=0))
    assert (len(lb),len(ulb),len(dev))==(500,7562,1000)
    expected=json.loads((a.run_root/'simregmatch_full'/'run_config.json').read_text())
    assert {k:sha256_file(v) for k,v in paths.items()}==expected['split_sha256']
    tokenizer=AutoTokenizer.from_pretrained(str(a.pretrain_path),local_files_only=True,use_fast=True)
    rng=np.random.default_rng(FIXED_SEED)
    lb_indices=rng.permutation(len(lb))[:N_BATCHES*16]
    ulb_indices=rng.permutation(len(ulb))[:N_BATCHES*64]
    lb_fixed=lb.iloc[lb_indices].reset_index(drop=True)
    ulb_fixed=ulb.iloc[ulb_indices].reset_index(drop=True)
    lb_loader=DataLoader(GoldData(lb_fixed),batch_size=16,shuffle=False,collate_fn=TextCollator(tokenizer,128))
    ulb_loader=DataLoader(UnlabeledData(ulb_fixed),batch_size=64,shuffle=False,collate_fn=UnlabeledCollator(tokenizer,128))
    dev_loader=DataLoader(GoldData(dev),batch_size=32,shuffle=False,collate_fn=TextCollator(tokenizer,128))
    ckpts={arm:{part:a.run_root/arm/f'model_{part}.pth' for part in ('best','last')} for arm in ('supervised_only','simregmatch_full')}
    ckpt_hashes={arm:{part:sha256_file(path) for part,path in parts.items()} for arm,parts in ckpts.items()}
    result={'fixed_seed':FIXED_SEED,'n_batches':N_BATCHES,'labeled_ids_sha256':hashlib.sha256('\n'.join(map(str,lb_fixed.id)).encode()).hexdigest(),
            'unlabeled_ids_sha256':hashlib.sha256('\n'.join(map(str,ulb_fixed.id)).encode()).hexdigest(),
            'labeled_ids':list(map(str,lb_fixed.id)),'unlabeled_ids':list(map(str,ulb_fixed.id)),
            'data_sha256':expected['split_sha256'],'checkpoint_sha256':ckpt_hashes,
            'dev':{},'mechanism':{},'train_mc_error_exploratory':{}}
    for arm,parts in ckpts.items():
        result['dev'][arm]={}
        for part,path in parts.items():
            model,step=make_model(a.pretrain_path,path,device)
            v=dev_stats(model,dev_loader,device);v['step']=step
            result['dev'][arm][part]=v
            del model
    model,_=make_model(a.pretrain_path,ckpts['simregmatch_full']['last'],device)
    model.train()
    head_before=[param.detach().cpu().clone() for param in model.head.parameters()]
    head=tuple(model.head.parameters())
    threshold=10.0
    records=[];all_model=[];all_cal=[];train_var=[];train_err=[]
    for idx,((l_tok,l_y,l_ids),(u_tok,u_ids)) in enumerate(zip(lb_loader,ulb_loader)):
        l_tok={k:v.to(device) for k,v in l_tok.items()}
        u_tok={k:v.to(device) for k,v in u_tok.items()}
        l_y=l_y.to(device)
        model.train()
        model.zero_grad(set_to_none=True)
        supervised_pred,l_features=model.forward_with_features(**l_tok)
        supervised=F.l1_loss(supervised_pred,l_y)
        devices=[torch.cuda.current_device()] if device.type=='cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(FIXED_SEED+1000+idx)
            if device.type=='cuda':torch.cuda.manual_seed_all(FIXED_SEED+1000+idx)
            with torch.no_grad():
                draws=[];features=[]
                for _ in range(5):
                    z,f=model.forward_with_features(**u_tok)
                    draws.append(z);features.append(f)
            mc=torch.stack(draws)
            mean,var,uncertainty=mc_statistics(mc)
            neighbors,weights=similarity_labels(torch.stack(features).mean(0),l_features,l_y,1.0)
            calibrated=calibrated_targets(mean,neighbors,.6)
            mask=filter_mask(uncertainty,threshold)
            next_value=next_threshold(uncertainty,95.0)
            strong=model(**u_tok)
            unlabeled=masked_mse(strong,calibrated,mask)
            sup_grad=tensor_flat(torch.autograd.grad(supervised,head,retain_graph=True))
            unsup_grad=tensor_flat(torch.autograd.grad(unlabeled,head))
        sup_norm=float(sup_grad.norm().item())
        raw_u_norm=float(unsup_grad.norm().item())
        scaled_u_norm=.01*raw_u_norm
        cosine=float(F.cosine_similarity(sup_grad[None],unsup_grad[None]).item())
        entropy=-(weights*weights.clamp_min(1e-12).log()).sum(-1)
        effective=entropy.exp()
        inv_simpson=1/weights.square().sum(-1)
        l_mean=l_y.mean(0)
        label_std=l_y.std(0,unbiased=False)
        model_dist=(mean-l_mean).norm(dim=1)
        cal_dist=(calibrated-l_mean).norm(dim=1)
        neighbor_dist=(neighbors-l_mean).norm(dim=1)
        model_var=mean.var(dim=0,unbiased=False)
        cal_var=calibrated.var(dim=0,unbiased=False)
        records.append({'batch':idx,'labeled_ids':list(map(str,l_ids)),'unlabeled_ids':list(map(str,u_ids)),
                        'entropy':entropy.cpu().tolist(),'effective_neighbors':effective.cpu().tolist(),
                        'inverse_simpson_neighbors':inv_simpson.cpu().tolist(),
                        'max_neighbor_weight':weights.max(1).values.cpu().tolist(),
                        'neighbor_to_label_mean_l2':neighbor_dist.cpu().tolist(),
                        'model_to_label_mean_l2':model_dist.cpu().tolist(),
                        'calibrated_to_label_mean_l2':cal_dist.cpu().tolist(),
                        'label_batch_std_vad':label_std.cpu().tolist(),
                        'mean_neighbor_vad':neighbors.mean(0).cpu().tolist(),
                        'label_batch_mean_vad':l_mean.cpu().tolist(),
                        'model_var_vad':model_var.cpu().tolist(),'calibrated_var_vad':cal_var.cpu().tolist(),
                        'calibration_closer_to_label_mean_fraction':float((cal_dist<model_dist).float().mean().item()),
                        'retained_fraction':float(mask.float().mean().item()),
                        'mc_uncertainty_mean':float(uncertainty.mean().item()),
                        'threshold_used':threshold,'threshold_next':next_value,
                        'supervised_head_grad_norm':sup_norm,'unlabeled_head_grad_norm_raw':raw_u_norm,
                        'unlabeled_head_grad_norm_scaled':scaled_u_norm,
                        'unlabeled_supervised_grad_ratio':scaled_u_norm/max(sup_norm,1e-12),
                        'head_gradient_cosine':cosine,
                        'supervised_loss':float(supervised.detach().item()),
                        'unlabeled_loss':float(unlabeled.detach().item())})
        threshold=next_value
        all_model.append(mean.detach().cpu().numpy())
        all_cal.append(calibrated.detach().cpu().numpy())
        # These labels trained the checkpoint: exploratory in-sample correlation only.
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(FIXED_SEED+2000+idx)
            if device.type=='cuda':torch.cuda.manual_seed_all(FIXED_SEED+2000+idx)
            with torch.no_grad():
                label_draws=torch.stack([model(**l_tok) for _ in range(5)])
                label_mean,_,label_uncertainty=mc_statistics(label_draws)
        train_var.extend(label_uncertainty.cpu().tolist())
        train_err.extend((label_mean-l_y).abs().mean(1).cpu().tolist())
        print('batch',idx+1,'sup_grad',round(sup_norm,5),'ulb_grad_scaled',round(scaled_u_norm,7),'cos',round(cosine,4),flush=True)
    assert all(torch.equal(before,now.detach().cpu()) for before,now in zip(head_before,model.head.parameters()))
    all_model=np.concatenate(all_model)
    all_cal=np.concatenate(all_cal)
    var=np.asarray(train_var)
    err=np.asarray(train_err)
    q25,q75=np.quantile(var,[.25,.75])
    def flat(key):return [value for row in records for value in row[key]]
    result['mechanism']={
        'entropy':summaries(flat('entropy')),
        'entropy_over_log16':summaries(np.asarray(flat('entropy'))/math.log(16)),
        'effective_neighbors_exp_entropy':summaries(flat('effective_neighbors')),
        'effective_neighbors_inverse_simpson':summaries(flat('inverse_simpson_neighbors')),
        'max_neighbor_weight':summaries(flat('max_neighbor_weight')),
        'neighbor_to_label_mean_l2':summaries(flat('neighbor_to_label_mean_l2')),
        'model_to_label_mean_l2':summaries(flat('model_to_label_mean_l2')),
        'calibrated_to_label_mean_l2':summaries(flat('calibrated_to_label_mean_l2')),
        'calibration_closer_to_label_mean_fraction':summaries([r['calibration_closer_to_label_mean_fraction'] for r in records]),
        'supervised_head_grad_norm':summaries([r['supervised_head_grad_norm'] for r in records]),
        'unlabeled_head_grad_norm_raw':summaries([r['unlabeled_head_grad_norm_raw'] for r in records]),
        'unlabeled_head_grad_norm_scaled':summaries([r['unlabeled_head_grad_norm_scaled'] for r in records]),
        'unlabeled_supervised_grad_ratio':summaries([r['unlabeled_supervised_grad_ratio'] for r in records]),
        'head_gradient_cosine':summaries([r['head_gradient_cosine'] for r in records]),
        'head_gradient_negative_cosine_fraction':float(np.mean([r['head_gradient_cosine']<0 for r in records])),
        'retained_fraction':summaries([r['retained_fraction'] for r in records]),
        'model_pseudo_var_vad_pooled':all_model.var(axis=0).tolist(),
        'calibrated_pseudo_var_vad_pooled':all_cal.var(axis=0).tolist(),
        'pooled_calibrated_over_model_variance_vad':(all_cal.var(axis=0)/all_model.var(axis=0)).tolist(),
        'mean_within_batch_calibrated_over_model_variance_vad':np.mean([np.asarray(r['calibrated_var_vad'])/np.asarray(r['model_var_vad']) for r in records],axis=0).tolist(),
        'head_parameters_unchanged':True,
        'per_batch':records,
    }
    result['train_mc_error_exploratory']={
        'n_in_sample_labeled':len(var),
        'spearman_uncertainty_vs_mean_absolute_error':float(spearmanr(var,err).statistic),
        'uncertainty':summaries(var),
        'normalized_absolute_error':summaries(err),
        'low_uncertainty_q25_error_mean':float(err[var<=q25].mean()),
        'high_uncertainty_q75_error_mean':float(err[var>=q75].mean()),
        'warning':'These 128 TRAIN labels contributed to fitting the checkpoint. This is in-sample exploration, not unbiased pseudo-label quality evidence.'
    }
    a.out.mkdir(parents=True)
    (a.out/'diagnostic.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print('DIAGNOSTIC_JSON',a.out/'diagnostic.json')
    print('ENTROPY',result['mechanism']['entropy'])
    print('EFFECTIVE_NEIGHBORS',result['mechanism']['effective_neighbors_exp_entropy'])
    print('GRAD_RATIO',result['mechanism']['unlabeled_supervised_grad_ratio'])
    print('GRAD_COSINE',result['mechanism']['head_gradient_cosine'])
    print('VARIANCE_RATIO',result['mechanism']['pooled_calibrated_over_model_variance_vad'])
    print('EXPLORATORY_MC_ERROR',result['train_mc_error_exploratory']['spearman_uncertainty_vs_mean_absolute_error'])

if __name__=='__main__':main()
