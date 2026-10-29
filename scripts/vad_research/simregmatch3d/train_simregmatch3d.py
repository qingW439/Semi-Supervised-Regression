#!/usr/bin/env python3
"""SimRegMatch-3D adaptation. Reads TRAIN and DEV only; never reads TEST."""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO))
from semilearn.datasets.nlp_datasets.emobank import GoldData,TextCollator,UnlabeledData
from scripts.emobank.train_emobank_supervised import metric_report
from scripts.vad_research.train_roberta_vad_supervised import RobertaVADRegressor,load_checked_splits,set_seed,sha256_file,git_commit
from scripts.vad_research.train_roberta_vad_p2 import UnlabeledCollator,forever
from scripts.vad_research.simregmatch3d.calibration import mc_statistics,similarity_labels,calibrated_targets
from scripts.vad_research.simregmatch3d.uncertainty_filter import filter_mask,next_threshold,masked_mse
MODES=('supervised_only','consistency_only','calibration_only','filtering_only','simregmatch_full')
CALIBRATION=('calibration_only','simregmatch_full')
FILTERING=('filtering_only','simregmatch_full')
DATA_ROOT=REPO.parent/'data'/'emobank_processed'
MODEL_DIR=REPO.parent/'pretrained'/'roberta-base'

def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',required=True,choices=MODES)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--data-root',type=Path,default=DATA_ROOT)
    p.add_argument('--pretrain-path',type=Path,default=MODEL_DIR)
    p.add_argument('--p1-dir',type=Path,default=None)
    p.add_argument('--out',type=Path,default=None)
    p.add_argument('--steps',type=int,default=160)
    p.add_argument('--eval-every',type=int,default=32)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--ulb-batch-size',type=int,default=64)
    p.add_argument('--eval-batch-size',type=int,default=32)
    p.add_argument('--max-length',type=int,default=128)
    p.add_argument('--mc-passes',type=int,default=5)
    p.add_argument('--beta',type=float,default=0.6)
    p.add_argument('--temperature',type=float,default=1.0)
    p.add_argument('--lambda-u',type=float,default=0.01)
    p.add_argument('--filter-percentile',type=float,default=95.0)
    p.add_argument('--initial-threshold',type=float,default=10.0)
    p.add_argument('--lr',type=float,default=5e-6)
    p.add_argument('--weight-decay',type=float,default=0.01)
    p.add_argument('--clip-grad-norm',type=float,default=1.0)
    p.add_argument('--dropout',type=float,default=0.1)
    p.add_argument('--device',choices=('cuda','cpu','auto'),default='auto')
    p.add_argument('--check-only',action='store_true')
    a=p.parse_args(argv)
    if a.p1_dir is None:a.p1_dir=REPO/'saved_models'/'vad_research'/f'p1_roberta_lb500_s{a.seed}'
    if a.out is None:a.out=REPO/'saved_models'/'vad_research'/f'simregmatch3d_{a.mode}_s{a.seed}_{int(time.time())}'
    if a.seed<0 or any(getattr(a,k)<=0 for k in ('steps','eval_every','batch_size','ulb_batch_size','eval_batch_size','max_length')):p.error('invalid size or seed')
    if a.mc_passes<2 or not 0<=a.beta<=1 or a.temperature<=0 or a.lambda_u<0 or not 0<=a.filter_percentile<=100:p.error('invalid SimRegMatch settings')
    if a.lr<=0 or a.weight_decay<0 or a.clip_grad_norm<0 or not 0<=a.dropout<1 or a.initial_threshold<0:p.error('invalid optimizer/model settings')
    return a

class FeatureRegressor(RobertaVADRegressor):
    """Identical P1 parameter names and prediction path, exposing pooled features."""
    def forward_with_features(self,**inputs):
        states=self.encoder(**inputs,return_dict=True).last_hidden_state
        m=inputs['attention_mask'].unsqueeze(-1).to(dtype=states.dtype)
        pooled=(states*m).sum(dim=1)/m.sum(dim=1).clamp_min(1)
        return self.head(self.dropout(pooled)),pooled
    def forward(self,**inputs):
        pred,_=self.forward_with_features(**inputs)
        return pred

def mode_flags(mode):
    if mode not in MODES:raise ValueError(mode)
    return mode!='supervised_only',mode in CALIBRATION,mode in FILTERING

def make_unlabeled_loss(model,labeled_features,labeled_targets,unlabeled_tokens,mode,args,threshold):
    use_u,use_cal,use_filter=mode_flags(mode)
    if not use_u:raise ValueError('supervised mode has no unlabeled loss')
    preds=[];feats=[]
    with torch.no_grad():
        for _ in range(args.mc_passes):
            pred,feat=model.forward_with_features(**unlabeled_tokens)
            preds.append(pred);feats.append(feat)
    mc=torch.stack(preds)
    model_mean,per_dim_var,uncertainty=mc_statistics(mc)
    mean_features=torch.stack(feats).mean(0)
    if use_cal:
        neighbor,weights=similarity_labels(mean_features,labeled_features,labeled_targets,args.temperature)
        pseudo=calibrated_targets(model_mean,neighbor,args.beta)
        calibration_magnitude=(pseudo-model_mean.detach()).abs().mean()
        similarity_entropy=-(weights*weights.clamp_min(1e-12).log()).sum(1).mean()
    else:
        pseudo=model_mean.detach().clamp(0.0,1.0)
        calibration_magnitude=uncertainty.new_zeros(())
        similarity_entropy=uncertainty.new_zeros(())
    mask=filter_mask(uncertainty,threshold) if use_filter else torch.ones_like(uncertainty,dtype=torch.bool)
    upcoming=next_threshold(uncertainty,args.filter_percentile) if use_filter else threshold
    strong=model(**unlabeled_tokens)
    loss=masked_mse(strong,pseudo,mask)
    stats={'mc_variance':float(per_dim_var.mean().item()),'uncertainty_mean':float(uncertainty.mean().item()),
           'mc_draw_difference':float((mc[0]-mc[1]).abs().mean().item()),'calibration_magnitude':float(calibration_magnitude.item()),
           'similarity_entropy':float(similarity_entropy.item()),'filter_pass_fraction':float(mask.float().mean().item()),
           'filter_pass_count':int(mask.sum().item()),'threshold_used':float(threshold),'threshold_next':float(upcoming)}
    return loss,stats,upcoming

def evaluate(model,loader,device):
    model.eval();gold=[];pred=[]
    with torch.inference_mode():
        for tok,y,_ in loader:
            tok={k:v.to(device) for k,v in tok.items()}
            z=model(**tok)
            if z.shape!=y.shape or not torch.isfinite(z).all():raise RuntimeError('Invalid DEV prediction')
            gold.append(y.numpy());pred.append(z.cpu().numpy())
    return metric_report(np.concatenate(gold),np.concatenate(pred))

def check_inputs(a):
    set_seed(a.seed)
    lb,dev,ulb,paths=load_checked_splits(argparse.Namespace(data_root=a.data_root,budget=500,seed=a.seed))
    if (len(lb),len(ulb),len(dev))!=(500,7562,1000):raise ValueError('Incorrect split counts')
    ckpt=a.p1_dir/'model_best.pth';config_path=a.p1_dir/'run_config.json'
    if not ckpt.is_file() or not config_path.is_file() or not a.pretrain_path.is_dir():raise FileNotFoundError('P1 checkpoint/config or pretrained model missing')
    prior=json.loads(config_path.read_text())
    if int(prior['seed'])!=a.seed or int(prior['budget'])!=500 or int(prior['max_length'])!=a.max_length:raise ValueError('P1 seed/budget/max-length mismatch')
    if abs(float(prior['dropout'])-a.dropout)>1e-12:raise ValueError('P1 dropout mismatch')
    if Path(prior['model']).name!=a.pretrain_path.name:raise ValueError('P1 pretrained model family mismatch')
    # P1 config can contain the original host path; verify relocated model bytes.
    expected_pretrain='5bde1d28afb363d0103324efeb5afc8b2b397fe5e04beabb9b1ef355255ade81'
    if sha256_file(a.pretrain_path/'model.safetensors')!=expected_pretrain:raise ValueError('Pretrained model SHA256 mismatch')
    hashes={k:sha256_file(v) for k,v in paths.items()}
    if hashes!=prior['split_sha256']:raise ValueError('P1 and experiment split hashes differ')
    return lb,dev,ulb,ckpt,hashes

def run(a):
    lb,dev,ulb,ckpt_path,hashes=check_inputs(a)
    if a.check_only:
        print('CHECK_ONLY PASS: TRAIN 500/7562, DEV 1000, P1 and split SHA256 verified; no TEST; no output',flush=True)
        return
    out=a.out.expanduser().resolve()
    if out.exists():raise FileExistsError(f'Output already exists: {out}')
    if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    device=torch.device('cpu' if a.device=='cpu' or not torch.cuda.is_available() else 'cuda')
    model_name=str(a.pretrain_path.resolve())
    tokenizer=AutoTokenizer.from_pretrained(model_name,local_files_only=True,use_fast=True)
    lb_dl=DataLoader(GoldData(lb),batch_size=a.batch_size,shuffle=True,collate_fn=TextCollator(tokenizer,a.max_length),generator=torch.Generator().manual_seed(17000+a.seed))
    ulb_dl=DataLoader(UnlabeledData(ulb),batch_size=a.ulb_batch_size,shuffle=True,collate_fn=UnlabeledCollator(tokenizer,a.max_length),generator=torch.Generator().manual_seed(19000+a.seed))
    dev_dl=DataLoader(GoldData(dev),batch_size=a.eval_batch_size,shuffle=False,collate_fn=TextCollator(tokenizer,a.max_length))
    model=FeatureRegressor(model_name,dropout=a.dropout,offline=True).to(device)
    state=torch.load(ckpt_path,map_location='cpu',weights_only=False)
    model.load_state_dict(state['model'],strict=True);del state
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    config={k:(str(v) if isinstance(v,Path) else v) for k,v in vars(a).items()}
    config.update({'git_commit':git_commit(),'p1_checkpoint_sha256':sha256_file(ckpt_path),'pretrained_model_sha256':'5bde1d28afb363d0103324efeb5afc8b2b397fe5e04beabb9b1ef355255ade81','split_sha256':hashes,'device_resolved':str(device),
                   'supervised_loss':'mean L1 on normalized VAD (P2 protocol)',
                   'unlabeled_loss':'sum-dimension MSE / retained samples (official SimRegMatch form)',
                   'filtering':'previous batch threshold; current batch quantile sets next threshold',
                   'target_range':'clamped [0,1] for pseudo targets only','mc_rng':'forked per step; restores supervised RNG state'})
    out.mkdir(parents=True)
    (out/'run_config.json').write_text(json.dumps(config,indent=2,ensure_ascii=False)+'\n')
    history=[];dev_history=[];best=float('inf');best_step=-1
    labeled_iter=forever(lb_dl);unlabeled_iter=forever(ulb_dl)
    threshold=float(a.initial_threshold);visited=set();start=time.monotonic()
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    def eval_save(step):
        nonlocal best,best_step
        metrics=evaluate(model,dev_dl,device)
        record={'step':step,'elapsed_sec':time.monotonic()-start,'metrics':metrics}
        dev_history.append(record)
        macro=metrics['macro_mae_original_scale']
        print(f'{a.mode} step={step} DEV Macro-MAE={macro:.6f}',flush=True)
        if macro<best-1e-8:
            best=macro;best_step=step
            torch.save({'model':model.state_dict(),'step':step,'dev_metrics':metrics,'run_config':config},out/'model_best.pth')
            (out/'dev_metrics_best.json').write_text(json.dumps(record,indent=2)+'\n')
    eval_save(0)
    for step in range(1,a.steps+1):
        tic=time.monotonic();model.train()
        tok,y,_=next(labeled_iter)
        tok={k:v.to(device) for k,v in tok.items()};y=y.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred,features=model.forward_with_features(**tok)
        supervised=F.l1_loss(pred,y);total=supervised
        row={'step':step,'supervised_loss':float(supervised.detach().item()),'unlabeled_loss':0.0,
             'effective_unlabeled_grad_norm':0.0,'mc_variance':0.0,'uncertainty_mean':0.0,
             'mc_draw_difference':0.0,'calibration_magnitude':0.0,'similarity_entropy':0.0,
             'filter_pass_fraction':0.0,'filter_pass_count':0,'threshold_used':threshold,'threshold_next':threshold}
        if a.mode!='supervised_only':
            u_tok,u_ids=next(unlabeled_iter);visited.update(u_ids)
            u_tok={k:v.to(device) for k,v in u_tok.items()}
            devices=[torch.cuda.current_device()] if device.type=='cuda' else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(1_000_000+a.seed*10000+step)
                if device.type=='cuda':torch.cuda.manual_seed_all(1_000_000+a.seed*10000+step)
                u_loss,stats,threshold=make_unlabeled_loss(model,features,y,u_tok,a.mode,a,row['threshold_used'])
            row.update(stats);row['unlabeled_loss']=float(u_loss.detach().item())
            if row['filter_pass_count']:
                grads=torch.autograd.grad(a.lambda_u*u_loss,tuple(model.head.parameters()),retain_graph=True,allow_unused=True)
                row['effective_unlabeled_grad_norm']=math.sqrt(sum(float(g.detach().square().sum().item()) for g in grads if g is not None))
            total=total+a.lambda_u*u_loss
        if not torch.isfinite(total):raise RuntimeError('Nonfinite total loss')
        total.backward()
        grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),a.clip_grad_norm) if a.clip_grad_norm>0 else torch.tensor(0.)
        if not torch.isfinite(torch.as_tensor(grad_norm)):raise RuntimeError('Nonfinite gradients')
        optimizer.step()
        row['total_loss']=float(total.detach().item());row['grad_norm_before_clip']=float(grad_norm)
        row['unique_unlabeled_ids']=len(visited);row['step_seconds']=time.monotonic()-tic
        row['peak_gpu_bytes']=int(torch.cuda.max_memory_allocated(device)) if device.type=='cuda' else 0
        history.append(row)
        if step%a.eval_every==0 or step==a.steps:eval_save(step)
    torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':a.steps,'run_config':config},out/'model_last.pth')
    pd.DataFrame(history).to_csv(out/'training_history.csv',index=False)
    (out/'dev_metrics_history.json').write_text(json.dumps(dev_history,indent=2)+'\n')
    summary={'mode':a.mode,'seed':a.seed,'best_step':best_step,'best_macro_mae':best,'last_step':a.steps,
             'last_metrics':dev_history[-1]['metrics'],'duration_sec':time.monotonic()-start,
             'unique_unlabeled_ids':len(visited),
             'mean_filter_pass_fraction':float(np.mean([r['filter_pass_fraction'] for r in history])),
             'mean_effective_unlabeled_grad_norm':float(np.mean([r['effective_unlabeled_grad_norm'] for r in history])),
             'peak_gpu_bytes':int(torch.cuda.max_memory_allocated(device)) if device.type=='cuda' else 0,
             'git_commit':config['git_commit']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('SUMMARY',json.dumps(summary),flush=True);print('OUTPUT',out,flush=True)

if __name__=='__main__':run(parse_args())
