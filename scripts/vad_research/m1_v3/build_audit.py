#!/usr/bin/env python3
"""TRAIN-only 400/100 audit builder; full-500 checkpoints are forbidden here."""
import argparse, json, random, subprocess, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO))
from semilearn.datasets.nlp_datasets.emobank import GoldData, TextCollator, NORM_COLUMNS, read_gold
from scripts.vad_research.train_roberta_vad_supervised import RobertaVADRegressor, sha256_file
from scripts.vad_research.m1_v3.calibration import normalize_features
from scripts.vad_research.m1_v3.cpu_resources import require_cpu_memory

def partition(ids,seed):
    if len(ids)!=500 or len(set(ids))!=500: raise ValueError('Expected 500 unique TRAIN IDs')
    order=np.random.default_rng(seed).permutation(np.argsort(np.asarray(ids,dtype=str)))
    fit,audit=order[:400],order[400:]
    folds=np.array_split(np.random.default_rng(seed+1).permutation(400),5)
    return fit,audit,folds

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def predict(model,frame,collator,device):
    model.eval(); parts=[]
    loader=DataLoader(GoldData(frame),batch_size=16,shuffle=False,collate_fn=collator)
    with torch.inference_mode():
        for tokens,_,_ in loader:
            tokens={k:v.to(device) for k,v in tokens.items()}
            parts.append(model(**tokens).clamp(0,1).cpu().numpy())
    return np.concatenate(parts)

def supervised_update(model, optimizer, tokens, labels, device):
    """Exactly one P1/P2 supervised control update, shared by audit and benchmark."""
    model.train()
    tokens={k:v.to(device) for k,v in tokens.items()}
    optimizer.zero_grad(set_to_none=True)
    F.l1_loss(model(**tokens),labels.to(device)).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    optimizer.step()

def fit_reference(frame,model_path,collator,device,seed):
    seed_all(seed)
    model=RobertaVADRegressor(model_path,dropout=0.1,offline=True).to(device)
    loader=DataLoader(GoldData(frame),batch_size=16,shuffle=True,collate_fn=collator,
                      generator=torch.Generator().manual_seed(seed+17))
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=0.01)
    print(f'REFERENCE seed={seed} fit_count={len(frame)} P1_epochs=20 P2_steps=160',flush=True)
    for epoch in range(1,21):
        for tokens,labels,_ in loader: supervised_update(model,optimizer,tokens,labels,device)
        print(f'REFERENCE seed={seed} p1_epoch={epoch}/20',flush=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=5e-6,weight_decay=0.01)
    iterator=iter(loader)
    for step in range(1,161):
        try: tokens,labels,_=next(iterator)
        except StopIteration:
            iterator=iter(loader); tokens,labels,_=next(iterator)
        supervised_update(model,optimizer,tokens,labels,device)
        if step%32==0: print(f'REFERENCE seed={seed} p2_step={step}/160',flush=True)
    return model.eval()

def extract_features(frame,model_path,tokenizer,device):
    encoder=AutoModel.from_pretrained(model_path,local_files_only=True).to(device).eval()
    parts=[]
    with torch.inference_mode():
        for start in range(0,len(frame),16):
            texts=frame.iloc[start:start+16]['text'].tolist()
            tokens=tokenizer(texts,padding=True,truncation=True,max_length=128,return_tensors='pt')
            tokens={k:v.to(device) for k,v in tokens.items()}
            h=encoder(**tokens,return_dict=True).last_hidden_state
            m=tokens['attention_mask'].unsqueeze(-1).to(h.dtype)
            parts.append(((h*m).sum(1)/m.sum(1).clamp_min(1)).cpu().numpy())
    del encoder
    if device.type=='cuda': torch.cuda.empty_cache()
    return normalize_features(np.concatenate(parts))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',type=Path,default=REPO.parent/'data/emobank_processed/lb500_s0/train_lb.parquet')
    p.add_argument('--pretrain-path',type=Path,default=REPO.parent/'pretrained/roberta-base')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    a=p.parse_args(); frame=read_gold(a.train,expected_n=500).reset_index(drop=True)
    fit,audit,folds=partition(frame['id'].to_numpy(),a.seed)
    fit_ids=frame.iloc[fit]['id'].tolist(); audit_ids=frame.iloc[audit]['id'].tolist()
    if set(fit_ids)&set(audit_ids): raise AssertionError('Split overlap')
    print(f'TRAIN SHA256={sha256_file(a.train)} fit=400 audit=100 OOF=5x80',flush=True)
    if a.check_only:
        print('PASS: no DEV/TEST read, no output written',flush=True); return
    if a.out.exists(): raise FileExistsError(f'Output exists: {a.out}')
    if a.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    device=torch.device('cuda' if a.device!='cpu' and torch.cuda.is_available() else 'cpu')
    if device.type=='cpu': require_cpu_memory()
    tokenizer=AutoTokenizer.from_pretrained(a.pretrain_path,local_files_only=True,use_fast=True)
    collator=TextCollator(tokenizer,128)
    a.out.mkdir(parents=True)
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=REPO,capture_output=True,text=True,check=True).stdout.strip()
    config={'mode':'full_audit','seed':a.seed,'device':str(device),'git_commit':commit,
      'train_path':str(a.train),'train_sha256':sha256_file(a.train),
      'pretrain_path':str(a.pretrain_path),'fit_count':400,'audit_count':100,
      'oof_folds':5,'p1_epochs':20,'p2_steps':160,'batch_size':16,
      'p1_lr':2e-5,'p2_lr':5e-6,'weight_decay':0.01,'clip_grad_norm':1.0,
      'dropout':0.1,'model':'pretrained RoBERTa + P1 VAD head, refit per fold'}
    (a.out/'run_config.json').write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
    features=extract_features(frame.loc[:,['id','text']],str(a.pretrain_path),tokenizer,device)
    fit_frame=frame.iloc[fit].reset_index(drop=True)
    oof=np.full((400,3),np.nan,dtype=np.float64); provenance=[]
    for index,held in enumerate(folds):
        train_pos=np.setdiff1d(np.arange(400),held)
        train_ids=fit_frame.iloc[train_pos]['id'].tolist()
        held_ids=fit_frame.iloc[held]['id'].tolist()
        if set(train_ids)&set(held_ids): raise AssertionError('OOF overlap')
        model=fit_reference(fit_frame.iloc[train_pos],str(a.pretrain_path),collator,device,a.seed+100+index)
        oof[held]=predict(model,fit_frame.iloc[held],collator,device)
        fold_path=a.out/f'oof_fold_{index}.pth'
        torch.save({'model':model.state_dict(),'train_ids':train_ids,
                    'held_out_ids':held_ids,'seed':a.seed+100+index},fold_path)
        provenance.append({'fold':index,'train_ids':train_ids,'held_out_ids':held_ids,
                           'seed':a.seed+100+index,'checkpoint':fold_path.name,
                           'checkpoint_sha256':sha256_file(fold_path)})
        (a.out/'fold_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
        del model
        if device.type=='cuda': torch.cuda.empty_cache()
        print(f'OOF fold {index+1}/5 complete',flush=True)
    if not np.isfinite(oof).all(): raise RuntimeError('Incomplete OOF')
    model=fit_reference(fit_frame,str(a.pretrain_path),collator,device,a.seed+200)
    reference=predict(model,frame.iloc[audit],collator,device)
    torch.save({'model':model.state_dict(),'train_ids':fit_ids,'seed':a.seed+200},a.out/'reference_400.pth')
    np.savez_compressed(a.out/'audit_inputs.npz',fit_ids=np.asarray(fit_ids),audit_ids=np.asarray(audit_ids),
      fit_features=features[fit],audit_features=features[audit],
      fit_labels=fit_frame.loc[:,NORM_COLUMNS].to_numpy(np.float64),
      audit_labels=frame.iloc[audit].loc[:,NORM_COLUMNS].to_numpy(np.float64),
      oof_predictions=oof,reference_audit=reference)
    weight_path=next((a.pretrain_path/n for n in ('model.safetensors','pytorch_model.bin') if (a.pretrain_path/n).is_file()),None)
    if weight_path is None: raise FileNotFoundError('Pretrained RoBERTa weights missing')
    manifest={'seed':a.seed,'train_sha256':sha256_file(a.train),'git_commit':commit,
      'run_config_sha256':sha256_file(a.out/'run_config.json'),
      'pretrain_weight_sha256':sha256_file(weight_path),
      'pretrain_path':str(a.pretrain_path),'pretrain_config_sha256':sha256_file(a.pretrain_path/'config.json'),
      'reference_train_ids':fit_ids,'audit_ids':audit_ids,'folds':provenance,
      'feature_encoder':'unfitted pretrained RoBERTa, label-blind',
      'training':'P1 fixed 20 epochs + P2 fixed 160 supervised steps; no DEV selection',
      'reference_checkpoint_sha256':sha256_file(a.out/'reference_400.pth')}
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(f'AUDIT_INPUT_DIR={a.out}',flush=True)
if __name__=='__main__': main()
