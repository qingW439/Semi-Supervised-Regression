#!/usr/bin/env python3
"""Debug U0 training on one fold: ridge vs linear SGD vs sigmoid/Huber."""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from semilearn.datasets.nlp_datasets.emobank import NORM_COLUMNS, read_gold  # noqa
from scripts.emobank.train_emobank_supervised import metric_report  # noqa
from scripts.vad_research.train_acvr import encode  # noqa

DIMS=("V","A","D")
def sm(y,p):
    m=metric_report(y,p); return m["macro_mae_original_scale"], float(np.mean([m[d]["ccc"] for d in DIMS]))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data-root",type=Path); ap.add_argument("--manifest",type=Path); ap.add_argument("--pretrain-path",type=Path); ap.add_argument("--device",default="cuda")
    a=ap.parse_args(); dev=torch.device(a.device)
    lb=read_gold(a.data_root/"lb500_s0"/"train_lb.parquet",expected_n=500)
    y=lb.loc[:,list(NORM_COLUMNS)].to_numpy(np.float32)
    man=json.loads(a.manifest.read_text()); hold=man["fold_ids"]["0"]; train=[i for i in lb.id if i not in set(hold)]
    ti=lb.index[lb.id.isin(train)].to_numpy(); hi=lb.index[lb.id.isin(hold)].to_numpy()
    tok=AutoTokenizer.from_pretrained(str(a.pretrain_path),local_files_only=True,use_fast=True)
    enc=AutoModel.from_pretrained(str(a.pretrain_path),local_files_only=True).to(dev).eval()
    feat=encode(enc,lb.text.tolist(),tok,dev).cpu().numpy(); del enc
    sc=StandardScaler().fit(feat[ti]); Xtr=torch.tensor(sc.transform(feat[ti]),dtype=torch.float32,device=dev); Xho=torch.tensor(sc.transform(feat[hi]),dtype=torch.float32,device=dev)
    ytr=torch.tensor(y[ti],device=dev)
    # ridge
    for alpha in (10,100,1000):
        m=Ridge(alpha=alpha).fit(Xtr.cpu().numpy(),ytr.cpu().numpy()); p=m.predict(Xho.cpu().numpy())
        print("ridge",alpha,sm(y[hi],p))
    # linear SGD MSE
    torch.manual_seed(0); net=nn.Linear(768,3).to(dev); opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-2)
    for ep in range(500):
        opt.zero_grad(); loss=F.mse_loss(net(Xtr),ytr); loss.backward(); opt.step()
        if ep%100==0:
            with torch.no_grad(): print("linear ep",ep,"train",sm(y[ti],net(Xtr).cpu().numpy()),"hold",sm(y[hi],net(Xho).cpu().numpy()))
    with torch.no_grad(): print("linear final",sm(y[hi],net(Xho).cpu().numpy()))
    # sigmoid/huber
    torch.manual_seed(0); net=nn.Sequential(nn.Linear(768,3)); opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-2)
    for ep in range(500):
        opt.zero_grad(); mu=torch.sigmoid(net(Xtr)); loss=F.huber_loss(mu,ytr,delta=0.1); loss.backward(); opt.step()
        if ep%100==0:
            with torch.no_grad(): print("sig ep",ep,"train",sm(y[ti],torch.sigmoid(net(Xtr)).cpu().numpy()),"hold",sm(y[hi],torch.sigmoid(net(Xho)).cpu().numpy()))
    with torch.no_grad(): print("sig final",sm(y[hi],torch.sigmoid(net(Xho)).cpu().numpy()))

if __name__=="__main__": main()
