#!/usr/bin/env python3
"""Extract frozen *pretrained* RoBERTa features for M0 B0/B1/B2.

Only EmoBank labeled TRAIN, unlabeled TRAIN text, and DEV are read. No P1
checkpoint and no TEST file is opened. Evaluation-mode, attention-mask-aware
mean pooling matches the P1 architecture before its task-specific head.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from semilearn.datasets.nlp_datasets.emobank import GoldData, TextCollator, UnlabeledData
from scripts.vad_research.train_roberta_vad_p2 import UnlabeledCollator
from scripts.vad_research.train_roberta_vad_supervised import load_checked_splits, sha256_file


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budget', type=int, choices=(500,), default=500)
    parser.add_argument('--seed', type=int, choices=(0, 1, 2), default=0)
    parser.add_argument('--data-root', type=Path, default=REPO.parent/'data'/'emobank_processed')
    parser.add_argument('--pretrain-path', type=Path, default=REPO.parent/'pretrained'/'roberta-base')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-length', type=int, default=128)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.out is None:
        args.out = REPO/'saved_models'/'vad_research'/'m0'/f'pretrained_features_lb{args.budget}_s{args.seed}'
    if args.batch_size <= 0 or args.max_length <= 0:
        parser.error('batch size and maximum length must be positive')
    return args


@torch.inference_mode()
def extract(model, loader, device, labeled):
    features, ids, targets = [], [], []
    model.eval()
    for batch in loader:
        if labeled:
            tokens, target, batch_ids = batch
        else:
            tokens, batch_ids = batch
        tokens = {key:value.to(device) for key,value in tokens.items()}
        states = model(**tokens, return_dict=True).last_hidden_state
        mask = tokens['attention_mask'].unsqueeze(-1).to(dtype=states.dtype)
        pooled = (states*mask).sum(1)/mask.sum(1).clamp_min(1)
        features.append(pooled.cpu().numpy().astype(np.float32))
        ids.extend(batch_ids)
        if labeled:
            targets.append(target.numpy().astype(np.float32))
    x = np.concatenate(features)
    y = np.concatenate(targets) if labeled else None
    return x, np.asarray(ids, dtype=str), y


def main():
    args = parse_args()
    lb, dev, ulb, paths = load_checked_splits(args)
    hashes = {name:sha256_file(path) for name,path in paths.items()}
    weight_path = args.pretrain_path/'model.safetensors'
    if not weight_path.is_file():
        raise FileNotFoundError(f'pretrained weight file missing: {weight_path}')
    print(json.dumps({'labeled':len(lb),'unlabeled':len(ulb),'dev':len(dev),
                      'split_sha256':hashes,'pretrain_sha256':sha256_file(weight_path)},sort_keys=True))
    if args.check_only:
        print('PASS: pretrained weight and train/dev splits verified; no model, TEST or output writes')
        return
    if args.out.exists():
        raise FileExistsError(f'refusing to overwrite: {args.out}')
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(str(args.pretrain_path), local_files_only=True, use_fast=True)
    model = AutoModel.from_pretrained(str(args.pretrain_path), local_files_only=True).to(device)
    gold_collator = TextCollator(tokenizer, args.max_length)
    ulb_collator = UnlabeledCollator(tokenizer, args.max_length)
    loaders = {
        'lb':DataLoader(GoldData(lb),batch_size=args.batch_size,shuffle=False,collate_fn=gold_collator),
        'ulb':DataLoader(UnlabeledData(ulb),batch_size=args.batch_size,shuffle=False,collate_fn=ulb_collator),
        'dev':DataLoader(GoldData(dev),batch_size=args.batch_size,shuffle=False,collate_fn=gold_collator),
    }
    lb_x, lb_ids, lb_y = extract(model, loaders['lb'], device, True)
    ulb_x, ulb_ids, _ = extract(model, loaders['ulb'], device, False)
    dev_x, dev_ids, dev_y = extract(model, loaders['dev'], device, True)
    if (len(lb_x),len(ulb_x),len(dev_x)) != (len(lb),len(ulb),len(dev)):
        raise RuntimeError('feature count changed during extraction')
    args.out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(args.out/'pretrained_features.npz',
                        lb_x=lb_x,ulb_x=ulb_x,dev_x=dev_x,
                        lb_y=lb_y,dev_y=dev_y,
                        lb_ids=lb_ids,ulb_ids=ulb_ids,dev_ids=dev_ids)
    metadata={'model':'frozen pretrained RoBERTa-base, mask-aware mean pooling',
              'pretrain_path':str(args.pretrain_path.resolve()),
              'pretrain_sha256':sha256_file(weight_path),'split_sha256':hashes,
              'budget':args.budget,'seed':args.seed,'max_length':args.max_length,
              'counts':{'lb':len(lb),'ulb':len(ulb),'dev':len(dev)}}
    (args.out/'feature_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')


if __name__=='__main__':
    main()
