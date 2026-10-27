#!/usr/bin/env python3
"""Read-only DEV evaluation of an M1 checkpoint. TEST is never opened."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from semilearn.datasets.nlp_datasets.emobank import GoldData, TextCollator
from scripts.vad_research.train_roberta_vad_p2 import dev_evaluate
from scripts.vad_research.train_roberta_vad_supervised import (
    RobertaVADRegressor, load_checked_splits, sha256_file,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', choices=('best', 'last'), default='best')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    config = json.loads((args.run_dir/'run_config.json').read_text(encoding='utf-8'))
    if config['mode'] not in ('hpl_original', 'hpl_ggr', 'hpl_diagonal',
                              'hpl_crossdim', 'hpl_crossdim_ggr'):
        raise ValueError('Not an M1 run')
    split_args = argparse.Namespace(data_root=Path(config['data_root']),
                                    budget=config['budget'], seed=config['seed'])
    _, dev, _, paths = load_checked_splits(split_args)
    if {key: sha256_file(path) for key, path in paths.items()} != config['split_sha256']:
        raise ValueError('Split hashes differ from training run')
    checkpoint_path = args.run_dir/f'model_{args.checkpoint}.pth'
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    tokenizer = AutoTokenizer.from_pretrained(config['pretrain_path'],
                                               local_files_only=True, use_fast=True)
    loader = DataLoader(GoldData(dev), batch_size=config['eval_batch_size'],
                        collate_fn=TextCollator(tokenizer, config['max_length']))
    device = torch.device(args.device)
    model = RobertaVADRegressor(config['pretrain_path'], dropout=config['dropout'],
                                offline=True).to(device)
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'], strict=True)
    metrics, stats, _, _, _ = dev_evaluate(model, loader, device)
    print(json.dumps({'checkpoint': str(checkpoint_path), 'step': state['step'],
                      'metrics': metrics, 'stats': stats}, indent=2))


if __name__ == '__main__':
    main()
