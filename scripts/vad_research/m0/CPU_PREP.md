# M0 CPU preparation (no training results)

This branch adds CPU-tested mathematical components and untrained A0/A1, feature-extraction, and B0/B1/B2 entry points. No full RoBERTa forward/backward or experiment result exists yet. Historical P1/P2 artifacts are read only.

## HPL component

`hpl_core.py` follows the official STS-B `sts/main_ours.py` head-level update: a differentiable inner optimizer step for the regression head, then a labeled meta batch updates the uncertainty network. The unlabeled term is `mean(exp(-s) * (strong - stopgrad(weak))^2 / 2)` over batch and V/A/D. Uncertainty input is `[stopgrad(strong-weak), stopgrad(strong)]` per dimension. The outer loss is labeled MSE minus `lambda2 * mean(s_meta)`.

The official STS-B implementation uses sentence-pair augmentation and a scalar target. EmoBank will use unchanged single-sentence text with dropout views and three output coordinates. Its main training loop must retain the P1 model, `(gold-1)/4` targets, Dev-only selection, and independent labeled meta batches. Never port the official STS-B loop that reads Test after every epoch.

`higher==0.2.1` is required for the meta step. For CPU tests it was installed only into `/hy-tmp/rankup_project/third_party/python_higher`, outside the shared Python environment and this Git repo. If the instance changes, install this pinned dependency to a new isolated target directory and set `PYTHONPATH` accordingly. No model weights or historical results were changed.

## Spectral component

`spectral_core.py` implements a deterministic Nyström approximation to the ICML 2026 full kernel eigenbasis. The basis is fitted only from training proxy features; `transform()` evaluates Dev without refitting. B1 and B2 must use the same fixed kernel gamma, landmark count, spectral rank, and Ridge tuning procedure. Select Ridge regularization from labeled training folds only. B2 may use unlabeled training features to learn the basis; neither B1 nor B2 may use Dev features to fit it. Pretrained, frozen RoBERTa-base features must be used for B0/B1/B2, never P1 fine-tuned features. Feature extraction and B0/B1/B2 code is prepared, but extraction and real-data fitting are unrun.

## CPU checks performed

Run from repository root:

```
PYTHONPATH=/hy-tmp/rankup_project/third_party/python_higher:$PWD \
  /hy-tmp/rankup_project/envs/rankup/bin/python -m unittest discover \
  -s tests_m0 -p 'test_*.py' -v
```

Twelve tests passed on CPU: HPL weighting, zero residual, shapes, stopped weak gradient, differentiable inner-to-outer uncertainty gradient, unchanged real head during virtual update, Nyström full-landmark kernel reconstruction, deterministic Dev transform, and invalid inputs. Additional tests check train-only spectral fitting, Ridge cross-validation, feature ID isolation, and P1-compatible mask-aware pooling. These are mechanism tests, not an EmoBank training smoke test. `train_hpl_vad.py --mode a0/a1 --check-only --seed 0` and `extract_roberta_features.py --check-only --seed 0` passed without loading models or writing outputs.

## Pending gates

1. Review `train_hpl_vad.py` against the P1 architecture and official HPL loop. It uses normalized MSE and Adam for both A0/A1. higher==0.2.1 rejects AdamW in the differentiable inner loop; optimizer and loss differ from historical P2 AdamW/L1. Run a GPU one-step test before full training.
2. Run `extract_roberta_features.py` with the original pretrained encoder. Check output feature hashes and ID isolation. Then run `run_spectral_ridge.py` B0/B1/B2 and record Nyström time/memory.
3. Run seed 0 A0/A1 and B0/B1/B2, then HPL seeds 1/2 only after the mechanism check. No M0 choice until comparable results exist.
