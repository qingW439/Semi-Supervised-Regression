# M0 CPU preparation (no training results)

This branch adds tested mathematical components for the planned HPL and spectral comparisons. It does **not** contain a finished A0/A1 trainer or B0/B1/B2 results. Historical P1/P2 artifacts are read only.

## HPL component

`hpl_core.py` follows the official STS-B `sts/main_ours.py` head-level update: a differentiable inner optimizer step for the regression head, then a labeled meta batch updates the uncertainty network. The unlabeled term is `mean(exp(-s) * (strong - stopgrad(weak))^2 / 2)` over batch and V/A/D. Uncertainty input is `[stopgrad(strong-weak), stopgrad(strong)]` per dimension. The outer loss is labeled MSE minus `lambda2 * mean(s_meta)`.

The official STS-B implementation uses sentence-pair augmentation and a scalar target. EmoBank will use unchanged single-sentence text with dropout views and three output coordinates. Its main training loop must retain the P1 model, `(gold-1)/4` targets, Dev-only selection, and independent labeled meta batches. Never port the official STS-B loop that reads Test after every epoch.

`higher==0.2.1` is required for the meta step. For CPU tests it was installed only into `/hy-tmp/rankup_project/third_party/python_higher`, outside the shared Python environment and this Git repo. If the instance changes, install this pinned dependency to a new isolated target directory and set `PYTHONPATH` accordingly. No model weights or historical results were changed.

## Spectral component

`spectral_core.py` implements a deterministic Nyström approximation to the ICML 2026 full kernel eigenbasis. The basis is fitted only from training proxy features; `transform()` evaluates Dev without refitting. B1 and B2 must use the same fixed kernel gamma, landmark count, spectral rank, and Ridge tuning procedure. Select Ridge regularization from labeled training folds only. B2 may use unlabeled training features to learn the basis; neither B1 nor B2 may use Dev features to fit it. Pretrained, frozen RoBERTa-base features must be used for B0/B1/B2, never P1 fine-tuned features. Feature extraction and B0/B1/B2 are pending.

## CPU checks performed

Run from repository root:

```
PYTHONPATH=/hy-tmp/rankup_project/third_party/python_higher:$PWD \
  /hy-tmp/rankup_project/envs/rankup/bin/python -m unittest discover \
  -s tests_m0 -p 'test_*core.py' -v
```

Eight tests passed on CPU: HPL weighting, zero residual, shapes, stopped weak gradient, differentiable inner-to-outer uncertainty gradient, unchanged real head during virtual update, Nyström full-landmark kernel reconstruction, deterministic Dev transform, and invalid inputs. These are mechanism tests, not an EmoBank training smoke test.

## Pending gates

1. Build a full A0/A1 training entry point that reuses P2 split/checkpoint checks and saves to new output directories. Match 160 supervised steps, seed, labeled batch order, and optimizer; log step 0 and all three dimensions. A1 alone adds HPL meta and unlabeled losses. Verify exact forward/backward on GPU before full run.
2. Build frozen pretraining feature extraction with ID and source checkpoint hashes. Implement B0/B1/B2 with train-only CV and fit isolation. Measure Nyström time/memory before 8062-sample B2.
3. Run seed 0 A0/A1 and B0/B1/B2, then HPL seeds 1/2 only after the mechanism check. No M0 choice until comparable results exist.
