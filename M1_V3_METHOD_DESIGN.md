# M1-v3: continuous VAD pseudo-label bias audit (frozen design)

## Status and gate

This branch contains **candidate target generation and the TRAIN-only audit**, not an approved semi-supervised result. The existing all-500 P1/P2/HPL checkpoints were trained on the proposed 100 audit labels and cannot serve as the audit reference. The builder therefore starts each reference from the local *pretrained* RoBERTa, trains on only its declared fit IDs, and freezes it for prediction. It uses the P1 architecture, 20 fixed P1 epochs (L1, AdamW 2e-5, batch 16, weight decay .01), then 160 fixed P2 supervised-control steps (L1, AdamW 5e-6, batch 16, weight decay .01). There is no DEV selection. This is a new reference fit, not a reuse of the historical P2 checkpoint; its accuracy must be judged in the audit.

The label-blind shared feature encoder is the local pretrained RoBERTa with attention-mask mean pooling. It is run in eval mode, pooled vectors are L2-normalized, and all four candidate arms use the same feature arrays and fitted reference prediction. OOF residuals use five independent fit models: each model sees 320 of the 400 fit labels and predicts only its held-out 80. The audit reference sees all 400 fit labels and predicts only the untouched 100. The 100 IDs, labels, and texts never enter a model fit, neighbor library, or parameter choice.

## Fixed candidates

For normalized prediction p, K=8 cosine neighbors, temperature=0.1, and neighbor softmax weights a_i:

| Arm | Target before [0,1] clipping | Meaning |
|---|---|---|
| A reference_target | p | Frozen reference, no calibration |
| B simregmatch_calibration | 0.5 p + 0.5 sum(a_i y_i) | Fixed neighbor-label interpolation inspired by SimRegMatch; not a claim of literal algorithm reproduction |
| C residual_direct | p + sum(a_i (y_i - p_i^OOF)) | Local OOF residual propagation |
| D residual_consistency | p + g sum(a_i (y_i - p_i^OOF)), g=abs(sum(a_i sign(y_i-p_i^OOF))) per V/A/D | Damps corrections when local residual directions disagree |

These coefficients and K/temperature are fixed before reading audit labels. Targets are NumPy arrays detached from any training graph; later integration must convert them to `torch.no_grad()` tensors. A clamp to [0,1] is applied equally. The query ID cannot occur in its own neighbor library. The feature encoder is shared and has seen no EmoBank label.

## One-time audit decision

The 500 labeled TRAIN IDs are sorted and permuted by seed 0; 400 fit, 100 audit. The 400 are independently permuted into five 80-item OOF folds. `manifest.json` records exact fit/held-out IDs and hashes. `audit_targets.py` checks the split and every fold before reading the audit metrics. It writes each arm's V/A/D and macro MAE in normalized units, correction direction accuracy for changed predictions, mean absolute correction, prediction variance, and changed fraction. Three paired candidate-vs-A bootstrap comparisons use 10,000 fixed resamples and a two-sided Bonferroni familywise 95% lower bound. **GO requires a positive lower bound.** Otherwise NO-GO: do not integrate into HPL or run 160-step GPU experiments. This gate is a conservative fixed audit criterion, not proof of generalization.

The old full-500 P2 score (0.175589±0.000676 on DEV in original [1,5] units) is context only. It is not a fair benchmark for the 400-fit audit. No DEV or TEST file is opened by the v3 builder or audit script.

## Current execution

`tests_m1_v3` and `tests_m1` pass on CPU. Check-only validates the 500 TRAIN split without writing output. The connected host currently reports no CUDA device and only 2 GB RAM; full six-model reference fitting has not run. No audit output, GO result, training modes, or DEV improvement is claimed. The conditional semi-supervised integration remains pending the precommitted audit gate.
