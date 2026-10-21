# EmoBank SSR-RCUS 3D adaptation: fixed protocol

Base commit: `2174f50c80f23363c70fd5f767dea2384d06bd91`. This branch adds files only. Old B0/B1/B2/P1/R0/R1 outputs are read-only. All new artifacts go to `saved_models/emobank/rcus3d_adapt/`. No script in this protocol opens Test.

## Preparatory stages

1. **Supervised audit.** Compare historical E1 and B0 Dev predictions with training-label mean/median. Probe frozen E1 features with Ridge for diagnosis, and separately use *base pretrained BERT* features for a clean Ridge probe. Ridge alpha is chosen from `[0.01, 0.1, 1, 10, 100, 1000]` by five-fold training-only macro MAE; Dev is evaluated once. The E1-feature CV has encoder label contamination because E1 saw all 500 labels, so its CV score is not a generalization estimate.
2. **Head-only warmup.** Load the original E1 checkpoint strictly. Encode all 500 labeled texts in eval mode. Freeze encoder and three-output regressor. Train only a linear `[hidden,3,2]` rank head on unique labeled pairs for three epochs, batch 8, AdamW learning rate `1e-3`, weight decay 0, seed 3700. Equal gold labels have target 0.5. Save the final head and unchanged E1 backbone. Training-pair accuracy and confidence before/after are descriptive, not independent validation.
3. **Gradient weight.** On eight fixed labeled batches, use the warmup checkpoint with no optimizer step. Set the supervised Close Mixup coefficient to `0.1 × median(||∇encoder L1||) / median(||∇encoder Lclose,l||)`. Record all norms, loss values and cosines. This is a predeclared 10% *median norm* target; it does not guarantee the same ratio for every batch.
4. **Strict OOF calibration.** Five folds of the 500 labeled rows, shuffled with seed 0. For each fold, train a new BERT regression model from base pretrained weights on only the 400 training rows for a fixed 14 epochs, then train its linear rank head for three epochs on those same rows. Score unique pairs among the 100 held-out rows. This avoids reusing E1's encoder, which had seen held-out labels. No Dev label is used. Fit one scalar temperature per V/A/D on non-tied OOF pairs; report 10-bin reliability and ECE. Evaluate threshold grid `[0.5,0.6,0.7,0.8,0.85,0.9,0.925,0.95,0.975,0.99]`; choose the lowest threshold with pooled pseudo-rank accuracy at least 0.8, each-fold accuracy at least 0.7, at least 200 pooled and 20 per fold accepted pairs. Disable a dimension when none qualifies. This gate was fixed before reviewing OOF outcomes. Pairs sharing sentences are dependent; these OOF accuracies are calibration diagnostics, not independent uncertainty bounds. Fold encoders differ from the full-data E1 encoder, which is a transfer limitation.

## Fixed joint-training comparison

All three starts load the **same full head-only warmup checkpoint**. All use EmoBank lb500/seed0, 25 epochs, 63 labeled optimizer steps per epoch, labeled batch 8, unlabeled batch 8 when active, AdamW `1e-5`, weight decay `5e-4`, text max length 128, dropout 0.1, latent Mixup `α=0.5`, and the same gradient-derived supervised coefficient. Separate Torch and NumPy random streams preserve labeled sampling, labeled dropout and labeled Mixup draws across modes. Use Dev Macro-MAE on original 1–5 scale to choose each best checkpoint, and save epochs 1/5/15/25 independently.

- **A0:** L1 + supervised Close Mixup; unsupervised coefficient 0.
- **A1:** same supervised terms + unsupervised Close Mixup coefficient 1.0; fixed 0.95 for all dimensions, temperature 1.
- **A2:** same as A1 except dimensionwise OOF temperatures, thresholds and active masks. The denominator remains all unique candidate pairs × three dimensions, even if one dimension is disabled.

For endpoints `i<j`, generate `z_i'=λz_i+(1−λ)z_j` and `z_j'=(1−λ)z_i+λz_j` with `λ=max(λ',1−λ')`, `λ'~Beta(0.5,0.5)`. The two local rank logits compare `i` with `i'` and `j'` with `j`; both target the endpoint rank. Rank BCE terms are summed and divided by `3 × number of unique pairs`; unlabeled terms are masked before that division. The weak pseudo-rank uses the same text in eval mode; the strong representation differs by train-mode dropout. No EMA, RDA, uncertainty network, numerical pseudo label or semantic text alteration is used.

Because the auxiliary head is linear, its two local logits *mathematically equal* `(1−λ)` times the student endpoint pair logit. Thus Close Mixup changes ranking margins rather than adding an independently expressive local relation in this architecture. This identity is tested. It is also a reason to interpret later gains carefully, not a reason to modify A0/A1/A2 midstream.

The three experiments will be run once, with no adaptive hyperparameter search. Smoke-test Dev metrics will not be treated as full results. One seed cannot establish stability.
