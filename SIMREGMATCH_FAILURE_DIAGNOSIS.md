# SimRegMatch-3D failure diagnosis

## Scope and provenance

This is a read-only post-hoc diagnosis of the completed seed0 E0-E4 experiment (`acd1b63` protocol). It does not change the network, loss, threshold, optimizer, historical outputs, or checkpoints. It creates no optimizer and makes no parameter update. The script asserts that all regression-head parameters are bitwise unchanged. It reads only labeled/unlabeled TRAIN and DEV; no TEST. Existing E0/E4 best and Step160 checkpoints are loaded from the completed run. DEV is used for description only, not selection or tuning.

Fixed sampling: NumPy RNG seed `20261028`, eight non-overlapping labeled batches of 16 (128 TRAIN labels) and eight non-overlapping unlabeled batches of 64 (512 TRAIN texts). Selected labeled-ID SHA256 `676b54a99f82b27ad857367cb53c76e01948b5fca30e1ca71396087aaec74c88`; unlabeled-ID SHA256 `16d867be62c6d3ef657c948611938274b78579ab5257227db20bfd19f1143e0d`. Full IDs, data/checkpoint hashes, per-batch values, and raw statistics are in the separate `diagnostic.json`. Mechanism statistics use **E4 Step160** checkpoint in train mode for five MC dropout draws, the fixed beta=0.6, temperature=1.0, lambda_u=0.01, and the same previous-batch 95th-percentile filter order. DEV statistics use E0/E4 best and last checkpoints in eval mode. The scripted diagnostic performs forward/backward calculations but no optimization.

Definitions: similarity entropy `H=-sum_j w_j log(w_j)`; effective neighbors `exp(H)` and inverse-Simpson `1/sum_j w_j^2`. Distances and variances below use normalized [0,1] VAD. Gradient norms are over regression-head parameters; the reported scaled unlabeled gradient includes lambda_u=0.01. Supervised loss is P2 mean L1; unlabeled loss is retained-sample sum-dimension MSE.

## 1. Neighborhood calibration is almost batch averaging

| Statistic across 512 unlabeled samples | Mean | P10 | P50 | P90 |
|---|---:|---:|---:|---:|
| Similarity entropy (maximum `ln(16)=2.772589`) | 2.772350 | 2.772160 | 2.772380 | 2.772497 |
| Effective neighbors `exp(H)` (maximum 16) | 15.996179 | 15.993145 | 15.996667 | 15.998536 |
| Inverse-Simpson effective neighbors | 15.992405 | 15.986402 | 15.993348 | 15.997076 |
| Maximum single-neighbor weight (uniform=0.0625) | 0.064360 | 0.063761 | 0.064302 | 0.064975 |
| Neighbor VAD distance to labeled-batch mean, L2 | 0.000871 | 0.000323 | 0.000769 | 0.001578 |
| Model pseudo VAD distance to labeled-batch mean, L2 | 0.075608 | 0.032545 | 0.068386 | 0.129599 |
| Calibrated VAD distance to labeled-batch mean, L2 | 0.045541 | 0.019509 | 0.041164 | 0.078111 |

The neighbor label is only 1.15% as far from the labeled-batch mean as the model pseudo label. All 512 calibrated targets move closer to the batch mean. The distance ratio `0.045541/0.075608 = 0.6023` is almost exactly the fixed model-prediction share beta=0.6. This is direct evidence that this temperature/feature setting supplies a nearly uniform batch-label average rather than discriminative neighbors. The mean labeled-batch V/A/D standard deviations were `0.0829/0.0672/0.0578`, far larger than the 0.000871 neighbor-to-mean L2 distance.

Within a batch, calibrated/model pseudo-label variance ratios are `0.3627/0.3616/0.3620` for V/A/D, almost beta-squared `0.36`. Across all 512 samples pooled, the ratios are `0.3946/0.3789/0.4275` because batch means differ. The calibration therefore compresses prediction variation; this is a mechanism observation, not proof of the full DEV performance cause.

## 2. Unlabeled head gradients are small after the fixed loss weight

Across eight identical paired TRAIN batches, the mean supervised head-gradient norm is `0.98077`; the unscaled unlabeled norm is `0.18952`; the actual lambda-scaled unlabeled norm is `0.0018952`. The per-batch scaled-unlabeled/supervised norm ratio has mean `0.001934` (0.193%), P10 `0.001383`, median `0.001688`, P90 `0.002750`, range `0.001225–0.003860`.

The cosine has mean `0.368`, median `0.457`, range `-0.447–0.797`; two of eight batches are negative. Conflict occurs on some batches, but a consistently opposing gradient is **not** supported. The dominant observation is the very small scaled unlabeled contribution. This is local to the regression head and these fixed batches; it does not establish the encoder-gradient ratio or a causal ablation of lambda_u.

## 3. MC uncertainty reliability is unproven

On the 128 selected labeled TRAIN samples, sample-level MC variance sum versus mean normalized absolute error has Spearman correlation `-0.029`. The lowest-variance quartile has mean error `0.02300`; the highest-variance quartile `0.02054`, so lower MC variance did not correspond to lower error in this sample. **These labels participated in fitting the E4 checkpoint. This in-sample comparison is exploratory only and is not an unbiased pseudo-label quality validation.** No independent unlabeled gold or held-out TRAIN audit was used here. Thus the present evidence cannot establish that the 95th-percentile MC filter identifies reliable targets.

## 4. DEV prediction spread, fixed descriptive evaluation

All standard deviations and errors in this table are on original [1,5] VAD scale. Gold DEV standard deviations are V `0.33975`, A `0.25093`, D `0.20181`.

| Checkpoint | Step | Macro MAE | V pred std / MAE / CCC | A pred std / MAE / CCC | D pred std / MAE / CCC |
|---|---:|---:|---|---|---|
| E0 best | 128 | 0.175543 | 0.26151 / 0.18351 / 0.65986 | 0.17227 / 0.18314 / 0.41711 | 0.12811 / 0.15999 / 0.24951 |
| E4 best | 128 | 0.175878 | 0.25953 / 0.18571 / 0.65437 | 0.16944 / 0.18198 / 0.41353 | 0.12925 / 0.15994 / 0.25165 |
| E0 Step160 | 160 | 0.177198 | 0.26683 / 0.18436 / 0.66259 | 0.17317 / 0.18765 / 0.39900 | 0.12752 / 0.15958 / 0.25820 |
| E4 Step160 | 160 | 0.176964 | 0.26609 / 0.18508 / 0.66004 | 0.17286 / 0.18520 / 0.40765 | 0.12789 / 0.16060 / 0.25817 |

Both methods have predictions narrower than gold on all dimensions. At the best step, E4 V and A predictions are slightly narrower than E0; D is slightly wider. E4 best V MAE worsens by `0.002202`, A improves by `0.001156`, and D is nearly unchanged. DEV numbers are post-hoc descriptions and were not used to change parameters.

## Final A–D judgment

**A. Yes.** Neighbor calibration is numerically almost the labeled minibatch mean: entropy is 99.991% of its maximum, effective neighbors 15.996/16, and neighbor-to-mean L2 is 0.000871.

**B. The actual unlabeled head gradient is very small, not consistently conflicting.** Its norm averages 0.193% of the supervised norm after lambda_u, with two negative cosines in eight batches.

**C. No reliable MC uncertainty evidence.** The only available label-error comparison is in-sample TRAIN exploration, showing essentially no favorable association; it cannot validate pseudo-label quality.

**D. The strongest supported explanation is that cosine-softmax neighbors are nearly uniform and calibration acts as a beta-weighted batch-mean shrinkage, while the lambda-scaled unlabeled head update is tiny.** DEV also shows mild V/A spread compression and no best-MAE gain over E0. These observations do not isolate one causal component: independent pseudo-label accuracy, encoder-gradient effects, whether different fixed hyperparameters would work, and multi-seed stability remain undetermined. The completed seed0 experiment remains NO-GO; no new 160-step training or method extension is justified by this diagnostic.

The reproducible read-only script is `scripts/vad_research/simregmatch3d/diagnose_failure.py`; use a new output directory. The full raw result is `saved_models/vad_research/simregmatch3d_failure_diag_20261028T104731Z/diagnostic.json` in the AutoDL worktree. Historical run files remain untouched.
