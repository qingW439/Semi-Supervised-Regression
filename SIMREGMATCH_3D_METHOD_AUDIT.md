# SimRegMatch-3D method audit

Source: Jo et al., *Deep semi-supervised regression via pseudo-label filtering and calibration*, Applied Soft Computing (2024), [paper](https://www.sciencedirect.com/science/article/pii/S1568494624004447). Official [repository](https://github.com/YongwonJo/SimRegMatch), inspected at `0a57b897caeb5e7beebbe8379f999cb2f409bece`: `tasks/SimRegMatchTrainer.py`, `utils/args.py`, `models/resnet_proposed.py`.

| Component | Official implementation | SimRegMatch-3D adaptation |
| --- | --- | --- |
| Task | AgeDB image, scalar age regression | EmoBank text, 3 continuous VAD outputs, normalized targets [0,1] |
| Backbone | ResNet50, average pooled image feature | P1 RoBERTa-base with masked mean pooled text feature and same 3-output head |
| Random views | Repeated weak-image forwards with dropout; strong-image augmentation as student input | Five independent RoBERTa dropout forwards on the same unmodified text; separate dropout student forward. No back translation or EDA. |
| MC uncertainty | Prediction variance per output, summed over output dimensions | Unbiased variance per V/A/D, sum to one sample uncertainty |
| Similarity | L2 normalized weak and labeled features, cosine-softmax over current labeled batch | Same calculation using pooled RoBERTa features and current labeled batch only; temperature 1.0 |
| Calibration | `beta * model_pseudo + (1-beta) * similarity_label` | Same vector calculation with beta 0.6, detached and clamped to [0,1] |
| Filtering | Apply previous threshold; update from current batch after mask | Same order; initial threshold 10; 95th percentile calculated from current batch for next step |
| Percentile correction | `utils/args.py` sets `percentile=0.95` and passes it to `np.percentile`: **0.95th percentile** | Precommitted **95th percentile**. This is a deliberate adaptation correction, not an exact reproduction of original code. |
| Unlabeled loss | Sum coordinate MSE for each retained sample, average over retained samples | Same 3-coordinate sum; empty retained set yields differentiable zero |
| Supervised optimization | Official scalar MSE with Adam and its own learning rate | Matched EmoBank P2 control: normalized 3D mean L1, AdamW 5e-6, weight decay 0.01, clipping 1.0. All five arms share this. |
| Initialization | Original image model initialization | Frozen P1 seed-matched best checkpoint; checkpoint and pretrained SHA256 verified |
| DEV / TEST | Original code constructs a test loader and inference route | DEV only for precommitted best selection; no TEST reads or inference path |

The official `SimRegMatchTrainer.py` calculates five weak predictions, averages both predictions and features, uses normalized cosine similarity to current labeled minibatch labels, and applies the threshold from the previous batch before computing the next batch's percentile. Its `loss_u` can divide by zero before a NaN fallback. The adaptation uses a zero-safe differentiable loss. E1-E4 retain this calibrated or uncalibrated pseudo target as appropriate. They use a same-text dropout view as student target consistency; this is a text adaptation of image weak/strong views, not a claim of exact official task reproduction.

No HPL uncertainty learner, GGR, cross-dimensional utility, DEV tuning, OOF residual module, or new method component is included. The work is an established-method admission experiment.
