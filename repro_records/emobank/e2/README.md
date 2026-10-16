# EmoBank E2 ARC diagnostics

These records use the official EmoBank DEV split only. TEST was not opened.
Model checkpoints and per-example DEV predictions remain on the GPU server and
are not versioned.

## Two-epoch smoke checkpoint

`rankupvad_smoke_lb250_s0/` records the original two-epoch pilot made before
the confidence-filter fix. Its `training_history.csv` acceptance columns used
temperature-sharpened probabilities and are retained only as historical logs.
`arc_dev_diagnostics.json` reevaluates the saved checkpoint with unsharpened
probabilities:

| Dimension | All-pair accuracy | Coverage at 0.70 | Precision at 0.70 |
| --- | ---: | ---: | ---: |
| V | 0.5644 | 0.0254 | 0.7143 |
| A | 0.5610 | 0.0277 | 0.7047 |
| D | 0.5413 | 0.0153 | 0.5805 |

Coverage at cutoff 0.95 was zero in all three dimensions.

## Supervised-ranking diagnostic

`rankonly_diag_lb250_s0/` is a 10-epoch diagnostic with
`arc_loss_ratio=1.0` and `arc_ulb_loss_ratio=0.0`. The unsupervised loss was
still computed for logging, but its contribution to the optimized total loss
was zero. The best DEV regression checkpoint was epoch 5. Supervised ranking
loss fell from 0.6962 at epoch 1 to 0.4079 at epoch 10, demonstrating that the
ARC can learn from labeled pairs.

The epoch-5 checkpoint produced:

| Dimension | All-pair accuracy | Coverage at 0.80 | Precision at 0.80 |
| --- | ---: | ---: | ---: |
| V | 0.6545 | 0.0757 | 0.8526 |
| A | 0.6055 | 0.0916 | 0.7605 |
| D | 0.5783 | 0.0402 | 0.6636 |

These results do not justify a single cutoff for formal E2 training. In
particular, D remains substantially weaker than V and A. The next E2 design
should validate dimension-specific confidence behavior and the emotional
fidelity of the masked-token strong view before running three seeds.
