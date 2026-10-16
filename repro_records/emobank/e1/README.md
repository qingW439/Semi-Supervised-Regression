# E1 provenance and shared VAD backbone

The `lb*_s0` metrics and `training_script.sha256` were recorded from the
original standalone E1 script at commit `44d992b`. The checksum identifies
that historical script, not the refactored copy at the current branch tip.
To inspect the original source, run:

```bash
git show 44d992b:scripts/emobank/train_emobank_supervised.py
```

The later refactor imports `BertVADRegressor` from
`semilearn/nets/bert/bert_vad.py` and data loading from
`semilearn/datasets/nlp_datasets/emobank.py`. It does not change the archived
metrics. Run `scripts/emobank/check_e1_backbone.py` in the server environment
to compare the archived E1 forward path and shared backbone after strict
loading of the existing `lb250_s0/model_best.pth` checkpoint. It uses dev
texts only. The observed maximum absolute output difference was zero.

`semilearn/algorithms/rankup_vad` provides dimension-wise ranking components
for E2. No E2 training results are claimed here. An E2 training loop must
supply distinct weak and strong views of unlabeled text before enabling its
unsupervised loss.
