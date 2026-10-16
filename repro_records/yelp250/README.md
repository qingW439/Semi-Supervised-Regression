# Yelp Review, 250 labels: Test evaluation record

This directory records the nine `rankup-reproduction` experiments: supervised,
RankUp, and RankUp+RDA, each with seeds 0, 1, and 2. The YAML files used here
were byte-for-byte identical to their copies in the experiment archive when
this record was prepared. The nine `model_best.pth` files also matched their
archived counterparts byte-for-byte. The checksum lists record the files used
for this evaluation; model weights and Yelp data are not stored in Git.

The archived training logs confirm that supervised seed 0 and RankUp seed 0
used `overwrite=True` and the original `load_path` values. Those two YAMLs
are kept as actually trained, reversing the later protective edits on GitHub.
Do not rerun training against the existing model directories without first
changing the output path or protecting the saved weights.

## Reproduce the metrics

Run from the repository root, using the training environment and the cached
BERT tokenizer/configuration:

```bash
HF_HOME=/hy-tmp/rankup_project/hf_cache \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/hy-tmp/rankup_project/envs/rankup/bin/python \
  scripts/eval_yelp_nine_models.py --num-workers 2
```

The script uses each YAML's `eval_batch_size=8`. This matters because the BERT
regressor averages hidden states across dynamically padded tokens. Changing
the batch size can change predictions. It reads the actual Yelp Test split,
loads checkpoint EMA weights, and restores output labels using the scaling
range saved in each checkpoint. It does not train or modify the models.

`test_metrics_by_seed.csv` reports MAE, R², Spearman correlation (SRCC), and
Pearson correlation (LCC) from each **best** checkpoint. It also includes the
MAE from each final checkpoint and the MAE recorded in the training log.
`test_metrics_summary.csv` summarizes the best-checkpoint metrics across three
seeds using the arithmetic mean and sample standard deviation (`ddof=1`).

## Why the logged `test/best_mae` can differ

In the training code, `EvaluationHook.after_run()` loads `model_best.pth`
before testing, but `AlgorithmBase.evaluate()` then applies an EMA shadow
that remains at the final training iteration. `AlgorithmBase.load_model()`
does not reset that shadow. With these runs' `ema_m=0`, the logged
`test/best_mae` therefore describes `latest_model.pth`, despite its name.
The script checks the logged number against that final checkpoint and keeps
the best-checkpoint result separate. A `CHECK` in the CSV means the final
checkpoint did not reproduce the logged MAE within 0.003 and requires review.
