#!/bin/bash
cd /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-vadswap
/root/autodl-tmp/rankup_project/envs/rankup/bin/python - <<'PY'
import pandas as pd
d = pd.read_parquet('/root/autodl-tmp/rankup_project/data/emobank_processed/test_gold.parquet')
print('shape', d.shape)
print('columns', list(d.columns))
print(d[['V','A','D']].describe().round(3).to_string())
PY
