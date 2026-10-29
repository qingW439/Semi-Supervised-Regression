import json, glob, os, statistics

COPY = '/root/autodl-tmp/rankup_project/Semi-Supervised-Regression-vadswap/saved_models/vad_research'
ORIG = '/root/autodl-tmp/rankup_project/Semi-Supervised-Regression/saved_models/vad_research'

rows = []
for root in (COPY, ORIG):
    for p in glob.glob(os.path.join(root, '*', 'dev_metrics_best.json')):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        m = d['metrics']
        name = os.path.basename(os.path.dirname(p))
        ccc = [m[k]['ccc'] for k in ('V', 'A', 'D')]
        r2 = [m[k]['r2'] for k in ('V', 'A', 'D')]
        rows.append((m['macro_mae_original_scale'], statistics.mean(ccc), statistics.mean(r2),
                     name, d.get('step', ''), p))

rows.sort()
print(f"{'MAE':>9} {'CCC':>7} {'R2':>7} {'step':>5}  name")
for mae, ccc, r2, name, step, p in rows:
    print(f'{mae:9.6f} {ccc:7.4f} {r2:7.4f} {str(step):>5}  {name}')
