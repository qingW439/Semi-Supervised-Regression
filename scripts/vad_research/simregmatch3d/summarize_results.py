"""Summarize E0-E4 runs and apply the precommitted seed-0 gate."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from scripts.vad_research.simregmatch3d.train_simregmatch3d import MODES

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    a=p.parse_args()
    summaries={m:json.loads((a.root/m/'summary.json').read_text()) for m in MODES}
    e0=summaries['supervised_only'];e4=summaries['simregmatch_full']
    gate=(e4['best_macro_mae']<e0['best_macro_mae'] and e4['best_step']>0 and
          e4['best_macro_mae']<summaries['calibration_only']['best_macro_mae'] and
          e4['best_macro_mae']<summaries['filtering_only']['best_macro_mae'] and
          e4['mean_effective_unlabeled_grad_norm']>0)
    report={'decision':'PRELIMINARY_GO' if gate else 'NO-GO','runs':summaries}
    path=a.root/'comparison.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
