"""Read-only DEV summary for an already completed SimRegMatch-3D run."""
import argparse,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    a=p.parse_args()
    best=json.loads((a.run_dir/'dev_metrics_best.json').read_text())
    last=json.loads((a.run_dir/'summary.json').read_text())
    print(json.dumps({'best':best,'last':last},indent=2))

if __name__=='__main__':main()
