#!/usr/bin/env bash
# AutoDL SimRegMatch-3D precommitted seed-0 E0-E4 experiment.
set -euo pipefail
REPO=/root/autodl-tmp/rankup_project/SSR-simregmatch
PY=/root/autodl-tmp/rankup_project/envs/rankup/bin/python
P1=/root/autodl-tmp/rankup_project/Semi-Supervised-Regression/saved_models/vad_research/p1_roberta_lb500_s0
ROOT="${1:?Provide a unique, nonexistent output directory}"
cd "$REPO"
if [[ -e "$ROOT" ]]; then echo "Refusing to overwrite: $ROOT" >&2; exit 2; fi
if [[ -n "$(git status --porcelain)" ]]; then echo 'Dirty source worktree' >&2; exit 2; fi
mkdir -p "$ROOT"
trap 'code=$?; echo "$code" > "$ROOT/exit_code"; exit "$code"' EXIT
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
printf '%s\n' "$(git rev-parse HEAD)" > "$ROOT/git_commit.txt"
git remote -v > "$ROOT/git_remote.txt"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader > "$ROOT/gpu_start.txt"
"$PY" -m pytest -q tests_m0 tests_m1 tests_m1_v3 tests_simregmatch3d > "$ROOT/tests.log" 2>&1
"$PY" scripts/vad_research/simregmatch3d/train_simregmatch3d.py --mode simregmatch_full --seed 0 --p1-dir "$P1" --device cuda --check-only > "$ROOT/check_only.log" 2>&1
for MODE in supervised_only consistency_only calibration_only filtering_only simregmatch_full; do
  "$PY" scripts/vad_research/simregmatch3d/train_simregmatch3d.py --mode "$MODE" --seed 0 --p1-dir "$P1" --device cuda --steps 160 --eval-every 32 --out "$ROOT/$MODE" > "$ROOT/$MODE.log" 2>&1
  echo "DONE $MODE $(date -u +%FT%TZ)"
done
"$PY" scripts/vad_research/simregmatch3d/summarize_results.py --root "$ROOT" > "$ROOT/comparison.log" 2>&1
