#!/usr/bin/env bash
# Explicit full CPU audit launcher. Never run this as part of tests or benchmark.
set -Eeuo pipefail
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=""
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
python_bin="/hy-tmp/rankup_project/envs/rankup/bin/python"
root="${1:-$repo/saved_models/vad_research}"
mkdir -p "$root"
run_root="$root/m1_v3_cpu_seed0_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir "$run_root"
trap 'code=$?; printf "exit_code=%s\n" "$code" > "$run_root/exit_status.txt"' EXIT
cd "$repo"
git rev-parse HEAD > "$run_root/git_commit.txt"
printf 'OMP_NUM_THREADS=%s\nMKL_NUM_THREADS=%s\nCUDA_VISIBLE_DEVICES=%s\nseed=0\ndevice=cpu\n' \
  "$OMP_NUM_THREADS" "$MKL_NUM_THREADS" "$CUDA_VISIBLE_DEVICES" > "$run_root/environment.txt"
printf 'RUN_ROOT=%s\n' "$run_root"
"$python_bin" scripts/vad_research/m1_v3/build_audit.py \
  --device cpu --seed 0 --out "$run_root/build" --check-only \
  > "$run_root/check_only.log" 2>&1
"$python_bin" scripts/vad_research/m1_v3/build_audit.py \
  --device cpu --seed 0 --out "$run_root/build" \
  > "$run_root/build.log" 2>&1
"$python_bin" scripts/vad_research/m1_v3/audit_targets.py \
  --inputs "$run_root/build" --out "$run_root/audit" \
  > "$run_root/audit.log" 2>&1
printf 'AUDIT_REPORT=%s\n' "$run_root/audit/audit_report.json"
