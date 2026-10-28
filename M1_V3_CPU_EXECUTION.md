# M1-v3 CPU execution report

Branch baseline: `vad-m1-v3-pseudolabel-calibration`, `e50814f261d88a7afa625c9dbf660ea3ff094cfe`. The seven pre-existing M0 working-tree modifications are excluded from this work.

## Verification before edits

- `build_audit.py --help` lists `--device {auto,cpu,cuda}` and `--check-only`. The real train/inference path moves model, tokens, and labels to the selected `torch.device`. CUDA seed and cache calls are guarded; no CUDA-only forward, AMP, or mixed-device operation was found.
- With `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=`: `pytest -q tests_m0 tests_m1 tests_m1_v3` gave **36 passed**, and `build_audit.py --device cpu --check-only` confirmed TRAIN SHA256 `c7c67afd9c62b71cc7dfd5a31e4958072805f157dc7bf92fa06d52f3ba9fb094`, fit=400, audit=100, five OOF folds.

## CPU implementation

`benchmark_cpu.py` executes real RoBERTa-base supervised updates through the same `supervised_update` function as the audit. Its default two warmup and three measured steps in each P1/P2 phase are **benchmark only** and cannot generate pseudo-label quality, GO, or NO-GO results. `run_cpu_audit.sh` must be invoked explicitly. It fixes the three environment variables and seed 0, writes to a unique timestamped root, keeps check-only/build/audit logs, stores the exit code, and never reuses a historical output directory. The builder saves each fold model, per-fold fit and held-out IDs, model seed, checkpoint SHA256, incremental provenance, final reference model, run config, split/pretrain hashes, commit, and the final manifest. The audit verifies these before reporting metrics. Each fold model starts from the local pretrained RoBERTa, not the full-500 P1/P2 checkpoint.

## Measured resource block

The current container's cgroup memory limit is **2 GiB**, CPU quota is **1 CPU**, and there is no swap. `free -h` exposes the host's 2 TiB RAM, which is not the container limit. The first attempted full-parameter RoBERTa benchmark was OOM-killed before producing a timed update. Thus **no valid per-training-step seconds were measured**. The new container-aware preflight detects 2.00 GiB and exits before loading a model. Its conservative minimum is 8 GiB; this is an operational floor, not a guarantee of sufficiency. A CPU host with at least 8 GiB allocated RAM is required for the short benchmark. The full audit has not been launched.

The fixed audit consists of five 320-label fold models and one 400-label reference model. P1: `5 × 20 × 20 + 20 × 25 = 2,500` optimizer updates. P2: `6 × 160 = 960` optimizer updates. Total: **3,460 full-parameter RoBERTa updates**, plus 500 feature extractions, 400 OOF predictions, 100 reference predictions, model loads, hashes, and checkpoint I/O. Once the CPU benchmark runs, it reports measured P1/P2 step seconds and `2500 × P1_mean + 960 × P2_mean + 6 × model_load_seconds`; inference and I/O remain additional time. No wall-time number is fabricated for the present 2 GiB host.

## Commands after increasing CPU container RAM

From the repository root, first benchmark only:

```bash
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=""
cd /hy-tmp/rankup_project/Semi-Supervised-Regression
BENCH="saved_models/vad_research/m1_v3_cpu_benchmark_seed0_$(date -u +%Y%m%dT%H%M%SZ)_$$"
/hy-tmp/rankup_project/envs/rankup/bin/python scripts/vad_research/m1_v3/benchmark_cpu.py --seed 0 --warmup-steps 2 --timed-steps 3 --out "$BENCH"
cat "$BENCH/benchmark.json"
```

After inspecting the benchmark and resources, **manually** start the complete audit:

```bash
cd /hy-tmp/rankup_project/Semi-Supervised-Regression
bash scripts/vad_research/m1_v3/run_cpu_audit.sh
```

The launcher prints `RUN_ROOT`. Inspect its `exit_status.txt`, `build.log`, `audit.log`, `build/fold_provenance.json`, and `audit/audit_report.json`. Running it on the current 2 GiB container exits nonzero at the memory preflight and produces no audit decision.

## Final checks on this container

After implementation, `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= python -m pytest -q tests_m0 tests_m1 tests_m1_v3` returned **39 passed, 1 existing dependency warning**. CPU `--check-only`, `benchmark_cpu.py --help`, Python compilation, and `bash -n run_cpu_audit.sh` passed. The benchmark preflight returned exit code 1 with `detected 2.00 GiB`; no benchmark output directory or quality decision was created. The full audit launcher was not executed.
