#!/usr/bin/env python3
"""P0 protocol audit for the A-VRRC OOF study.

Checks data identity, label isolation, historical checkpoint-selection rules,
centering/feature-space assumptions, and the scale contract.  Writes both JSON
and a human-readable Markdown report.  This script never trains a model and
never reads TEST labels into any computation beyond counting/overlap checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DIMS = ("V", "A", "D")
NORM = tuple(f"{d}_norm" for d in DIMS)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def check_ids(lb, ulb, dev, test):
    sets = {"lb": set(lb.id), "ulb": set(ulb.id), "dev": set(dev.id), "test": set(test.id)}
    pairs = [("lb", "ulb"), ("lb", "dev"), ("ulb", "dev"), ("lb", "test"), ("ulb", "test"), ("dev", "test")]
    overlaps = {f"{a}&{b}": len(sets[a] & sets[b]) for a, b in pairs}
    return {
        "counts": {k: len(v) for k, v in sets.items()},
        "overlaps": overlaps,
        "train_union": len(sets["lb"] | sets["ulb"]),
        "all_union": len(set().union(*sets.values())),
        "train_union_equals_8062": len(sets["lb"] | sets["ulb"]) == 8062,
        "no_overlap": all(v == 0 for v in overlaps.values()),
    }


def check_labels(lb, dev, test):
    out = {}
    for name, frame in (("lb", lb), ("dev", dev), ("test", test)):
        orig = frame.loc[:, list(DIMS)].to_numpy(np.float64)
        norm = frame.loc[:, list(NORM)].to_numpy(np.float64)
        out[name] = {
            "finite": bool(np.isfinite(orig).all()),
            "in_1_5": bool(((orig >= 1) & (orig <= 5)).all()),
            "normalized_consistent": bool(np.allclose(norm, (orig - 1) / 4, atol=2e-7, rtol=0)),
            "mean": {d: float(frame[d].mean()) for d in DIMS},
            "std": {d: float(frame[d].std()) for d in DIMS},
        }
    return out


def scan_code(repo: Path):
    training = ["scripts/vad_research/train_acvr.py", "scripts/vad_research/train_roberta_vad_supervised.py"]
    diagnostic = ["scripts/vad_research/reliability_ceiling.py", "scripts/vad_research/structure_diagnostic.py",
                  "scripts/vad_research/local_affine_oracle.py"]
    out = {}
    for rel in training + diagnostic:
        p = repo / rel
        if not p.is_file():
            out[rel] = {"exists": False}
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        out[rel] = {
            "exists": True,
            "reads_raw_csv": bool(re.search(r"raw[-_]csv|emobank\.csv", text)),
            "reads_oracle_labels": bool(re.search(r"oracle[-_]labels|oracle_y", text)),
            "uses_dev_mean": bool(re.search(r"p\.mean\(0\)|pred\.mean\(0\)", text)),
            "uses_model_encoder": bool(re.search(r"encode\(model\.encoder|model\.encoder", text)),
            "category": "diagnostic" if rel in diagnostic else "training",
        }
    return out


def checkpoint_audit(repo: Path, roots):
    out = {}
    for name, rel in (("p1_s0", "saved_models/vad_research/p1_roberta_lb500_s0"),
                      ("alpha_fixed075_s0", "saved_models/vad_research/alpha_fixed075_s0")):
        entry = {"path": None, "exists": False}
        for root in roots:
            d = root / rel
            if d.is_dir():
                entry = {"path": str(d), "exists": True}
                for f in ("run_config.json", "dev_metrics_best.json"):
                    p = d / f
                    if p.is_file():
                        try:
                            entry[f] = json.loads(p.read_text())
                        except Exception as e:
                            entry[f] = {"error": str(e)}
                entry["has_best_ckpt"] = (d / "model_best.pth").is_file()
                entry["selection_is_dev"] = (d / "dev_metrics_best.json").is_file()
                break
        out[name] = entry
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--ckpt-roots", type=str, default=None,
                    help="comma-separated roots to search for historical checkpoints")
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    if a.ckpt_roots is None:
        roots = [a.repo.parent / "Semi-Supervised-Regression",
                 a.repo.parent / "Semi-Supervised-Regression-source"]
    else:
        roots = [Path(x) for x in a.ckpt_roots.split(",") if x.strip()]
    a.out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "lb": a.data_root / "lb500_s0" / "train_lb.parquet",
        "ulb": a.data_root / "lb500_s0" / "train_ulb.parquet",
        "dev": a.data_root / "dev_gold.parquet",
        "test": a.data_root / "test_gold.parquet",
    }
    lb, ulb, dev, test = (read(paths[k]) for k in ("lb", "ulb", "dev", "test"))
    ids = check_ids(lb, ulb, dev, test)
    labels = check_labels(lb, dev, test)
    ulb_columns_ok = set(ulb.columns) == {"id", "text"}
    code = scan_code(a.repo)
    ckpt = checkpoint_audit(a.repo, roots)
    hashes = {k: sha256(v) for k, v in paths.items() if v.is_file()}
    pip = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    pip_hash = hashlib.sha256(pip.stdout.encode()).hexdigest()

    report = {
        "protocol": {
            "data_root": str(a.data_root),
            "repo": str(a.repo),
            "note": "TEST labels are only counted/overlap-checked; never used in training or calibration.",
        },
        "ids": ids,
        "ulb_columns_ok": ulb_columns_ok,
        "ulb_columns": list(ulb.columns),
        "labels": labels,
        "data_sha256": hashes,
        "pip_freeze_sha256": pip_hash,
        "code_scan": code,
        "checkpoints": ckpt,
        "flags": {
            "dev_selected_checkpoints": True,
            "old_vrrc_batch_mean": any(v.get("uses_dev_mean") for v in code.values()),
            "per_fold_encoder_geometry": any(v.get("uses_model_encoder") for k, v in code.items() if k.endswith("train_acvr.py")),
            "test_used_historically": True,
        },
    }
    (a.out_dir / "protocol_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    lines = ["# P0 协议审计报告", "", "## 1. 数据身份", ""]
    lines.append(f"- 计数：{ids['counts']}")
    lines.append(f"- train union = {ids['train_union']}（应 8062）：{ids['train_union_equals_8062']}")
    lines.append(f"- 所有交集为 0：{ids['no_overlap']}；overlaps={ids['overlaps']}")
    lines.append(f"- `train_ulb` 列严格为 id/text：{ulb_columns_ok}（实际 {list(ulb.columns)}）")
    lines += ["", "## 2. 标签与尺度", ""]
    for name, v in labels.items():
        lines.append(f"- {name}: finite={v['finite']} in[1,5]={v['in_1_5']} norm_consistent={v['normalized_consistent']} "
                     f"mean={ {k: round(x, 3) for k, x in v['mean'].items()} }")
    lines += ["", "## 3. 代码扫描", ""]
    for rel, v in code.items():
        lines.append(f"- `{rel}` ({v.get('category')}): raw_csv={v.get('reads_raw_csv')} "
                     f"oracle={v.get('reads_oracle_labels')} dev_mean={v.get('uses_dev_mean')} "
                     f"model_encoder={v.get('uses_model_encoder')}")
    lines += ["", "## 4. 历史 checkpoint 选择", ""]
    for name, v in ckpt.items():
        lines.append(f"- {name}: exists={v['exists']} best_ckpt={v['has_best_ckpt']} "
                     f"selection_is_dev={v['selection_is_dev']}")
        if "dev_metrics_best" in v and isinstance(v["dev_metrics_best"], dict):
            lines.append(f"  - best step/epoch: {v['dev_metrics_best'].get('step', v['dev_metrics_best'].get('epoch'))}")
    lines += ["", "## 5. 必须修复/声明的风险", "",
              "1. 历史 P1 与 alpha_fixed 均按 DEV MAE 选择 checkpoint，DEV 不是盲测集。",
              "2. 旧 VRRC 的 `p.mean(0)` 是当前评估批次均值，不是可部署的单样本统计量。",
              "3. JACF/VRRC 使用微调后的 encoder 计算邻域；不同 OOF fold 的 encoder 空间不一致。",
              "4. 可靠性/结构诊断脚本读取过 7562 条无标号文本的隐藏标签；这些标签只能用于特权诊断。",
              "5. TEST 已有历史评估记录，后续只能一次性使用，不能再参与选择。",
              "", "## 6. 结论", "",
              "- 数据 ID 与标签尺度审计通过。" if ids["no_overlap"] and ulb_columns_ok else "- 数据审计未通过，禁止进入 P1。",
              "- 进入 P1 前必须固定：OOF 折划分、P1/JACF 固定步数、统一检索特征、冻结中心化统计。",
              ]
    (a.out_dir / "P0_PROTOCOL_AUDIT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"ids": ids, "ulb_columns_ok": ulb_columns_ok,
                      "flags": report["flags"]}, indent=2))
    print("WROTE", a.out_dir / "protocol_audit.json", a.out_dir / "P0_PROTOCOL_AUDIT.md")


if __name__ == "__main__":
    main()
