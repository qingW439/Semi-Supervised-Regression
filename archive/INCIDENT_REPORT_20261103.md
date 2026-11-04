# 2026-11-03 清理事故报告（c0 权重误删）

## 1. 结论

本次清理按 manifest 删除 99 个文件、释放 26.32 GB。其中 **1 个 P0 资产被误删**：

| 项 | 值 |
|---|---|
| 路径 | `Semi-Supervised-Regression-lcrankup/saved_models/vad_research/lcrankup_c0_s0/model_best.pth` |
| 大小 | 501,057,867 B |
| sha256 | `3275bd33f1ba95dada24b047e4cb84d91a49005a0d442381e2d987495cd8e88f` |
| 同 sha 副本 | 无（全盘仅 `lcrankup_m1_s0/model_best.pth` 是指向它的符号链接） |

## 2. 原因

`lcrankup_m1_s0/model_best.pth` 是一个 **32 字节符号链接**，目标为 `../lcrankup_c0_s0/model_best.pth`。
`delete_assets.py` 在校验时使用了 `Path.resolve()`，把链接解析成真实路径后再 `unlink()`，
因此删掉的是**链接目标 c0**，而不是链接本身。

正确行为应为：`lstat` 判断链接，拒绝或只删链接，绝不跟随 `resolve()`。

## 3. 影响评估

- `lcrankup_c0_s0/dev_metrics_best.json` 记录的 best step = **0**，
  Macro-MAE = 0.178264358（与 P1 基线 0.178264362 在 1e-9 内一致）。
  即该 best 权重 = **P1 初始化权重 + ARC 辅助头**，未经过任何有效训练步。
- P1 权重（`p1_roberta_lb500_s0/model_best.pth`）仍在 P0 中，**报告指标可复现**；
  但 c0 文件本身（含 ARC 头的完整 state_dict）**不可恢复**。
- 该文件不承担“唯一可复现证据”的角色：指标可由 P1 + 评估脚本复现，
  训练态复现需重新生成（见第 4 节）。

## 4. 恢复方案（待实例恢复内存后执行）

当前容器 `memory.max = 2 GiB`，训练与普通 `torch.load` 均被 OOM kill（rc=137），
**现在无法重训**。实例以 GPU 模式（≥16 GB 内存）重启后执行：

```bash
cd /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-lcrankup
/root/autodl-tmp/rankup_project/envs/rankup/bin/python scripts/vad_research/train_rankup3d.py \
  --budget 500 --seed 0 --steps 160 \
  --p1-dir /root/autodl-tmp/rankup_project/Semi-Supervised-Regression/saved_models/vad_research/p1_roberta_lb500_s0 \
  --out saved_models/vad_research/lcrankup_c0_s0 \
  --device cuda --arc-loss-ratio 0.05 --pl-calib none
```

预期：best 在 step 0 保存，`dev_metrics_best.json` 的 Macro-MAE 应回到 0.178264358（±1e-6）。
产出为新文件（新哈希），需同步更新 `cleanup_20261103/assets.jsonl` 并进行一次加载验证。

## 5. 防复发

1. `delete_assets.py` 已改为：凡 `lstat` 为符号链接者一律拒绝删除并报错；
2. 删除前对 manifest 每条做 `lstat` + `sha256` 双校验（本次 sha 校验通过，问题出在路径解析）；
3. 后续任何批量删除只允许以 `--apply` 且 `rejected == 0` 为前提；
4. 本文档保留在 `cleanup_20261103/` 内，与 manifest / deleted.jsonl 同级。

## 6. 本次清理结果

- 删除：99 文件 / 26.32 GB（P1 13.16 + P2 13.15）
- 磁盘：91 G → 65 G used，可用 32 G
- P0 加载验证：删除前 137/137 通过；删除后 136/137（唯一失败即上述 c0）
