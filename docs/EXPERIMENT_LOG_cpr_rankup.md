# CPR-RankUp-VAD 实验记录（2026-10-29）

## 0. 结论

**NO-GO。**

CPR（用 SimRegMatch 校准的连续伪标签生成 ARC 伪排序）在 lb=500 三种子上：

- 未超过 P2 强监督（0.176460 vs 0.175589）；
- 相对原始 RankUp-3D 仅 +0.000786，且只有 1 个种子真正改善；
- 关闭无标签 ARC 后结果几乎不变（0.176598 vs 0.176460，ARC 独立贡献 ≈ 0.00014）；
- Macro-CCC 从 P2 的 0.4418 掉到 0.3798。

按预注册停止条件：**结束 RankUp 训练侧创新路线，不再往 CPR 上叠加 GCF/VRRC 制造正结果。**

## 1. 代码与协议

- worktree：`/root/autodl-tmp/rankup_project/Semi-Supervised-Regression-cpr`
- 分支：`m1/cpr-rankup-vad`，父提交 `ff72489`
- 唯一接口替换：无标签 ARC 的 pseudo-rank target/mask 生成
  - 原逻辑：教师 pair softmax → `p_cutoff=0.95` mask → argmax 硬标签
  - CPR：教师 K=5 次 dropout 前向 → SimRegMatch 不确定性过滤（前一批次 95 分位阈值）→ 标注 bank 相似性校准（β=0.6, τ=1.0）→ 逐维 pairwise 排序标签 + 联合 mask
- 复用 `vad-m0-simregmatch-3d` 分支的 `simregmatch3d/calibration.py` 与 `uncertainty_filter.py`
- 不改：RoBERTa、回归头、ARC 头、有标签回归/ARC 损失、RDA、optimizer、160 步协议
- 不启用：GCF、LA-PLC、VRRC

固定协议：P1 初始化、lb500、labeled 16 / unlabeled 64、ARC 权重 0.05、AdamW lr 5e-6 wd 0.01、160 步、每 32 步 DEV 选模。

## 2. 测试与 smoke

- `tests_vad/test_arc_pseudo_calibrator.py`：**7 passed**
  - 形状、排序方向与原 ARC 类别编码一致、交换反向、自比较/相等值/零有效对、detach、梯度回传到 encoder、dropout 生效
- 5-step smoke：DEV 0.177707；20-step smoke：0.174006
- CPR 诊断（20 步内）：阈值 ≈ 0.0035–0.0047，样本通过率 0.89–0.98，有效 pair 比例 0.78–0.95，不确定性均值 ≈ 0.0018–0.0022

smoke 只证明工程正确，不作为效果判断。

## 3. 三种子结果（DEV，lb=500，raw predictions）

| 配置 | Macro-MAE | Macro-CCC | Macro-R² | best step |
|---|---|---|---|---|
| C0 P2 supervised | **0.175589** | **0.4418** | 0.1750 | 128 |
| C1 原始 RankUp-3D（ARC+RDA） | 0.177246 | 0.3692 | 0.1492 | — |
| M1 CPR-RankUp-VAD | 0.176460 | 0.3798 | 0.1706 | — |
| A1 CPR − 无标签 ARC | 0.176598 | 0.4036 | 0.1729 | — |

逐种子：

| seed | C0 | C1 | M1 | A1 |
|---|---|---|---|---|
| 0 | 0.175543 | 0.178011 | 0.178010 | 0.178264 |
| 1 | 0.176286 | 0.179617 | 0.175319 | 0.175706 |
| 2 | 0.174937 | 0.174109 | 0.176052 | 0.175823 |

配对差值：

- C0 − M1：s0 **-0.002467**、s1 +0.000968、s2 **-0.001114**
- C1 − M1：s0 +0.000001、s1 +0.004298、s2 -0.001942
- A1 − M1：s0 +0.000254、s1 +0.000387、s2 -0.000229

## 4. Gate 判定

| 关卡 | 结果 | 判定 |
|---|---|---|
| 1. M1 三种子 raw MAE 优于 P2 | 均值 0.176460 > 0.175589，仅 1/3 种子改善 | **FAIL** |
| 2. M1 优于原始 RankUp，≥2 seeds 改善 | 均值 +0.000786，2/3 名义改善但 s0 仅 +1e-6 | 边缘 |
| 3. 去掉无标签 ARC 后提升消失/减弱 | A1 0.176598 vs M1 0.176460，差 0.00014 | **FAIL（ARC 无独立贡献）** |
| 4. CCC/R²/三维误差不明显退化 | CCC 0.3798 vs 0.4418（-0.062），R² 0.1706 vs 0.1750 | **FAIL** |

## 5. 解释

1. **ARC 本身在 VAD 上伤害 CCC**：C1 的 CCC 只有 0.3692，P2 是 0.4418；CPR 把 MAE 拉回一点，但 CCC 仍是 0.3798。
2. **无标签 ARC 几乎不起作用**：A1（关掉无标签 ARC）MAE 0.176598、CCC 0.4036，反而比 M1 的 CCC 好 0.024。说明 CPR 的伪排序监督没有提供有效梯度，甚至污染了表示。
3. **CPR 相对 C1 的 +0.0008 不足以抵消 CCC 的损失**，也追不上 P2。
4. 这与之前 LC-RankUp 的 NO-GO 一致：RankUp 训练侧的排序/校准接口在三维 VAD 上没有稳定收益。

## 6. 决策

- **停止 RankUp 训练侧创新线**：不再做 CPR + GCF/VRRC 的叠加，也不再把 ARC 作为主贡献。
- 保留唯一有正结果的方向：**检索校准（LA-PLC/VRRC）+ GCF 过滤**；其中消融已证明 VRRC 是主贡献（-0.00615），训练侧 GCF+LA-PLC 是次级（-0.00171）。
- 论文定位调整为“低标注 VAD 回归的检索校准预测框架”，RankUp-3D 作为近三年基座与负迁移证据；不再宣称训练侧排序接口创新。

## 7. 复现

```bash
cd /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-cpr
PY=/root/autodl-tmp/rankup_project/envs/rankup/bin/python

$PY -m pytest -q tests_vad/test_arc_pseudo_calibrator.py

# C1
$PY scripts/vad_research/train_rankup3d.py --budget 500 --seed 0 --steps 160 \
  --p1-dir /root/autodl-tmp/rankup_project/Semi-Supervised-Regression/saved_models/vad_research/p1_roberta_lb500_s0 \
  --arc-loss-ratio 0.05 --pl-calib rda --arc-pseudo-source original \
  --out saved_models/vad_research/c1_orig_s0 --device cuda

# M1
$PY scripts/vad_research/train_rankup3d.py --budget 500 --seed 0 --steps 160 \
  --p1-dir /root/autodl-tmp/rankup_project/Semi-Supervised-Regression/saved_models/vad_research/p1_roberta_lb500_s0 \
  --arc-loss-ratio 0.05 --pl-calib rda --arc-pseudo-source simreg_calibrated --mc-passes 5 \
  --out saved_models/vad_research/m1_cpr_s0 --device cuda

# A1：同 M1，加 --arc-ulb-ratio 0
```

## 8. 结果文件

- `saved_models/vad_research/c1_orig_s{0,1,2}/dev_metrics_best.json`
- `saved_models/vad_research/m1_cpr_s{0,1,2}/dev_metrics_best.json`（含 checkpoint）
- `saved_models/vad_research/a1_noarc_s{0,1,2}/dev_metrics_best.json`
