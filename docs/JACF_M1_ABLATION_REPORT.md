# JACF M1 消融与归因报告（2026-10-30）

## 0. 结论：JACF 的“联合估计”不成立（NO-GO）

固定 lb=500、160 步、P1 初始化、EMA 0.99、MC=5、k=20、τ=0.1、原 VRRC，三种子：

| 机制 | raw MAE | raw CCC | raw R² | VRRC MAE | VRRC CCC | VRRC R² |
|---|---|---|---|---|---|---|
| A0 仅多视图 | 0.173241 | 0.4411 | 0.1971 | 0.168855 | 0.4469 | 0.2313 |
| **A1 仅锚点** | **0.169460** | 0.4253 | **0.2241** | **0.166864** | 0.4327 | **0.2432** |
| A2 顺序 GCF+LA-PLC | 0.171189 | 0.4328 | 0.2137 | 0.167716 | 0.4395 | 0.2399 |
| A3 完整 JACF | 0.172091 | 0.4387 | 0.2066 | 0.168069 | 0.4447 | 0.2370 |
| A4 JACF 去可靠性 | 0.171961 | 0.4431 | 0.2069 | 0.168135 | 0.4474 | 0.2354 |

配对差值（baseline − JACF；正=JACF 更好）：

| 对比 | VRRC MAE 逐种子 | 均值 | raw 均值 |
|---|---|---|---|
| A1 − A3 | -0.000997 / -0.001506 / -0.001112 | **-0.001205** | **-0.002631** |
| A2 − A3 | -0.000470 / -0.000075 / -0.000515 | -0.000353 | -0.000902 |
| A0 − A3 | +0.000801 / +0.000386 / +0.001172 | +0.000786 | +0.001149 |
| A4 − A3 | +0.000420 / -0.000223 / +0.000001 | +0.000066 | -0.000130 |

判定：

1. **JACF 只赢了“仅多视图”（A0）**，说明多视图单独不够。
2. **JACF 输给“仅锚点”（A1）**，三种子全部为负，raw 差 0.00263；联合融合把锚点证据稀释了。
3. **JACF 输给顺序组合（A2）**，三种子全部为负（幅度小）。
4. **JACF ≈ A4（去掉可靠性）**：可靠性加权本身没有贡献。

→ 按预注册标准：**JACF 的联合估计不优于单来源，也不优于旧顺序组合；NO-GO。**

## 1. 可靠性诊断（严格 TRAIN 内部 320/80/100）

用参考模型 + meta 80 锚点 + probe 100 评估 JACF 的 r：

- Spearman(r, 真实伪标签绝对误差) = **-0.080**；
- 按 r 四分位的平均误差：0.04169 / 0.04189 / 0.04239 / 0.03970（非单调）；
- 高 r 且高误差样本占高误差样本的 **28.0%**（随机基线 25%）；
- α 均值：V 0.174 / A 0.252 / D 0.343（D 更依赖锚点）；
- r 均值：V 0.463 / A 0.491 / D 0.496。

结论：**r 几乎不能区分真实误差**，与 A4≈A3 的结果一致。JACF 的“目标+可靠性联合”主张没有实验支持。

## 2. 审计说明

- 独立 worktree：`Semi-Supervised-Regression-jacfabl`，分支 `m1/jacf-ablation`，父提交 `376baf2`。
- 新增 `jacf_modes.py`（A0–A4 可切换目标），`train_acvr.py` 同时输出 raw 与 VRRC 指标；8 个单元测试通过。
- 原有 JACF 运行（`jacf_a1_vrrc_s*`）使用 `--no-save-ckpt`，**没有保留 checkpoint**，无法事后补 raw；因此本轮用同一 trainer 重跑了 A0–A4，协议完全一致，checkpoint 保留在 `abl_*_s0`。
- B1/B3 同样没有 checkpoint，raw 无法从旧运行恢复；本报告以 A0–A4 的同代码矩阵为准。
- 所有指标在同一 checkpoint 上同时计算 raw 与 VRRC；选模仍按原 VRRC DEV MAE，未为 raw 重选步数。

## 3. 对论文含义

- **JACF 不能作为“联合估计优于单来源”的核心创新点**：MAE 输给 A1/A2，可靠性也不跟踪误差。
- 但 JACF 仍是**唯一同时改善 MAE 与 CCC** 的配置（相对 P2）；A1 的 MAE 更好，代价是 CCC 下降 0.0091，接近“明显损害”的边界。
- 因此存在两种诚实的论文写法：
  1. 若主指标是 MAE、且允许 CCC 小幅退化：M1 = **A1 仅锚点 + VRRC**（0.166864），但创新点变成“锚点伪监督”，与 LA-PLC/SimRegMatch 更接近，必须正面对比。
  2. 若要求 MAE 与 CCC 同时不退化：M1 仍用 JACF（0.168069 / 0.4447），但贡献只能写成“锚点主导的融合 + 方差恢复”，不能宣称联合估计优越；可靠性模块应删除（A4 等价）。

## 4. 决策

| 选项 | 内容 | 评价 |
|---|---|---|
| A | M1 = A1 仅锚点 + VRRC | MAE 最好（0.166864），CCC 0.4327（低于 P2 0.0091）；创新边界最弱 |
| B | M1 = JACF + VRRC，但删掉“联合估计”主张 | MAE/CCC 双改善；需承认 A4≈A3、可靠性无效 |
| C | 停止 M1 训练侧创新，直接做 M2 推理校准 | 最保守，但放弃已有效果 |

建议：**B 作为模型、A 作为“MAE 上限”对照**；论文中把 M1 的贡献改写为“锚点主导的伪监督 + 方差恢复校准”，并把“联合估计/可靠性”作为负结果写进消融。若导师要求必须有训练侧新机制，则选 A，并补 A1 的 250/1000 与 TEST。

## 5. 复现

```bash
cd /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-jacfabl
PY=/root/autodl-tmp/rankup_project/envs/rankup/bin/python
$PY -m pytest -q tests_vad/test_jacf_modes.py

for mode in multi anchor seq jacf no_rel; do
  for s in 0 1 2; do
    $PY scripts/vad_research/train_acvr.py --budget 500 --seed $s --steps 160 \
      --mode $mode --p1-dir /root/autodl-tmp/rankup_project/Semi-Supervised-Regression/saved_models/vad_research/p1_roberta_lb500_s$s \
      --eval-knn-alpha 0.4 --eval-scale 1.2 --device cuda \
      --out saved_models/vad_research/abl_${mode}_s$s
  done
done
```
