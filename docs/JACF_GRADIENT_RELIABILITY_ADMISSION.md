# JACF 梯度可靠性准入报告（2026-10-30）

## 0. 结论：NO-GO

固定 α=0.75、rel=1 的 JACF 上，对 supervised L1 与 unsupervised L1 做全参数梯度审计，并比较普通 AdamW 与 PCGrad 单步更新：

- 全局余弦 **+0.038 ± 0.347**，5 个 batch 中 2 个为负、3 个为正——**不存在稳定的全局梯度冲突**；
- 负内积参数张量比例 0.439 ± 0.247（0.149–0.751），batch 依赖；
- 逐维：V +0.183（对齐），A −0.107、D −0.173（轻度冲突）；
- 普通 AdamW 单步使留出监督 MAE 平均 **+0.002074**；PCGrad 为 **+0.001952**，只挽回 **0.000122（5.9%）**；
- PCGrad 在 5/5 trial 上略好，但更新向量余弦 0.987，参数变化几乎相同。

→ **梯度冲突不是本任务上稳定、可利用的可靠性信号；PCGrad 的收益可忽略，且它本身不是新方法。不启动正式训练。**

## 1. 协议与公平性

- checkpoint：`alpha_fixed075_s0/model_best.pth`（fixed α=0.75，rel=1；best step 96）。
- 数据：TRAIN lb500 中固定划分 400 训练池 / 100 留出评价；评价样本不进入本步 batch，也不进入锚点库。
- 注意：fixed α 模型在全部 500 上训练过，因此这 100 条是“本步留出”，不是严格样本外；这是单步审计的限制。
- 教师：由于 checkpoint 未保存 EMA 状态，诊断用模型副本（eval）作为教师；这是审计近似，不是正式训练协议。
- 每 trial：同一初始模型副本、同一 batch、同一 AdamW 超参（lr 5e-6, wd 0.01）；普通与 PCGrad 各一份 fresh optimizer；评价前后 MAE。
- 未使用 DEV/TEST。

## 2. 梯度冲突（5 trials）

| 指标 | 均值 ± 标准差 | 逐 trial |
|---|---|---|
| 余弦(g_sup, g_unsup) | **+0.038 ± 0.347** | -0.553 / +0.227 / +0.445 / +0.199 / -0.126 |
| 负内积张量比例 | 0.439 ± 0.247 | 0.751 / 0.149 / 0.159 / 0.488 / 0.647 |
| 范数比 ‖g_sup‖/‖g_unsup‖ | 1.164 ± 0.370 | 0.994 / 0.825 / 0.991 / 1.133 / 1.877 |

逐维：

| 维度 | 余弦均值 | 范数比均值 |
|---|---|---|
| V | +0.183 | 0.913 |
| A | **−0.107** | 2.181 |
| D | **−0.173** | 1.685 |

解释：

- 全局层面没有一致的冲突方向，不能把“伪监督梯度与真实监督冲突”当成稳定现象。
- A、D 两维有轻度冲突，且监督梯度范数明显大于伪监督（2.18/1.69），说明伪监督在这两维上的更新占比更小。
- V 维基本对齐，说明三维不应共享同一个可靠性结论。

## 3. PCGrad 单步对照

| 指标 | 普通 AdamW | PCGrad |
|---|---|---|
| 留出 MAE 变化（均值） | +0.002074 | +0.001952 |
| 逐 trial 变化 | +0.000668 / +0.002851 / +0.004383 / +0.001516 / +0.000953 | +0.000169 / +0.002850 / +0.004382 / +0.001489 / +0.000870 |
| 更新范数 | 0.04630 | 0.04631 |
| 与普通更新的余弦 | — | 0.987 |

- PCGrad 5/5 trial 略好，但平均只挽回 0.000122，相当于单步损害的 5.9%；
- 更新向量几乎相同（余弦 0.987），说明投影只在少数参数上起作用；
- 单步本身会让留出 MAE 变差（SGD 噪声/过拟合），PCGrad 只能稍微减缓，不能逆转。

## 4. 与“可靠性”的关系

1. **梯度冲突不是稳定信号**：2/5 全局负、3/5 全局正；用它做样本级/维度级可靠性会把 batch 噪声当成机制。
2. **PCGrad 不是新方法**：即使有效，也只能作为已知优化技巧，不能作为 JACF 的可靠性创新。
3. **A/D 的轻度冲突值得记录**，但监督梯度占优，说明当前伪监督在这两维上的影响有限；这与 A1（仅锚点）MAE 更好、JACF CCC 更好的结果一致——问题仍在“来源权重”，不在梯度投影。
4. 结合 Q3（误差校准 NO-GO）、ARCL（残差 NO-GO）、A4≈A3（可靠性无效）：**可靠性模块在本任务上没有可验证的独立贡献**。

## 5. 决策

| 选项 | 结论 |
|---|---|
| 用 PCGrad 作为可靠性模块 | **NO-GO**：收益 0.00012，且非新方法 |
| 继续寻找梯度类可靠性信号 | 不建议：全局冲突不稳定，逐维差异不足以支撑新机制 |
| M1 定型 | 固定 α=0.75 或 JACF（按 MAE/CCC 取舍） |
| 下一步 | M1 冻结 → 250/500/1000 三种子 → TEST → M2 VRRC 诊断 |

## 6. 复现

```bash
cd /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-grad
PY=/root/autodl-tmp/rankup_project/envs/rankup/bin/python
$PY -m pytest -q tests_vad/test_gradient_audit.py

$PY scripts/vad_research/gradient_audit.py \
  --ckpt /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-source/saved_models/vad_research/alpha_fixed075_s0/model_best.pth \
  --data-root /root/autodl-tmp/rankup_project/data/emobank_processed \
  --pretrain-path /root/autodl-tmp/rankup_project/pretrained/roberta-base \
  --out outputs/gradient_audit.json --trials 5
```

计算开销：5 trials，每 trial 约 8 次全参数 backward（2 总 + 6 逐维）+ 4 次小 eval；单卡 4090 约 1–2 分钟，显存约 3–4 GB。
