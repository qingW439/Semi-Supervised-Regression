# D1 标注诊断修正 + U0/U1/U2 分歧监督试点（2026-11-02）

## 0. 结论

1. **D1 修正了此前的统计错误**：旧版把“绝对误差 SD”当成“残差 SD”，并据此宣称 A/D 模型残差低于标签噪声。正确使用**有符号残差 SD** 后，模型残差仍大于标签噪声；噪声解释 V/A/D 残差方差的约 **34% / 62% / 62%**，不是 100%。A/D 标签仍明显比 V 不可靠，但模型没有到达噪声上限。
2. **Reader 与 Writer 的平均值没有系统性差异**；A/D 的低一致性主要来自**同一视角内部的评分者分歧**，不是 Reader/Writer 两种视角的系统偏移。
3. **U2（分歧辅助监督）没有提高 VAD 均值预测**：U2 MAE 0.184044 vs U0 0.183876（略差），U2-shuffle 0.186244；CCC 的 +0.011 没有伴随 MAE 改善，且 U1 的方差预测与真实评分 SD 的 Spearman 只有 0.090。
4. 因此：**分歧监督路线 NO-GO；不接入 JACF，不作为 M1 的第二组件。**

---

## 1. D1：标注诊断修正

### 1.1 聚合规则

官方规则：先过滤 V=A=D=1 的评分行，再按 id 求均值，保留 N>1。按此重算 500 条 TRAIN 与 gold 的最大绝对差 **0.0064**，验证通过；过滤后平均 **9.26** 个评分/样本（5–12）。

### 1.2 Reader / Writer / Combined

| 来源 | 维度 | 评分内 SD | ICC(1,1) | split-half r | 与 gold 的平均绝对差 |
|---|---|---:|---:|---:|---:|
| Reader | V | 0.379 | 0.367 | 0.542 | 0.122 |
| Reader | A | 0.466 | 0.106 | 0.207 | 0.153 |
| Reader | D | 0.392 | 0.084 | 0.151 | 0.114 |
| Writer | V | 0.315 | 0.317 | 0.466 | 0.122 |
| Writer | A | 0.425 | 0.103 | 0.176 | 0.152 |
| Writer | D | 0.342 | 0.067 | 0.130 | 0.116 |
| Combined | V | 0.405 | 0.331 | **0.681** | 0.001 |
| Combined | A | 0.516 | 0.093 | **0.316** | 0.001 |
| Combined | D | 0.420 | 0.079 | **0.279** | 0.001 |

- V 的合并标签可靠性中等（0.68）；A/D 很低（0.32/0.28）。
- Reader-only 和 Writer-only 的 A/D split-half 也只有 0.13–0.21，说明**低一致性不是“两个视角混在一起”造成的**。

### 1.3 Reader vs Writer 配对比较

| 维度 | Writer − Reader | 配对 t 检验 p | Reader/Writer 相关 | n |
|---|---:|---:|---:|---:|
| V | +0.007 | 0.637 | 0.650 | 500 |
| A | −0.028 | 0.119 | 0.281 | 500 |
| D | +0.010 | 0.497 | 0.285 | 500 |

- 三种维度的平均差都不显著；**没有系统性视角偏移**。
- A/D 的 Reader/Writer 相关只有 0.28，说明两种视角对同一文本的 A/D 排序也不一致；但差值均值接近 0，分歧是双向的。

### 1.4 正确的残差统计

| 维度 | Signed residual SD | Absolute error SD | RMSE | Bias |
|---|---:|---:|---:|---:|
| V | 0.273 | 0.188 | 0.276 | +0.046 |
| A | 0.236 | 0.152 | 0.238 | +0.030 |
| D | 0.205 | 0.136 | 0.210 | +0.046 |

- 旧版误把 **absolute error SD** 当作残差 SD；正确比较应使用 **signed residual SD** 或 RMSE。
- 标签噪声 SD（合并，iid 假设）：V 0.158、A 0.185、D 0.161；噪声/残差方差比：V 0.335、A 0.617、D 0.617。
- 解释：A/D 标签噪声能解释约 62% 的残差方差，V 约 34%；**模型仍有改进空间，但 A/D 的可改进空间被标签噪声显著压缩**。
- 这个比值依赖 iid 标注等假设，只作为探索性分解，不是严格 MAE 上限。

---

## 2. U0/U1/U2 分歧监督试点

### 2.1 设计

- 固定冻结 RoBERTa 特征；每折在 400 条训练样本上拟合 StandardScaler + PCA(64)。
- **U0**：Ridge 均值（α 在 400 内 4 折 CV 选择）+ 常数方差。
- **U1**：Ridge 均值固定；小型 variance head 用个体评分 Gaussian NLL 训练。
- **U2**：Ridge 均值 + 零初始化 residual mean head + variance head；共享 trunk；损失 `Huber + η·MSE(logvar, log s²)`；η ∈ {0.01,0.03,0.1,0.3} 在训练折内部 80 条验证集选择。
- **U2-shuffle**：同 U2，但在训练折内随机打乱分歧标签。
- 外层 5 折 OOF；折外 100 条只用于评价。所有分布指标针对**个体评分**计算。

> 说明：此前直接用小型 MLP 从零训练均值，出现严重欠拟合/过拟合（MAE 0.22–0.77），不能用于比较。最终采用 Ridge 锚定的稳定基线，确保 U0 与之前的 Ridge probe 可比。

### 2.2 均值结果（5 折 OOF，500 条）

| 方法 | Macro-MAE | Macro-CCC | ΔMAE vs U0 | ΔCCC vs U0 |
|---|---:|---:|---:|---:|
| U0（Ridge 均值） | **0.183876** | 0.3165 | — | — |
| U1（均值固定 + 方差头） | 0.183876 | 0.3165 | 0 | 0 |
| U2（分歧辅助） | 0.184044 | **0.3274** | +0.000168 | +0.0109 |
| U2-shuffle | 0.186244 | 0.3179 | +0.002368 | +0.0014 |

- U2 的 MAE 比 U0 **略差** 0.000168，没有达到“均值回归改善”的准入条件。
- CCC 的 +0.011 没有伴随 MAE 改善，且 U2-shuffle 没有复现同样的 CCC 增益；不能作为稳定收益。
- U2-shuffle 的 MAE 更差（0.1862），说明打乱分歧标签后没有正则化收益，U2 的微小 CCC 变化也不像“分歧信息”带来的。

### 2.3 分布结果

| 方法 | Individual NLL | CRPS | 80% Coverage | Spearman(pred σ, obs SD) |
|---|---:|---:|---:|---:|
| U0（常数方差） | −1.7203 | 0.2122 | 0.762 | −0.051 |
| U1（学习方差） | −1.1253 | 0.2443 | 0.903 | **0.090** |
| U2（分歧辅助） | 2.6170 | 0.7078 | 1.000 | −0.018 |
| U2-shuffle | 2.8523 | 0.7645 | 1.000 | −0.004 |

- U1 的方差预测与真实评分 SD 只有 **0.090** 的 Spearman，说明文本特征对“哪些文本分歧大”只有很弱的可预测性。
- U1 的 80% 覆盖率 0.903，区间偏宽；NLL 比常数方差差。
- U2 的方差头用 dispersion MSE 训练，不是 NLL，因此 NLL/覆盖率不可直接比较；其均值指标才是准入依据。

---

## 3. 决策

| 问题 | 决策 |
|---|---|
| Reader/Writer 是否有系统差异？ | 无显著均值差；A/D 低一致性主要是同视角内评分者分歧 |
| 文本能否预测评分分歧？ | 只有弱信号（U1 Spearman 0.090），不足以支撑稳定机制 |
| 分歧辅助监督能否提高 VAD 均值？ | **NO-GO**：U2 MAE 略差于 U0，U2-shuffle 也差 |
| 是否接入 JACF？ | **不接入**；均值准入未通过 |
| 是否继续调 variance head / η / 特征？ | **不继续**；先回到研究定位问题 |

### 为什么不再继续调这个方向

- 均值门控失败：U2 没有 MAE 收益，继续调参只会围绕 0.0001–0.001 的噪声。
- 分歧可预测性弱：U1 的 Spearman 0.09，说明“文本预测评分分歧”本身信号有限；即使 variance head 调得更好，也难转化为均值收益。
- 标签噪声仍高：A/D split-half 0.32/0.28；即使均值略有改善，也需要排除噪声才能解释。

### 下一步建议

1. **停止分歧监督路线**，不再把它作为 M1 的第二组件。
2. 论文定位回到：
   - JACF 锚点数值校正（有稳定 MAE 收益）；
   - 可靠性/校准/结构诊断的系统性负结果与 Pareto 分析；
   - 标注可靠性分析（Reader/Writer、ICC、极端组分歧）。
3. 如果还要做第二个组件，应选择**能提高可预测信息**的方向（更强的表示学习/更多独立标注），而不是继续在现有冻结特征上加辅助任务。
4. 旅游应用必须采用多标注者协议，报告一致性；不能用单条聚合 VAD 作为唯一真值。

---

## 4. 复现

```bash
cd /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-avrrc
PY=/root/autodl-tmp/rankup_project/envs/rankup/bin/python

# D1
$PY scripts/vad_research/label_noise_diagnostic_d1.py \
  --data-root /root/autodl-tmp/rankup_project/data/emobank_processed \
  --emobank-root /root/autodl-tmp/rankup_project/data/emobank_source/EmoBank \
  --oof outputs/oof/oof_all.parquet --out-dir outputs/diagnostics/d1

# U0/U1/U2 v2
$PY scripts/vad_research/disagreement_pilot_v2.py \
  --data-root /root/autodl-tmp/rankup_project/data/emobank_processed \
  --emobank-root /root/autodl-tmp/rankup_project/data/emobank_source/EmoBank \
  --manifest outputs/p0/fold_manifest.json \
  --pretrain-path /root/autodl-tmp/rankup_project/pretrained/roberta-base \
  --out-dir outputs/u_pilot_v2 --device cuda
```
