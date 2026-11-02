# 标签噪声、表示瓶颈与谱特征准入诊断（2026-11-02）

## 0. 结论

三项诊断合起来给出了一个比“再换一个模型”更明确的判断：

1. **P2 的 4 倍尺度错误已修复**：MAE 差值统一为原始 [1,5] 尺度；A-VRRC 的 NO-GO 结论不变。
2. **标签噪声是当前最大的瓶颈**：EmoBank 的 A/D 聚合标签可靠性很低（split-half 0.32/0.28），模型在 A/D 上的残差标准差已经低于聚合标签的噪声标准差；预测向均值收缩是噪声下的合理行为，不是单纯的模型缺陷。
3. **表示/回归头不是主要瓶颈**：折内 encoder 上的诚实 Ridge probe 与模型 raw 几乎相同；微调主要提升 CCC，而不是 MAE。
4. **无标签特征学习有相对收益，但当前谱特征实现不可直接采用**：S2（TRAIN+ULB 谱特征）显著优于 S1（仅 TRAIN 谱特征），ΔMAE −0.0050、ΔCCC +0.0318；但 S2 仍明显差于 S0（原始冻结特征 Ridge）和现有模型+VRRC。

因此：**不直接复现 GCLSS/完整谱网络；下一步优先做“标注者感知的分布/不确定性回归”或“可靠性感知的评价与学习”，而不是继续在点预测 MAE 上做后处理。**

---

## 1. P0：P2 统计尺度修复

之前 P2 表格使用原始 [1,5] 尺度，而 bootstrap 差值使用归一化 [0,1] 尺度，导致差值恰好缩小 4 倍。已修复 `bootstrap_p2.py`：

- MAE 差值乘以 4，统一为原始 [1,5]；
- CCC 不受尺度影响；
- JSON 增加 `mae_scale: "original_[1,5]"`。

修正后的 DEV 配对 bootstrap（vs VRRC）：

| 方法 | ΔMAE（原始 [1,5]） | 95% CI | ΔCCC | 95% CI |
|---|---:|---|---:|---|
| Global Affine | −0.002381 | [−0.003642, −0.001144] | −0.0269 | [−0.0309, −0.0226] |
| Local Intercept | −0.000124 | [−0.000944, +0.000775] | +0.0054 | [+0.0026, +0.0081] |
| Local Affine | −0.000336 | [−0.000845, +0.000133] | +0.0034 | [+0.0019, +0.0050] |

Local Affine 的 MAE 改善仍不显著（CI 跨 0），且 OOF cross 无增益；A-VRRC **NO-GO** 不变。

---

## 2. 标签聚合一致性：已复核

EmoBank 官方聚合并非“所有评分的简单平均”，而是：

1. **先过滤掉 V=A=D=1 的评分行**；
2. 再按 id 求均值；
3. 只保留 N>1 的样本；
4. 四舍五入到两位小数。

用该规则重算 500 条 TRAIN 的 VAD，与 `emobank.csv` 的最大绝对差为 **0.0064**，聚合规则验证通过。过滤掉的全 1 评分行共 9,939 行（reader+writer 合计）。

这一步很重要：如果不做全 1 过滤，会把“明确中性”的评分算进去，导致标签噪声被高估、且部分样本的聚合值对不上。

---

## 3. 标签噪声与标注一致性

500 条 TRAIN，reader+writer 合计平均 **9.26** 个评分/样本（5–12）。

| 维度 | 评分内 SD | ICC(1,1) | split-half r | Spearman-Brown | 聚合标签噪声 SD | 模型 OOF 残差 SD | 噪声/残差方差比 |
|---|---:|---:|---:|---:|---:|---:|---:|
| V | 0.405 | 0.331 | 0.681 | 0.810 | 0.158 | 0.188 | 0.70 |
| A | 0.516 | 0.093 | 0.316 | 0.480 | 0.185 | 0.152 | **1.49** |
| D | 0.420 | 0.079 | 0.279 | 0.436 | 0.161 | 0.136 | **1.41** |

- V 的聚合标签相对可靠（split-half 0.68）；A/D 很低（0.32/0.28），说明 **A/D 的“真值”本身就有很大不确定性**。
- A/D 的模型 OOF 残差 SD 已经**低于**聚合标签噪声 SD；V 的残差 SD 略高于噪声 SD。
- 模型不可能在 A/D 上把 MAE 大幅降低而不去拟合标签噪声；这正是 A/D 预测压缩、CCC 低、以及 VRRC 需要“放大方差换 CCC”的根本原因。

### 3.1 标注分歧与模型误差

| 维度 | Spearman(评分内 SD, 模型 |error|) |
|---|---:|
| V | **0.430** |
| A | **0.348** |
| D | **0.265** |

样本的标注分歧越大，模型误差越高，V 维尤其明显。这说明相当一部分“模型误差”其实是标注歧义，而不是模型没学到。

### 3.2 极端情绪更不可靠

| 维度 | 低强度组评分 SD | 高强度组评分 SD |
|---|---:|---:|
| V | 0.102 | **0.629** |
| A | 0.397 | **0.661** |
| D | 0.209 | **0.622** |

高强度组的评分分歧是低强度组的 3–6 倍。JACF 锚点融合在高强度组 MAE 变差，不能简单归因于“融合压缩了真实变化”；**高强度标签本身更模糊**，融合把预测拉向邻居均值可能反而更接近潜在均值。

### 3.3 解释边界

- ICC/split-half 不是严格的 MAE 理论上限；模型可以拟合特定聚合标签的噪声（A/D 残差 SD 低于噪声 SD 就是证据）。
- 但“A/D 标签可靠性远低于 V、极端组更不可靠、分歧与误差正相关”这三个结论是稳健的。
- 因此，**任何在 EmoBank A/D 上小于约 0.01–0.02 MAE 的“改进”都必须先排除标签噪声**；不能把点预测 MAE 的微小下降直接当作方法创新。

---

## 4. 表示瓶颈 probe（R0–R4）

5 折 OOF，同一 400/100 划分；Ridge 的 α 在每个 400 内部用 4 折 CV 选择，不使用折外 100 标签。MLP 未收敛（MAE 0.894），标记为无效，不作为证据。

| 编号 | 读出器 | OOF Macro-MAE | Macro-CCC | Macro-R² |
|---|---|---:|---:|---:|
| R0 | 冻结预训练 RoBERTa + Ridge | 0.182099 | 0.3714 | 0.2290 |
| R1 | 折内微调 encoder + Ridge | **0.181627** | **0.4328** | **0.2394** |
| R2 | 折内 encoder + MLP | 0.894308（未收敛，无效） | 0.0272 | −31.56 |
| R3 | 原模型 raw | 0.181740 | 0.4267 | 0.2157 |
| R4 | 原模型 + VRRC | **0.177537** | **0.4370** | **0.2493** |

解释：

- **R0 → R1**：微调显著提升 CCC（0.371→0.433），MAE 基本不变；监督微调主要增加了“排序/相关性”信息，而不是降低点误差。
- **R1 ≈ R3**：在同样的折内 encoder 上，线性 Ridge 与模型 3 维 MLP 头几乎相同（MAE 0.181627 vs 0.181740；CCC 0.4328 vs 0.4267）。**回归头不是主要瓶颈**。
- **R4** 仍最好，说明检索平滑提供额外收益，但这是校准，不是新的表示信息。

结合标签噪声：R1/R3 已经接近标签噪声允许的误差水平；换头、换线性读出不会带来实质提升。

---

## 5. 谱特征 S0/S1/S2 准入

固定冻结 RoBERTa 特征，Nyström RBF 谱特征（m=400），5 折 OOF；Ridge α 在 400 内部选；S1 与 S2 使用相同的 400 标签、相同的评价样本，唯一区别是 S2 的谱特征还使用了 7562 条无标签文本。

| 编号 | 特征学习数据 | OOF MAE | OOF CCC | OOF R² |
|---|---|---:|---:|---:|
| S0 | 不做谱变换（原始冻结特征 Ridge） | **0.182099** | **0.3714** | **0.2290** |
| S1 | TRAIN 400 | 0.190114 | 0.2130 | 0.1331 |
| S2 | TRAIN 400 + ULB 7562 | 0.185094 | 0.2448 | 0.1698 |

配对 bootstrap（3000 次，按样本重采样，原始 [1,5] 尺度）：

| 对比 | ΔMAE | 95% CI | ΔCCC | 95% CI |
|---|---:|---|---:|---|
| S2 − S1 | **−0.005021** | [−0.008421, −0.001749] | **+0.0318** | [+0.0053, +0.0592] |

逐维 ΔMAE（S2−S1）：V −0.00709、A −0.00511、D −0.00286。

解释：

- **无标签数据参与谱特征学习确实带来显著相对增益**：S2 在 MAE、CCC、R² 上都优于 S1，且三个维度方向一致。这是本轮唯一一个“无标签信息产生独立表示收益”的正面证据。
- **但谱特征本身损失了太多信息**：S2（0.1851/0.2448）仍明显差于 S0（0.1821/0.3714）和模型+VRRC（0.1775/0.4370）。Nyström 谱映射不是可直接采用的 M1。
- 因此结论是：**机制层面 GO（无标签特征学习有信号），实现层面 NO-GO（当前谱特征不可直接作为模型）**。

---

## 6. 研究决策

| 问题 | 决策 |
|---|---|
| 是否直接复现 GCLSS / 完整谱网络？ | **暂不**；S2 仍差于 S0，且标签噪声限制了绝对收益 |
| A-VRRC 是否继续？ | **停止**；P2 NO-GO 不变 |
| 是否继续在 EmoBank 上追点预测 MAE？ | **不**；A/D 标签噪声已接近/超过模型残差，微小 MAE 改进不可解释 |
| 下一步优先方向 | **标注者感知的分布/不确定性回归**，或**可靠性感知的评价与学习** |
| 谱特征的地位 | 作为“无标签表示学习有信号”的证据；若做，需加残差/对比结构以保留 S0 的信息，而不是替换特征 |

### 建议的新 M1 方向

**标注者感知的分布回归（annotator-aware distributional regression）**：

- 目标不再是单一聚合 VAD，而是每个样本的 **评分分布**（或均值 + 方差）。
- 用 500 条 TRAIN 的逐评分数据训练 heteroscedastic / distributional head：均值预测 + 不确定性预测；损失用 Gaussian NLL / CRPS / 分位数损失。
- 评价不只看 MAE/CCC，还看：
  - 不确定性校准（NLL、coverage、与评分内 SD 的相关性）；
  - 在 A/D 高分歧样本上的表现；
  - 用预测不确定性给伪标签加权（这次是有真实标注者方差作为监督信号的可靠性，而不是之前的启发式）。
- 与 JACF 的关系：JACF 仍负责锚点数值校正；分布头负责“哪些样本的标签本身不可靠”，两者可以形成一个统一目标：
  `L = L_sup(dist) + λ_u L_pseudo(dist, JACF target)`。
- 这比“再换一个校准器”更贴合当前数据，也更能解释 A/D 的压缩与极端组问题。

### 对旅游应用的含义

- 不能用单条评论的聚合 VAD 当真值来宣称模型提升；应至少双人标注，报告评分者一致性。
- 对 A/D 这类低可靠性维度，应报告不确定性/区间，而不是只报 MAE。
- 旅游评分（星级）不能替代 VAD；跨语言需要 XLM-R 等共享骨干，并保持监督/半监督协议一致。

---

## 7. 复现

```bash
cd /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-avrrc
PY=/root/autodl-tmp/rankup_project/envs/rankup/bin/python

# 标签噪声/标注一致性
$PY scripts/vad_research/label_noise_diagnostic.py \
  --data-root /root/autodl-tmp/rankup_project/data/emobank_processed \
  --emobank-root /root/autodl-tmp/rankup_project/data/emobank_source/EmoBank \
  --oof outputs/oof/oof_all.parquet \
  --dev-pred outputs/p2_honest/p2_dev_predictions.parquet \
  --out-dir outputs/diagnostics

# R0-R4 表示 probe
$PY scripts/vad_research/representation_probe.py \
  --data-root /root/autodl-tmp/rankup_project/data/emobank_processed \
  --oof-dir outputs/oof --manifest outputs/p0/fold_manifest.json \
  --pretrain-path /root/autodl-tmp/rankup_project/pretrained/roberta-base \
  --out outputs/diagnostics/representation_probe.json --device cuda

# S0/S1/S2 谱特征准入
$PY scripts/vad_research/spectral_pilot.py \
  --data-root /root/autodl-tmp/rankup_project/data/emobank_processed \
  --oof-dir outputs/oof --manifest outputs/p0/fold_manifest.json \
  --pretrain-path /root/autodl-tmp/rankup_project/pretrained/roberta-base \
  --out outputs/diagnostics/spectral_pilot.json --device cuda

# S2-S1 配对 bootstrap
$PY scripts/vad_research/bootstrap_spectral.py \
  --npz outputs/diagnostics/spectral_pilot.npz \
  --out outputs/diagnostics/spectral_bootstrap.json
```
