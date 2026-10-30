# JACF 伪标签可靠性：机制穷举与 NO-GO 决策报告（2026-10-30）

## 0. 结论

**NO-GO。停止把“伪标签可靠性”作为 JACF 的创新组件。**

本轮不再只依赖 100 条 probe，而是用 EmoBank 官方 train split 中 **7562 条无标号文本的隐藏 VAD 标签**做诊断（仅分析，不参与训练），并补做了 **oracle 可靠性上限实验**：把真实伪标签误差直接用于 loss 加权/筛选。结果：

| 实验（lb=500, 160 steps） | 3-seed DEV Macro-MAE ↓ | 相对 none |
|---|---:|---:|
| none（固定 α=0.75，无可靠性） | **0.166938** ± 0.001469 | — |
| prop_dimnorm（真实可部署信号，3 seeds） | 0.167168 ± 0.001407 | **+0.000230（更差）** |
| **oracle_dimnorm（用隐藏标签的真实误差加权）** | **0.166834** ± 0.000994 | **−0.000103（仅 0.06%）** |
| oracle_filter（用隐藏标签筛掉 30% 最差伪标签，seed0） | 0.167528 | −0.000031 |
| oracle_inv（真实误差反比加权，seed0） | 0.167553 | −0.000006 |
| 320 steps：none / oracle | 0.167559 / 0.167553 | −0.000006 |

要点：

1. **可部署的可靠性信号全部无增益**；最好的 `prop_dimnorm` 三种子平均反而差 0.00023。
2. **即使把真实误差告诉模型（oracle），收益也只有 0.0001–0.0004**，远小于种子标准差 0.00147。
3. **oracle 硬筛选**（去掉 30% 真实最差伪标签）同样没有收益。
4. 因此问题不在“可靠性信号不够准”，而在 **per-sample 可靠性加权/筛选这个机制在本任务上没有可利用的 headroom**。

→ 论文中应删除可靠性主张；JACF 只保留“锚点先验融合（固定 α=0.75）”。不再实现 HPL/UCVME/GaussianMatch 的可靠性变体：它们都是 oracle 加权的受限版本，不可能超过 oracle 上限。

## 1. 本轮为什么能给出比之前更强的结论

之前的 NO-GO 分别来自：

- Q3 误差校准：80 条 meta 拟合误差预测器，在 100 条 probe 上 Spearman 方向错误；
- ARCL：100 条 probe 上局部残差不如 JACF；
- 梯度投影：5 个 batch 的 PCGrad 只挽回 0.000122；
- A0–A4：去掉可靠性（A4）与保留（A3）几乎相同。

这些实验的共同弱点是 **样本量小、只测“信号是否相关”，没有测“即使信号完美，训练能获益多少”**。本轮补上了后者：

- 全量 7562 条无标号文本的隐藏标签，统计功效远高于 100 条 probe；
- oracle 加权/筛选给出机制上限，而不是只比较代理信号；
- 三种子 + 与历史 none 逐 seed 对齐（0.167559 / 0.167994 / 0.165260，与历史 fixed α 完全一致）。

## 2. 机制矩阵（本轮及此前全部尝试）

| # | 机制/模块 | 信号来源 | 诊断结果 | 训练结果 | 判定 |
|---|---|---|---|---|---|
| 1 | 原 JACF `r = c·v0/(v0+v_f)` | 教师方差 + 锚点方差 + 冲突 | 全量池 Spearman(risk,error)=+0.073；capture 6.7% | A4≈A3 | NO-GO |
| 2 | Q3 误差校准 `r'` | meta 拟合教师/锚点/分歧 | probe Spearman +0.119（方向错） | 未启动 | NO-GO |
| 3 | ARCL 局部残差 | 教师 + top-k 邻域残差 | probe 不如 JACF | 未启动 | NO-GO |
| 4 | 梯度冲突 / PCGrad | 全参数梯度 | 全局余弦 +0.038±0.347，不稳定 | 单步挽回 0.000122 | NO-GO |
| 5 | `anchor_var`：`v0/(v0+v_a)` | 锚点邻域标签方差 | Spearman +0.191，AUC 0.591，capture 27.8% | seed0 0.167740 (+0.000181) | NO-GO |
| 6 | `prop_var`：`v0/(v0+α²v_a+(1−α)²v_t)` | 融合目标的真实方差 | 与 v_a 几乎相同（α=0.75 主导） | seed0 0.167833 (+0.000274) | NO-GO |
| 7 | `anchor_var` + 逐维归一化 | 同上，保持 V/A/D 总权重 | — | seed0 0.167645 (+0.000086) | NO-GO |
| 8 | `prop_var` + 逐维归一化 | 同上 | — | 3 seeds: 0.167470/0.168399/0.165635；均值 0.167168 (+0.000230) | NO-GO |
| 9 | `anchor_filter`：按 v_a 保留 70% | 锚点方差硬筛选 | filter capture 23.5% | seed0 0.167691 (+0.000132) | NO-GO |
| 10 | **oracle_filter**：按真实误差保留 70% | 隐藏标签 | — | seed0 0.167528 (−0.000031) | NO-GO |
| 11 | **oracle_dimnorm**：按真实误差反比加权 | 隐藏标签 | — | 3 seeds: 0.167172/0.167615/0.165716；均值 0.166834 (−0.000103) | NO-GO |
| 12 | **oracle 320 steps** | 隐藏标签 | — | 0.167553 vs 0.167559 | NO-GO |

结论：从“手工信号”到“学习信号”再到“完美 oracle”，12 种替换全部不能带来超过种子噪声的稳定增益。

## 3. 全量无标号池诊断（7562 条，隐藏标签仅用于分析）

### 3.1 伪标签误差结构

在 7562 条无标号文本上，用固定 α=0.75 的 checkpoint 生成 teacher / anchor / fused 预测，与隐藏标签比较（原始 [1,5] 尺度）：

| 目标 | Macro-MAE ↓ | V / A / D MAE |
|---|---:|---:|
| teacher（MC 均值） | 0.17453 | 0.1851 / 0.1812 / 0.1574 |
| anchor（kNN 加权） | 0.17437 | 0.1902 / 0.1826 / 0.1502 |
| fixed α=0.75 | **0.17130** | 0.1852 / 0.1796 / 0.1491 |
| oracle α*（逐样本最优融合） | **0.13740** | 0.1453 / 0.1462 / 0.1207 |
| best source（teacher/anchor 取优） | 0.14155 | 0.1502 / 0.1499 / 0.1246 |

- oracle α* 相对 fixed α 有 **19.8%** 的目标误差下降空间；
- 但 teacher 与 anchor 的误差相关高达 **0.85–0.86**，两个来源并不是独立证据；
- 误差变异系数 CV≈0.86–0.98，说明“逐样本误差差异”确实存在，不是常数。

### 3.2 信号筛选（逐元素，batch=64×3）

| 信号 | Spearman(risk,error) | AUC(高误差) | within-batch ρ | capture of oracle rank-weight gain (τ=8) |
|---|---:|---:|---:|---:|
| `v_a` 锚点方差 | +0.191 | 0.591 | +0.190 | **27.8%** |
| `neighbor_range` | +0.157 | 0.575 | +0.155 | 23.5% |
| `conf` 邻域相似度 | +0.111 | 0.549 | +0.114 | 12.3% |
| `r_jacf` 原可靠性 | +0.080 | 0.537 | +0.082 | 6.7% |
| `v_f` 原融合方差 | +0.083 | 0.541 | +0.085 | **5.2%** |
| `v_t` 教师 MC 方差 | +0.041 | 0.537 | +0.085 | 8.7% |

关键发现：**原 JACF 的 `v_f = v_t v_a/(v_t+v_a)` 是谐波方差，被较小的教师方差 `v_t` 主导，几乎屏蔽了锚点不确定性 `v_a`**。实测 `v_a` 均值 0.00267/0.00221/0.00167，而 `v_t` 只有 0.00058/0.00052/0.00049；`v_f` 因此接近 `v_t`。把 `v_f` 换成 `v_a` 或 `α²v_a+(1−α)²v_t` 后，可靠性信号从 capture 6.7% 提升到 27.8%（4 倍）。

这解释了“为什么原 JACF 的可靠性没用”，但**并没有让训练变好**——见下一节。

### 3.3 信号可学习性上限（隐藏标签监督，5 折 held-out）

用全部特征（v_t、v_a、conflict、conf、density、extremity、text_len 等 24 维）在 7562 条池内做 50/50 划分，训练误差预测器：

| 任务 | 结果 |
|---|---|
| 预测 per-element 伪标签误差（Ridge / HGB） | Spearman **0.224 / 0.231**，R² **0.075 / 0.077** |
| 预测“anchor 是否优于 teacher” | AUC **0.564** |
| 预测逐样本最优 α* | MAE **0.457**（接近常数 0.5） |

解释：

- **误差大小**有一定可预测性（R²≈7.5%），所以 `v_a` 之类的信号不是纯噪声；
- **误差方向/来源选择**几乎不可预测（AUC 0.56，α* MAE 0.457）。JACF 的可靠性只能利用“误差大小”，不能利用“该信谁”。

## 4. 训练验证

### 4.1 协议

- 数据：EmoBank lb=500，P1 初始化（每个 seed 用对应 `p1_roberta_lb500_s$s`），160 steps，batch 16/64，EMA 0.99，MC=5，k=20，τ=0.1。
- 模型：`Semi-Supervised-Regression-relstudy`，分支 `m1/reliability-study`。
- 指标：DEV Macro-MAE（原始 [1,5]），VRRC = kNN α=0.4 + scale 1.2；同时记录 raw。
- 基线 `none` 三种子：0.167559 / 0.167994 / 0.165260，与历史 fixed α 逐 seed 完全一致。

### 4.2 三种子对照

| 配置 | seed0 | seed1 | seed2 | mean | Δ vs none |
|---|---:|---:|---:|---:|---:|
| none | 0.167559 | 0.167994 | 0.165260 | **0.166938** | — |
| prop_dimnorm | 0.167470 | 0.168399 | 0.165635 | 0.167168 | +0.000230 |
| oracle_dimnorm | 0.167172 | 0.167615 | 0.165716 | 0.166834 | −0.000103 |

种子标准差 0.00147；机制效应 0.0001–0.0002，完全被种子噪声淹没。

### 4.3 单种子补充

| 配置 | seed0 DEV MAE | Δ vs none |
|---|---:|---:|
| anchor_var | 0.167740 | +0.000181 |
| prop_var | 0.167833 | +0.000274 |
| anchor_dimnorm | 0.167645 | +0.000086 |
| anchor_filter | 0.167691 | +0.000132 |
| oracle_inv | 0.167553 | −0.000006 |
| oracle_filter | 0.167528 | −0.000031 |
| oracle_inv_dimnorm | 0.167172 | −0.000387 |

## 5. 为什么 oracle 也没有用

1. **训练预算太短，逐样本权重没有时间发挥作用**：160 steps × 64 无标号/step = 10240 次访问，7562 条文本平均只被访问 **1.35 次**。per-sample 权重主要影响“哪条样本被多看一次”，而不是长期塑造表示。
2. **权重在 batch 内归一化**：`(rel·err).sum()/rel.sum()` 只保留 batch 内相对排序；全局可靠性水平被消掉。即使信号全局相关，batch 内排序噪声很大。
3. **L1 伪标签损失 + 固定 α=0.75**：目标本身已经由锚点主导，误差的 85% 与 teacher 共享；能通过重加权去掉的主要是噪声，而不是系统偏差。
4. **瓶颈是预测信息，不是目标噪声**：oracle α* 的 19.8% 目标误差空间主要来自“模型没有学到的真实变化”，不是“伪标签被污染”。把伪标签误差告诉模型，不会给它新的输入信息。

## 6. 决策

| 项目 | 决策 |
|---|---|
| JACF 可靠性模块 | **删除**；论文中作为负结果/消融，不作为贡献 |
| HPL、UCVME、GaussianMatch 的可靠性变体 | **不再实现**：它们都是 oracle 加权的受限版本，无法超过 oracle 上限 |
| JACF 主体 | 保留“锚点先验融合（固定 α=0.75）”，这是唯一有稳定 MAE 增益的部分 |
| M1 主模型 | fixed α=0.75（raw 0.169666 / VRRC 0.166938）；若要求 CCC 不退化，用 JACF 但删除可靠性主张 |
| 下一步 | 转入 M2：局部连续结构诊断与“局部仿射校准”，见 `M2_STRUCTURE_AND_LOCAL_AFFINE_20261030.md` |

## 7. 相关顶会/期刊工作核查（2024–2026）

| 工作 | 会议/期刊 | 机制 | 与本轮的关系 |
|---|---|---|---|
| RankUp | NeurIPS 2024 | 辅助 ranking 分类器 + RDA | 已复现；EmoBank 无明显优势，保留为历史对照 |
| HPL | NeurIPS 2025 | 异方差伪标签 + bi-level 不确定性学习 | 本轮 oracle 上限覆盖其机制；不单独实现 |
| GaussianMatch | CVPR 2026 | 多视图 Gaussian 一致性过滤 GCF + AGDS | 过滤类机制；oracle_filter 已 NO-GO |
| UCVME | AAAI 2023 | 异方差 + 不确定性一致性 + 变分模型集成 | 与 HPL 同族；oracle 上限覆盖 |
| SimRegMatch | Applied Soft Computing 2024 | 不确定性过滤 + 特征相似度校准 | 与 JACF 高度重叠；作为对比基线 |
| UCRC | Neurocomputing 2026 | 不确定性一致性 + 自适应伪标签校准 | 同上，重叠度高 |
| Hierarchical confidence + neighbor calibration | IEEE Access 2026 | 分层置信筛选 + 近邻校准 | JACF 的直接重叠工作，必须正面对比 |
| SSR-RCUS | AAAI 2026 | 近邻无标号样本排序保持 | 已复现；无稳定优势 |
| DKD | arXiv 2025 | 双流知识蒸馏 + 方差引导分布对齐 | 可靠性/蒸馏同族；oracle 上限覆盖 |
| CLSS | arXiv 2025 | 对比学习 + 谱排序 | 非伪标签路线，可作为后续 representation 方向 |
| CoGameReg | arXiv 2026 | 合作博弈 + kNN 误差下降筛选 | 筛选类；oracle_filter NO-GO |
| SpecRegMatch | IEEE Access 2024 | 单模型一致性 + 信息最大化 | 非伪标签一致性路线 |
| Noisy Proxy Covariates | ICML 2026 | 谱特征 + ridge，非伪标签 | 后续“无标号真实增益”对照 |
| FisherMatch | CVPR 2022 | 矩阵 Fisher 熵过滤 | 过滤类，早于本轮 |
| HACMatch | CVIU 2026 | hardness-aware curriculum | 课程类；oracle_filter 已否定固定筛选 |
| TNNR | 2021 | 成对差值 loop consistency | 一致性路线，非可靠性 |
| SSDKL | NeurIPS 2018 | 预测方差最小化 | 早期 SSR 基线 |

## 8. 复现

```bash
cd /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-relstudy
PY=/root/autodl-tmp/rankup_project/envs/rankup/bin/python

# 1) 全量隐藏标签诊断（仅分析）
$PY scripts/vad_research/reliability_ceiling.py \
  --budget 500 --seed 0 \
  --ckpt /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-source/saved_models/vad_research/alpha_fixed075_s0/model_best.pth \
  --data-root /root/autodl-tmp/rankup_project/data/emobank_processed \
  --raw-csv /root/autodl-tmp/rankup_project/data/emobank_source/EmoBank/corpus/emobank.csv \
  --pretrain-path /root/autodl-tmp/rankup_project/pretrained/roberta-base \
  --out-dir outputs --bootstrap 200 --device cuda

# 2) 信号打分
$PY scripts/vad_research/analyze_reliability.py \
  --npz outputs/reliability_ceiling_lb500_s0.npz \
  --out outputs/reliability_signals_lb500_s0.json

# 3) 三种子训练（none / prop_dimnorm / oracle_dimnorm）
for s in 0 1 2; do
  $PY scripts/vad_research/train_acvr.py --budget 500 --seed $s --steps 160 \
    --mode jacf --rel-mode prop_var --alpha-fixed 0.75 --rel-dimnorm \
    --p1-dir /root/autodl-tmp/rankup_project/Semi-Supervised-Regression/saved_models/vad_research/p1_roberta_lb500_s$s \
    --eval-knn-alpha 0.4 --eval-scale 1.2 --device cuda \
    --out saved_models/vad_research/rel_prop_dimnorm_s$s
done
```

注：oracle 相关运行使用 `--rel-mode oracle/oracle_filter` 和 `outputs/oracle_labels_all_train.npz`，**仅用于上限诊断，不能作为方法结果**。
