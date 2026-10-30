# M2：JACF/VRRC 的局部结构压缩与局部仿射校准上限（2026-10-30）

## 0. 结论

**局部连续结构压缩是真实存在的，但“全局方差恢复”不是正确的修复方式；局部仿射校准有 headroom，但必须用可部署的 TRAIN/OOF 锚点实现，不能继续调全局 scale。**

三个核心事实：

1. **压缩在 JACF 之前就存在，而且非常严重。** 在 7562 条无标号文本上，teacher 的局部差异斜率（pred diff 对 true diff 回归）只有 **0.148 / 0.076 / 0.027**（V/A/D），anchor 只有 **0.092 / 0.030 / 0.014**。fixed α=0.75 进一步降到 **0.106 / 0.042 / 0.017**。预测方差只有真实方差的 **8.4% / 4.2% / 3.7%**。
2. **JACF 锚点校正以“平均 MAE”换“极端样本”**：高情绪强度组的 MAE 从 teacher 的 0.0935/0.0839/0.0812 升到 fixed α 的 0.1065/0.0914/0.0831（V/A/D），而 low/mid 组改善。校正只有约 50–56% 的样本真正变好，校正量与真实增益的 Spearman 仅 −0.013 / +0.023 / +0.180。
3. **旧版 VRRC 并没有“恢复方差”**：kNN 校准把斜率从 raw 的 0.234/0.091/0.072 进一步压到 0.229/0.087/0.063；scale 1.2 只是把 CCC 拉回来一点。scale sweep 显示 MAE 与 CCC 严格权衡：
   - s=1.0：MAE **0.165713**，CCC 0.4065
   - s=1.2：MAE 0.168019，CCC **0.4427**
   - s=1.5：MAE 0.176781，CCC **0.4734**
   - 全局线性 oracle：MAE **0.164755**，CCC 0.3914

**但局部仿射 oracle 给出了真正的联合增益空间**：在 VRRC 基础上混入 25–50% 的“邻域局部线性校准”，可以在 DEV 上同时超过 VRRC：

| DEV 校准 | Macro-MAE ↓ | Macro-CCC ↑ | 说明 |
|---|---:|---:|---|
| raw | 0.170725 | 0.4403 | 学生原始预测 |
| VRRC（kNN α=0.4 + scale 1.2） | 0.168019 | 0.4427 | 现有做法 |
| VRRC + 0.5·local_lin(k=20) | **0.163646** | **0.4440** | **同时优于 VRRC** |
| VRRC + 0.25·local_lin(k=20) | 0.164978 | **0.4455** | MAE/CCC 双改善 |
| VRRC + 0.5·local_lin + 0.25·local_var | 0.167693 | **0.4693** | CCC 大幅提升，MAE 略优 |

注意：local_lin / local_var 使用 DEV 邻居的真实标签（不含自身），且混合系数在 DEV 上网格搜索，因此上表是 **oracle 上限**，不是可报告的方法结果。它的意义是：**局部仿射方向有真实 headroom，值得做可部署版本**；而继续调全局 scale 已经没有空间。

## 1. 诊断协议

- checkpoint：`alpha_fixed075_s0/model_best.pth`（fixed α=0.75，lb=500，best step 96）。
- 无标号池：7562 条，隐藏 VAD 标签仅用于分析，不参与训练。
- 局部对：2000 条随机 anchor，各取特征空间 top-5 邻居，约 1 万对；对每维回归 `pred_i − pred_j ~ true_i − true_j`，斜率 1 为理想。
- DEV：1000 条，raw 与 VRRC（kNN k=20、τ=0.1、α=0.4、c0=0.3、scale 1.2）对比；DEV 局部对取 500 个 anchor、top-5。
- 极端组：按 `|y−3|` 的 20%/60%/20% 分 low/mid/high。
- 所有数字在原始 [1,5] 尺度上报告（斜率/方差比除外）。

## 2. 无标号池：压缩与极端样本

### 2.1 局部斜率与方差比

| 方法 | V slope | A slope | D slope | V var ratio | A var ratio | D var ratio |
|---|---:|---:|---:|---:|---:|---:|
| teacher | 0.148 | 0.076 | 0.027 | 0.178 | 0.125 | 0.140 |
| anchor | 0.092 | 0.030 | 0.014 | 0.077 | 0.038 | 0.033 |
| fixed α=0.75 | 0.106 | 0.042 | 0.017 | 0.084 | 0.042 | 0.037 |

- 三种方法的斜率都远小于 1，说明预测对“情绪相近的文本”几乎给出相同值；
- anchor（kNN 加权）比 teacher 更平滑，fixed α 把 75% 权重给 anchor，因此进一步压缩；
- V 维的斜率最高（0.15），A/D 几乎为 0（0.03–0.08），与 A/D 的 CCC 最低一致。

### 2.2 极端组 MAE

| 方法 | V low/mid/high | A low/mid/high | D low/mid/high |
|---|---|---|---|
| teacher | 0.0266 / 0.0362 / **0.0935** | 0.0234 / 0.0363 / **0.0839** | 0.0209 / 0.0326 / **0.0812** |
| anchor | 0.0181 / 0.0359 / 0.1112 | 0.0160 / 0.0352 / 0.0941 | 0.0180 / 0.0297 / 0.0838 |
| fixed α | 0.0189 / 0.0350 / 0.1065 | 0.0168 / 0.0347 / 0.0914 | 0.0173 / 0.0297 / 0.0831 |

- 校正显著改善 low/mid（V mid 0.0362→0.0350，A 0.0363→0.0347），但 **high 组全面变差**（V +0.0130，A +0.0075，D +0.0019）；
- 这就是“平均 MAE 变好、极端情绪被拉向均值”的直接证据。

### 2.3 校正是否瞄准了真实误差

| 维度 | 校正后变好比例 | 平均增益 | 平均 |Δ| | Spearman(|Δ|, gain) |
|---|---:|---:|---:|---:|
| V | 50.5% | **−0.00015** | 0.0159 | −0.013 |
| A | 51.7% | +0.00036 | 0.0133 | +0.023 |
| D | 56.0% | +0.00212 | 0.0123 | +0.180 |

- V 维平均增益甚至是负的：校正幅度不小（0.016），但方向近似随机；
- D 维最好，但 Spearman 也只有 0.18；
- 说明 **anchor 校正在“降低平均误差”和“保留真实极端变化”之间存在系统性冲突**，而不是简单的“校正不足”。

## 3. DEV：raw vs VRRC

### 3.1 指标

| 方法 | Macro-MAE | Macro-CCC | V CCC | A CCC | D CCC |
|---|---:|---:|---:|---:|---:|
| raw | 0.170725 | 0.4403 | 0.663 | 0.387 | 0.271 |
| VRRC s=1.2 | **0.168019** | **0.4427** | 0.673 | 0.398 | 0.257 |

VRRC 的 MAE 与 CCC 都略优于 raw，但 D 维 CCC 反而下降（0.271→0.257）。

### 3.2 结构

| 方法 | V slope | A slope | D slope | V var ratio | A var ratio | D var ratio |
|---|---:|---:|---:|---:|---:|---:|
| raw | 0.234 | 0.091 | 0.072 | 0.237 | 0.166 | 0.161 |
| VRRC s=1.2 | 0.229 | 0.087 | 0.063 | 0.229 | 0.141 | 0.124 |

VRRC **进一步压缩**了局部斜率和方差比。它之所以改善 CCC，是因为 kNN 平滑降低了噪声、scale 1.2 又把全局离散度拉回一点；这不是“恢复方差”，而是“平滑 + 全局放大”的混合。

### 3.3 scale sweep：MAE–CCC 权衡

| scale | Macro-MAE | Macro-CCC | V CCC | A CCC | D CCC |
|---:|---:|---:|---:|---:|---:|
| 1.0 | **0.165713** | 0.4065 | 0.630 | 0.360 | 0.229 |
| 1.2 | 0.168019 | 0.4427 | 0.673 | 0.398 | 0.257 |
| 1.5 | 0.176781 | 0.4734 | 0.701 | 0.434 | 0.285 |
| 2.0 | 0.201949 | 0.4824 | 0.687 | 0.454 | 0.307 |
| 3.0 | 0.277186 | 0.4344 | 0.586 | 0.420 | 0.297 |
| 4.0 | 0.368680 | 0.3722 | 0.487 | 0.365 | 0.264 |

- s=1.0 的 MAE 最好（0.1657），但 CCC 只有 0.4065；
- s=1.2 是历史选择，处在 MAE–CCC 的折中点；
- s≥2 后 MAE 迅速恶化，CCC 也见顶回落；
- 全局线性 oracle（DEV 拟合）MAE 0.1648，但 CCC 0.3914——再次说明“降 MAE”靠压缩，“提 CCC”靠扩张，二者不可兼得。

## 4. Oracle 局部仿射校准

### 4.1 方法

对每个 DEV 样本 i：

- 在 DEV 特征空间取 k 个最近邻（不含 i）；
- **local_lin**：用邻居的 `(pred_j, y_j)` 逐维拟合 `y = a·pred + b`，应用到 i；
- **local_var**：用邻居的 `mean_y, std_y, mean_pred, std_pred`，做 `q_i = mean_y + (pred_i − mean_pred)·(std_y/std_pred)`；
- **mix**：`q = VRRC + a·(local_lin − VRRC) + b·(local_var − VRRC)`，a,b 在 DEV 上网格搜索（oracle）。

### 4.2 结果

| 方法 | Macro-MAE | Macro-CCC |
|---|---:|---:|
| raw | 0.170725 | 0.4403 |
| VRRC | 0.168019 | 0.4427 |
| local_lin k=20 | 0.165619 | 0.4279 |
| local_lin k=50 | 0.164380 | 0.4105 |
| local_lin k=100 | **0.163342** | 0.4059 |
| local_var k=20 | 0.201592 | 0.4742 |
| local_var k=50 | 0.198449 | 0.4822 |
| local_var k=100 | 0.198593 | **0.4858** |
| **mix lin 0.25** | 0.164978 | **0.4455** |
| **mix lin 0.50** | **0.163646** | 0.4440 |
| **mix lin 0.50 + var 0.25** | 0.167693 | **0.4693** |

解释：

- `local_lin` 单独用会把预测进一步拉向局部均值：MAE 大降，CCC 也降；
- `local_var` 单独用会放大局部方差：CCC 大升，MAE 大坏；
- **两者与 VRRC 的混合可以同时改善 MAE 和 CCC**：`0.5·local_lin` 相对 VRRC MAE −0.0044、CCC +0.0013；`0.5·local_lin + 0.25·local_var` 相对 VRRC MAE −0.0003、CCC +0.0266。
- 这说明“局部仿射”不是简单的全局 scale，而是有真实的条件结构可利用。

### 4.3 重要限制

- local_lin/local_var 使用的是 **DEV 邻居的真实标签**，且混合系数在 DEV 上搜索；这是 oracle，不是可部署方法。
- 可部署版本必须用 **TRAIN 500 标签 / OOF 预测** 拟合局部映射，并用 TRAIN 内部分折选择 shrinkage；否则就是 DEV 泄漏。
- 5 个 DEV 邻居只有 5 个点拟合 2 个参数，方差很大；oracle 的 k=20/50/100 结果说明 k 越大越稳，但局部性越弱。

## 5. 下一步：可部署的 A-VRRC（局部仿射校准）

### 5.1 设计约束

1. **不得使用 DEV/TEST 标签**；局部映射只用 TRAIN 500 标签或 TRAIN 内 OOF 预测拟合。
2. **必须同时报告 raw 与 calibrated**，并与现有 VRRC s=1.2 配对比较；不能只与 raw 比。
3. **先做 1 seed 的 oracle-vs-deployable gap 诊断**，再做 3 seeds。
4. 预注册准入：
   - DEV Macro-MAE ≤ 0.1665 **且** Macro-CCC ≥ 0.4450；或
   - 在 ≥2/3 seeds 上同时优于 VRRC（MAE 和 CCC 都不差），且平均 MAE 改善 ≥0.001。
5. 若可部署版本只能复制“压缩换 MAE”或“扩张换 CCC”，则判定 M2 也没有独立创新，停止 VRRC 路线。

### 5.2 具体实现草案

- 对每个 DEV 样本，在 TRAIN 500 中取 k=20–50 个近邻；
- 用这 k 个邻居的 **OOF 预测**（5-fold，每折用其余 400 训练）和真实标签，拟合逐维 `y = a·p_oof + b`；
- 对 query 的 VRRC 输出 `q` 应用 `a·q + b`，并向 identity 收缩：`q' = (1−λ)q + λ(a q + b)`；
- λ 由 TRAIN 内 OOF 选择（例如 0.25/0.5/0.75），不碰 DEV；
- 可选：同时拟合局部方差比 `s = std_y/std_p`，做 `q' = μ_y + s(q−μ_p)`，再与线性校正混合；
- 报告 MAE、CCC、每维、R²、极端组 MAE、斜率/方差比。

### 5.3 需要先回答的问题

- 用 TRAIN 500 标签拟合的局部仿射，能否复现 oracle 的 0.1636/0.4440？
- 还是说 oracle 的增益主要来自“用了 DEV 邻居标签”，换成 TRAIN 邻居就消失？
- 如果 TRAIN 版本的 gap 很大，说明局部仿射对邻域标签质量敏感，需要 OOF 或半监督锚点扩展；如果 gap 小，就值得做 3 seeds。

## 6. 复现

```bash
cd /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-relstudy
PY=/root/autodl-tmp/rankup_project/envs/rankup/bin/python

# 结构诊断 + scale sweep + 全局线性 oracle
$PY scripts/vad_research/structure_diagnostic.py \
  --budget 500 --seed 0 \
  --ckpt /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-source/saved_models/vad_research/alpha_fixed075_s0/model_best.pth \
  --data-root /root/autodl-tmp/rankup_project/data/emobank_processed \
  --raw-csv /root/autodl-tmp/rankup_project/data/emobank_source/EmoBank/corpus/emobank.csv \
  --pretrain-path /root/autodl-tmp/rankup_project/pretrained/roberta-base \
  --out outputs/structure_lb500_s0.json --device cuda

# oracle 局部仿射 + 混合网格
$PY scripts/vad_research/local_affine_oracle.py \
  --ckpt /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-source/saved_models/vad_research/alpha_fixed075_s0/model_best.pth \
  --data-root /root/autodl-tmp/rankup_project/data/emobank_processed \
  --pretrain-path /root/autodl-tmp/rankup_project/pretrained/roberta-base \
  --out outputs/local_affine_oracle.json --device cuda
```

## 7. 对论文的含义

- **不能**再把 VRRC 写成“方差恢复”：数据显示它进一步压缩，scale 只是补偿；
- **可以**把“JACF 校正的 MAE–CCC/极端样本权衡”作为诊断性贡献，但这不是新方法；
- 真正有希望的第二组件是 **局部仿射校准**，但必须证明可部署版本能复现 oracle 的一部分增益，并且不是 DEV 调参；
- 如果 A-VRRC 也失败，M2 应停止，论文退回“JACF 锚点校正 + 系统性负结果/Pareto 分析”，或转向提升预测信息（CLSS/谱特征/DKD）而不是后处理。
