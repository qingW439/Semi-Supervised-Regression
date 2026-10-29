# EmoBank 半监督三维 VAD 回归：模块替换实验记录

日期：2026-10-29  
实验目录（主机）：`/root/autodl-tmp/rankup_project/Semi-Supervised-Regression-vadswap`（从主项目复制，未改动原项目）  
数据：EmoBank VAD，`data/emobank_processed`，TRAIN 8062（lb 250/500/1000 + ulb），DEV 1000，TEST 不读取。

---

## 0. 结论速览

固定设置：`lb=500`、3 个配对种子（0/1/2）、相同 P1 初始化、160 步续训、DEV 选模。

| 模型 | Macro-MAE ↓ | Macro-CCC ↑ | Macro-R² ↑ |
|---|---|---|---|
| P2 supervised control（强基线） | **0.175589 ± 0.000676** | 0.4418 | 0.1750 |
| Mean Teacher (MT) | 0.175915 | 0.4402 | 0.1745 |
| MT + 伪标签检索校准（PL-only） | 0.173722 | 0.4396 | 0.1924 |
| 仅推理期检索校准（Infer-only） | 0.170507 | 0.4468 | 0.2144 |
| **RCMT（两者结合，最终模型）** | **0.169556 ± 0.000355** | **0.4448** | **0.2247** |

相对 P2 强基线：

- Macro-MAE：**-0.006033**（-3.44%），三个种子分别 -0.005893 / -0.006431 / -0.005774，方向一致；
- Macro-CCC：**+0.00294**（未损害预测变化能力，反而略升）；
- Macro-R²：**+0.04976**（+28.4%）。

三个主指标同时改善，满足“不能只降 MAE 而牺牲 CCC/R²”的准入条件。

---

## 1. 已有实验盘点

### 1.1 GitHub 仓库

`https://github.com/qingW439/Semi-Supervised-Regression` 是 RankUp（NeurIPS 2024）官方代码的 fork：
- `semilearn/`：USB 风格的 SSR 框架，含 `rankup / rankuprda / rda / ucvme / clss / meanteacher / mixmatch / pimodel / fullysupervised`；
- 原生任务为 UTKFace（图像年龄）、BVCC（音频质量）、Yelp Review（单目标文本评分）；
- 仓库本身不含 EmoBank/VAD 代码。

### 1.2 主机上的已有实验（`/root/autodl-tmp/rankup_project`）

主项目分支 `vad-m1-v3-pseudolabel-calibration`，另有 9 个实验副本（rank3d、rcus3d、strong-supervised、lact-audit、m1-utility-audit、spectral-readiness、m1-validity-v2 等）。

关键历史结果（DEV，[1,5] 原始尺度）：

| 实验 | Macro-MAE | 说明 |
|---|---|---|
| P1 RoBERTa supervised | 0.178264 / 0.179980 / 0.177377（s0/s1/s2） | 20 epoch，L1，AdamW 2e-5 |
| P2 supervised control | 0.175543 / 0.176286 / 0.174937 | 从 P1 续训 160 步，L1 |
| Mean Teacher | 0.174788 / 0.175580 / 0.177377 | EMA 伪标签，λ=0.2，不稳定 |
| HPL A1 / GGR / crossdim | 均为 0.178264（step 0） | 无增益，最优解退回 P1 |
| Spectral B0/B1/B2 | 0.177539 / 0.180401 / 0.185342 | 无增益 |
| RankUp-VAD smoke（E2） | 0.268122 | 实现不完整，不可用 |
| M1 维度效用（多种配置） | 均 0.178264（step 0） | 无增益 |
| SSR-simregmatch 3D | full 0.175878；filtering 0.175747 | 未超过 P2 |

结论：P2 control（0.175589 ± 0.000676）是当前最强、最稳定的基线；MT 在 s0 有偶然优势但三种子均值更差（0.175915）。

---

## 2. 文献核对与创新边界

已完成系统检索（2023–2026），核心结论：

1. **目前没有已发表的“文本半监督 VAD 回归”工作**；EmoBank 上的半监督设置本身是空白。最接近的半监督连续情绪回归是 Interspeech 2020 的跨文化音视频工作（非文本）。
2. 半监督回归（SSR）的代表工作：RankUp/RDA（NeurIPS 2024）、SSR-RCUS（AAAI 2026）、HPL（NeurIPS 2025）、GaussianMatch（CVPR 2026）、UCVME（AAAI 2023）、SimRegMatch（ASOC 2024）、UCRC（Neurocomputing 2026）、Ding et al.（FCS 2026）、MASSTER（IJCNN 2025，多目标实例-目标对选择）。
3. 必须避开的创新重合：
   - “用标注反馈学习伪标签可靠性” = **HPL 的双层优化**；
   - “按维度/目标加权” = **MASSTER 的实例-目标对选择**；
   - “把伪标签分布对齐到标注分布” = **RDA**；
   - “不确定性加权” = UCVME/HPL/UCRC/GaussianMatch。
4. 仍相对空白、且本文实际采用的点：**用标注样本在特征空间中的邻域结构，对无标签伪标签做逐维度的检索校准**（不是对齐全局分布，也不是学一个样本级标量不确定性），并在推理期做方差恢复式校准。RDA 对齐的是标签分布，HPL 学的是样本级标量 σ，本文利用的是“标注特征邻域”这一局部、逐维度的信号。

---

## 3. 方法：RCMT（Retrieval-Calibrated Mean Teacher）

### 3.1 基础框架

- 编码器：RoBERTa-base + attention-mask mean pooling + 2 层 MLP 回归头（3 维）；
- 初始化：冻结的 P1 有监督检查点；
- 续训：160 步，labeled batch 16，unlabeled batch 64，AdamW lr 5e-6，wd 0.01，L1；
- 教师：学生权重的 EMA（decay 0.99）。

### 3.2 组件 A：LA-PLC（Label-Anchored Pseudo-Label Calibration）

用 P1 编码器对 500 条标注文本和全部无标签文本编码并 L2 归一化，得到标注特征库 `{f_j, y_j}`。对每条无标签文本 `x_i`：

```
s_ij = <f_i, f_j> / (||f_i|| ||f_j||)
w_ij = softmax_j(s_ij / τ)          # 取 top-k 邻居
knn_i = Σ_j w_ij y_j
gate_i = clip((max_j s_ij - c0) / (1 - c0), 0, 1)
a_i = α · gate_i
p̃_i = (1 - a_i) · p_teacher(x_i) + a_i · knn_i
```

默认 `k=20, τ=0.1, c0=0.3, α=0.4`。逐维度独立进行（V/A/D 各自校准），因此同一条文本可以对 V 采用更多邻域信息、对 A/D 保留更多教师预测。

**伪标签质量证据**（P1 教师，DEV 上诊断，不参与训练）：

| 伪标签 | Macro-MAE |
|---|---|
| 原始教师 `p_teacher` | 0.178264 |
| LA-PLC 校准后 `p̃`（k=20, α=0.4） | **0.168668** |

即校准使伪标签误差下降约 0.0096（-5.4%），这是“组件改善伪标签质量”的直接证据。

### 3.3 组件 B：VRRC（Variance-Restored Retrieval Calibration，推理期）

对最终学生预测做同样的邻域混合，然后做逐维方差恢复：

```
q_i = (1 - a_i) · p_student(x_i) + a_i · knn_i
q̃_i = mean(q) + s · (q_i - mean(q)),   s = 1.2
```

`s` 的作用：邻域混合会进一步压缩预测方差（CCC 下降），`s>1` 把方差恢复到接近标注分布，从而在降 MAE 的同时保住/提升 CCC 与 R²。`s=1.2` 在 DEV 上固定，三种子一致使用。

### 3.4 与已有方法的关系

| 方法 | 校准对象 | 粒度 | 本文差异 |
|---|---|---|---|
| RDA | 伪标签 → 标注标签分布（分位数对齐） | 全局、单目标 | 本文用特征邻域、逐维度、局部 |
| HPL | 样本级标量 σ，双层优化 | 样本 | 本文不学标量不确定性，直接用邻域标签 |
| UCVME/UCRC | 异方差/一致性 | 样本 | 同上 |
| MASSTER | 实例-目标对选择（主动学习） | 目标 | 本文是伪标签侧、非主动查询 |
| GaussianMatch | 多视图高斯一致性过滤 | 样本 | 本文用标注邻域而非多视图 |

---

## 4. 模块筛选记录（失败模块）

在 `lb=500, seed=0` 上逐一替换/加入模块（均从同一 P1 初始化，160 步）：

| 模块 | 来源 | 配置 | 最优 DEV Macro-MAE | 结论 |
|---|---|---|---|---|
| CCC loss（监督） | 维度情绪回归常用 | w=0.3 | 0.178264（step 0） | 无增益 |
| CCC loss（监督+伪标签） | 同上 | w=0.3 | 0.178264 | 无增益 |
| 成对排序一致性 | RankUp / SSR-RCUS | w=0.1 | 0.177701 | 略差于 MT |
| 成对排序一致性 | 同上 | w=0.3 | 0.177041 | CCC 升、MAE 降 |
| 教师 dropout 分歧加权 | SimRegMatch / Ding | τ=0.05 | 0.178264 | 无增益 |
| 分歧加权 + top-q 过滤 | GaussianMatch 风格 | q=0.5 | 0.178264 | 无增益 |
| 异方差不确定性加权 | HPL / UCVME | Laplace NLL | 0.178264 | 无增益 |
| 师生一致性加权 | — | τ=0.05 | 0.174066 | 小幅但不如检索校准 |
| 分布对齐 DA | RDA / ReMixMatch | s=1 | 0.206（诊断） | MAE 明显变差 |
| EMA decay 0.999 | Mean Teacher | — | 0.173949 | 有增益但不跨种子稳定 |
| **LA-PLC（伪标签检索校准）** | 本文 | α=0.4, k=20 | **0.173079** | 稳定增益 |
| **LA-PLC + VRRC** | 本文 | +s=1.2 | **0.169650** | 最终 |

结论：排序、CCC、不确定性、分歧过滤、分布对齐等“常规模块替换”在 EmoBank 500 标签上均未带来稳定收益；真正有效的是利用标注特征邻域的检索校准 + 方差恢复。

---

## 5. 最终三种子结果（lb=500）

| seed | P2 control MAE | RCMT MAE | ΔMAE | RCMT CCC | RCMT R² |
|---|---|---|---|---|---|
| 0 | 0.175543 | 0.169650 | -0.005893 | 0.4488 | 0.2265 |
| 1 | 0.176286 | 0.169855 | -0.006431 | 0.4420 | 0.2285 |
| 2 | 0.174937 | 0.169163 | -0.005774 | 0.4435 | 0.2192 |
| **mean** | **0.175589** | **0.169556** | **-0.006033** | **0.4448** | **0.2247** |
| std | 0.000676 | 0.000355 | — | — | — |

消融（三种子均值）：

| 配置 | Macro-MAE | Macro-CCC | Macro-R² |
|---|---|---|---|
| P2 control | 0.175589 | 0.4418 | 0.1750 |
| MT | 0.175915 | 0.4402 | 0.1745 |
| + LA-PLC only | 0.173722 | 0.4396 | 0.1924 |
| + VRRC only | 0.170507 | 0.4468 | 0.2144 |
| + both（RCMT） | 0.169556 | 0.4448 | 0.2247 |

两个组件均有独立贡献且可叠加；LA-PLC 主要降 MAE，VRRC 同时提升 CCC/R²。

---

## 6. 复现命令

```bash
cd /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-vadswap
ENV=/root/autodl-tmp/rankup_project/envs/rankup/bin/python
ORIG=/root/autodl-tmp/rankup_project/Semi-Supervised-Regression

# P1（已有 lb500，可跳过；250/1000 需新训）
$ENV scripts/vad_research/train_roberta_vad_supervised.py \
  --budget 500 --seed 0 --device cuda --offline \
  --model /root/autodl-tmp/rankup_project/pretrained/roberta-base \
  --out saved_models/vad_research/p1_roberta_lb500_s0

# RCMT（lb500, seed0；seed1/2 改 --seed 与 --p1-dir）
$ENV scripts/vad_research/train_vad_modules.py \
  --mode mean_teacher --budget 500 --seed 0 \
  --p1-dir $ORIG/saved_models/vad_research/p1_roberta_lb500_s0 \
  --knn-calib 0.4 --knn-k 20 \
  --eval-knn-alpha 0.4 --eval-knn-k 20 --eval-scale 1.2 \
  --diag-teacher --device cuda \
  --out saved_models/vad_research/final_full_s0
```

结果目录：`saved_models/vad_research/final_full_s{0,1,2}/dev_metrics_best.json`。

---

## 7. 待完成

1. `lb=250 / 1000` 三种子（P1 检查点正在补训）。
2. 仅一次 TEST 评估（`test_gold.parquet`），用于论文最终表；当前所有结果均未读取 TEST。
3. 领域迁移（中英文旅游评论/游记）实验。
4. 若需要更强论文动机，可补：不同 k/α 的敏感性分析、标注预算-收益曲线、与 HPL/RankUp 官方实现的公平对比。
