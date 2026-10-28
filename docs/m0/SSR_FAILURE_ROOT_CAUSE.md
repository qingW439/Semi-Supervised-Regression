# EmoBank 500-label M0：半监督候选失败归因

**执行分支：核谱 TRAIN-only NO-GO 后的“情况二”。当前没有证据表明已建立可稳定超过 P2 的半监督 M0；停止现有候选的机制筛选，暂不进入 M1/M2。** 本轮只分析已有模型、TRAIN 数据与已有预测，另做一次预先固定的 CPU 负对照；未训练模型、未使用 GPU、未访问 TEST，也未用 DEV 选择诊断参数。

## 1. 已测试机制的证据边界

| 方法 | 可核证据 | 当前判断及可比性 |
|---|---|---|
| P2 强监督 | RoBERTa，500 TRAIN，3 seed DEV Macro-MAE `0.175589±0.000676`；各 seed 最优均 step 128 | 现有强监督参照，不与旧 BERT 或 400/100 TRAIN 探针直接做因果差值 |
| Mean Teacher | 同一 P2 系列，3 seed DEV `0.174788/0.175580/0.177377`；P2 同 seed 为 `0.175543/0.176286/0.174937` | 两 seed 有微小数值改善，一 seed 明显退化；均值约 `0.175915`，未获得稳定收益 |
| RankUp 3D | 旧 BERT 续训协议，seed 0 DEV `0.210317`；同协议 `supervised_control` 为 `0.199843` | 当前实现劣于本协议控制；不能将其与 RoBERTa P2 数值直接比较 |
| SSR-RCUS 3D | 旧 BERT 协议，seed 0 `R1=0.204727`、`R0=0.204727` | 未显示相对于该协议 R0 的收益；与 P2 不同编码器和优化协议 |
| HPL A0/A1 | 已完成匹配审计：A0 是纯监督，不是 SSL；A1 的无标签项确实反传，但最佳 checkpoint 在 step 0，DEV `0.178264`；匹配 A0 最优 step 96 为 `0.175329` | A1 未带来可用的更新收益。HPL 用 MSE、分离 Adam，与 P2 的 L1/AdamW 不可直接归因比较 |
| SimRegMatch-3D | 严格 TRAIN 400/100 审计，校准点估计增益为归一化尺度 `0.001726`，预设 Bonferroni bootstrap 下界 `−0.000682` | 正点估计未通过预设准入；不能写成“所有样本均退化”或可靠改善 |
| LACT-VAD | 机制审计 NO-GO：图有结构，但初始梯度尺度异常，普通头正则等价性未排尽；未训练完整模型 | 只有机制准入证据，没有可与 P2 比较的预测成绩；停止调 λ/K |
| 核谱 B1/B2 | 锁定 TRAIN-only 五折；B1 OOF `0.183908`、B2 `0.185951`，B2−B1 `+0.002043`，仅 1/5 折改善 | 直接按预设规则 NO-GO；OOF 与 rank/alpha 选择共用五折，仍是选择条件下估计，不等同独立外测 |

上述均是 **EmoBank 当前数据、实现与协议下的负结果**，不推及全部半监督回归。旧核谱 B0/B1/B2 的 seed 0 DEV 文件仅保留为历史资产：它们没有完整代码哈希，且与本轮 TRAIN OOF 不在同一评价集；本轮没有据其分数改参数，也没有重复 B0 或正式 DEV 拟合。

## 2. 固定 TRAIN 内诊断设计

计算前锁定 `docs/m0/SSR_FAILURE_DIAGNOSTIC_PROTOCOL.json`（SHA256 `a009a4e5b3e3dc38a49a73dce28422d0daf2f48b89489910e680ce97e3dcfe84`）。复用 P3B 既有 seed 0 **400 参考 / 100 留出** ID 和预测；标签盲表示取已经核验过的冻结预训练 RoBERTa 池化缓存，L2 归一化。400 个参考标签只用于目标差值与覆盖阈值；100 个留出标签只用于评价。固定余弦距离、`k=8`、随机种子 `20261028`，不按结果调整。

- **D1/D2 近邻目标一致性：** 对每条留出文本从 400 个参考文本取 8 个表示近邻，计算原始 [1,5] V/A/D 绝对标签差。随机对照为同一留出文本对全部 400 个参考标签的精确均值，另存一次不放回随机 8 邻居抽样。1000 次仅置乱参考标签、不动表示近邻或留出标签；10000 次按留出文本重采样给出配对增益区间。所有检验都是该探针上的描述性诊断。
- **D3 低覆盖误差：** 在 400 个参考标签的三维归一化 VAD 空间，计算每个参考点除自身外第 8 近邻半径；其第 75 百分位 `0.055902` 是预先定义的稀疏阈值。100 条留出样本的真实标签只用于分组评价，误差来自早已训练完成的 P3B 400→100 预测。
- **谱机制归因：** 另在计算前锁定一次性负对照 `SSR_FAILURE_ATTRIBUTION_PROTOCOL.json`（SHA256 `8701b607aa0470ef1e428d763abd7c8ac00b9ef5dfc28ca4a8941da65560bc41`）。完全复用核谱门槛的 5 折 gamma、256 共同锚点、rank 64、Ridge alpha 100；把 7562 条真实无标签表示一次性替换为从该折 400 条拟合表示有放回抽取的 7562 条固定样本（种子 `314159+fold`）。这是算法输入负对照，不是新模型或调参。重算 B1/B2 与已保存 OOF 逐值一致，最大差 `4.44e−16` 原尺度。

## 3. D1/D2：表示与真实 VAD 的关系

近邻平均余弦距离 `0.002687`，全部参考文本平均 `0.052969`，确认选到局部文本。随机期望减近邻的标签差值越大，表示局部对目标越有信息。

| 维度 | 8-NN 标签差 | 随机期望标签差 | 增益 | 留出文本 bootstrap 95% 区间 | 参考标签置乱下单侧 p | 平均逐留出 Spearman |
|---|---:|---:|---:|---:|---:|---:|
| V | 0.344875 | 0.378430 | **0.033555** | [0.010905, 0.055647] | 0.00599 | 0.0256 |
| A | 0.240775 | 0.270570 | **0.029794** | [0.015722, 0.043325] | 0.00100 | 0.0283 |
| D | 0.207600 | 0.220920 | 0.013319 | [0.000463, 0.025827] | 0.04296 | −0.0236 |
| 三维均值 | — | — | **0.025556** | [0.012871, 0.037735] | 0.00100 | 0.0294 |

V/A 和总体的最近邻优势在固定探针上可见；D 较弱，其置乱 p 值不满足三维同时检验的简单 Bonferroni 阈值 `0.05/3`，且全距离范围的逐留出排序相关略为负值。所有维度的距离—目标秩相关都很小，说明“最邻近文本略有目标一致性”不能提升为“整个表示距离连续、可靠地度量 VAD 差”。100 条留出中只有 1 条与 400 参考文本有完全相同的文本；去掉它后总体增益为 `0.025579`，与全样本 `0.025556` 几乎一致。局部关联不等于 7562 条无标签文本提供了额外可用信息。

## 4. D3：严格留出预测与标签覆盖

P3B 原有 400→100 预测在 100 条严格留出上的 Macro-MAE 为 `0.180650`。按固定标签空间半径阈值分组：

| 区域 | 数量 | V MAE | A MAE | D MAE | Macro-MAE |
|---|---:|---:|---:|---:|---:|
| 密集 | 80 | 0.171464 | 0.190717 | 0.118441 | **0.160197** |
| 稀疏 | 20 | 0.337429 | 0.220619 | 0.243774 | **0.262463** |

稀疏区误差高 `0.102266`，V/D 尤其明显。P3B 留出预测标准差/真实标准差为 V `0.857`、A `0.742`、D `0.620`，显示一定的向均值收缩；P3B 在高/低真值区也有既有偏差记录。分区是按真实留出标签做的**事后误差切片**，其阈值仅由 400 拟合标签决定；它不能证明“低标签覆盖”是误差的唯一因果来源，极端值难度与标注噪声同样可能造成差异。

## 5. 固定谱归因：无标签改变了表示，但未改善目标

同一 256 锚点坐标中，B1 与真实 B2 的 64 维子空间投影重叠均值为 **0.8213**；B1 与“重采样 7562 条拟合表示”对照的重叠为 **0.9416**。真实无标签数据确实使谱方向发生更明显的变化。五折 OOF Macro-MAE：B1 `0.183908`、真实 B2 `0.185951`、重采样对照 `0.184404`。真实 B2 比 B1 差 `0.002043`，比这一次固定重采样对照差 `0.001547`。单次随机对照不构成显著性检验，但排除了“B2 与 B1 谱方向基本相同、无标签根本没进入机制”这一解释。

| OOF 方法 | V 预测/真实标准差 | A | D | V/A/D MAE |
|---|---:|---:|---:|---|
| B1 | 0.451 | 0.386 | 0.369 | 0.227450 / 0.176684 / 0.147590 |
| 真实 B2 | 0.469 | 0.394 | 0.391 | 0.230920 / 0.179286 / 0.147647 |
| 重采样对照 | 0.449 | 0.383 | 0.374 | 0.227671 / 0.177497 / 0.148046 |

核谱三组都有明显预测方差收缩；B2 相对 B1 **略微增加**而非进一步压缩预测方差，因此 B2 较高 MAE 不能简单归咎于“B2 比 B1 更塌缩”。三维中 V/A 的 B2 MAE 退化较明显，D 几乎持平；B2 的 V/D CCC 小幅上升、A CCC 下降，未转化为总体 MAE 收益。两套分析使用不同评价设计：P3B 是一次严格 400→100 留出，核谱 500 OOF 与参数选择共用五折，不应直接横比绝对 MAE。

LACT 既有图审计也给出相容但不等价的证据：7562 个无标签节点的 P1 互 8-NN 图有 8984 条边、23.76% 孤立点，局部边比随机边短，方向协方差非各向同性，真实/随机图梯度不同；但这没有证明边方向与真实 VAD 变化一致。LACT 的 Q 在严格 TRAIN 折外有部分 VAD 协变，但固定坐标的单维关联并不稳定，机制仍因梯度门槛 NO-GO。该图来自用 500 标签训练的 P1 表示，不能与本次标签盲冻结表示的 kNN 数值直接混用。

## 6. 主要失败假设、反证和边界

**主要工作假设：** 当前冻结文本表示包含有限的局部 V/A 目标信息，但 7562 条无标签文本主导的**全局核谱方向调整与三维 VAD 回归目标不充分对齐**；核谱确实吸收了额外几何，却没有把它转化为更低的 MAE。支持证据是固定 400/100 探针中的局部目标增益、B1→B2 相对随机重采样更大的子空间变化，以及 B2 四折退化、总体 MAE 上升。

**反证与不确定性：** D 的局部关联弱；全距离秩相关几乎为零；B2 在 V/D 的 CCC 及预测方差略好；单个随机重采样对照不足以将“无标签分布偏移”与 Nyström 有限秩、Ridge 标准化/正则、500 标签抽样波动分开。低覆盖区高误差也可能由标签极端性或个体标注分歧造成。因此现有证据只能定位到“目标相关信息未被当前全局谱机制可靠转化”，**不能证明唯一微观原因**，也不能声称所有表示学习路线均无效。

Reader/Writer 个体分歧目前**无法估计**：TRAIN parquet 仅有合并后的 V/A/D，未标注 parquet 仅有 ID/text；官方本地 `corpus/emobank.csv` 的表头为 `id,split,V,A,D,text`，没有逐个读者/作者评分或有效人数。现有 manifest 说明标签按 Reader/Writer 有效人数合并，但没有保留逐标注方差。旧 E1 的 250/500/1000 标签学习曲线 DEV MAE `0.227936/0.205648/0.194429` 显示增加标注有利于其**旧 BERT 协议**，不能直接外推为 P2 的样本量因果曲线，更不能从中估计标注噪声。

**最终研究判断：** 当前 M0 尚未建立。冻结这些候选，不启动新的半监督框架、LACT-v2、M1/M2 或旅游域迁移。若以后需要继续研究，先明确如何区分目标几何、标签覆盖与标注可靠性；本轮诊断到此停止。

## 7. 复现、文件与范围

逐条 100 条探针指标：`docs/m0/SSR_FAILURE_DIAGNOSTICS.csv`；完整置乱、bootstrap、覆盖和历史核谱摘要：`SSR_FAILURE_DIAGNOSTICS.json`。一次性谱负对照的 500 条 OOF：`SSR_FAILURE_ATTRIBUTION_OOF.csv`，数值与子空间统计：`SSR_FAILURE_ATTRIBUTION.json`。脚本仅为分析器：`scripts/vad_research/m0/ssr_failure_diagnostics.py`、`ssr_failure_attribution.py`，不定义或训练新模型。

```bash
cd /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-spectral-readiness
PY=/root/autodl-tmp/rankup_project/envs/rankup/bin/python
F=/hy-tmp/rankup_project/Semi-Supervised-Regression/saved_models/vad_research/m0/pretrained_features_lb500_s0
P=/hy-tmp/rankup_project/Semi-Supervised-Regression/saved_models/vad_research/p3b_probe400_100_lb500_s0
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY" scripts/vad_research/m0/ssr_failure_diagnostics.py --feature-dir "$F" --probe-dir "$P" --protocol docs/m0/SSR_FAILURE_DIAGNOSTIC_PROTOCOL.json --spectral-params docs/m0/SPECTRAL_TRAIN_ONLY_PARAMS.json --out-csv docs/m0/SSR_FAILURE_DIAGNOSTICS.csv --out-json docs/m0/SSR_FAILURE_DIAGNOSTICS.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY" scripts/vad_research/m0/ssr_failure_attribution.py --feature-dir "$F" --spectral-params docs/m0/SPECTRAL_TRAIN_ONLY_PARAMS.json --spectral-oof docs/m0/SPECTRAL_TRAIN_ONLY_OOF.csv --protocol docs/m0/SSR_FAILURE_ATTRIBUTION_PROTOCOL.json --out-json docs/m0/SSR_FAILURE_ATTRIBUTION.json --out-csv docs/m0/SSR_FAILURE_ATTRIBUTION_OOF.csv
```

脚本拒绝覆盖既有输出；复现需先选择新输出路径。本轮未触碰历史 checkpoint 或原主工作树的未提交 M0 文件。
