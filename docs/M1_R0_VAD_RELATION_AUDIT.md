# M1-R0：VAD 多维关系信息审计（2026-11-02）

## 0. 结论：R0 NO-GO

在严格 OOF、固定批次与共享特征图的协议下，**GCLSS 风格的关系恢复没有超过现有模型排序（R1），且净纠错为负**：

- F0 冻结 RoBERTa：R3 的 Pairwise Accuracy 为 V 0.541 / A 0.516 / D 0.443，R1 为 0.705 / 0.618 / 0.558；
- F1 fold-model 特征：R3 提升到 0.675 / 0.595 / 0.490，仍全面低于 R1；
- kNN 稀疏图、normalized Laplacian、32 个锚点等变体均未改变方向；
- R3 的 CR（纠正率）约 0.30–0.46，但 IR（引入错误率）约 0.13–0.47，**NetCorrection 全为负**（V −2892 ~ −16203，A −2294 ~ −10127，D −4432 ~ −11344）；
- CLSS（R2）在所有设置下接近随机（0.45–0.47），说明冻结/微调特征的全局 Fiedler 排序不编码 VAD 关系；
- 3D Direction Agreement：R1 0.279，R3 最好 0.234；3D distance correlation 也低于 R1。

→ **共享文本特征图 + 锚点约束的关系恢复，在本任务上没有提供超越 Teacher 点态预测的独立关系信息。**
→ **不进入 M1 学生训练；组件二在这一设计下没有信息基础。**

---

## 1. 协议

- 数据：EmoBank lb=500（seed 0）；外层 5 折 400/100；
- 每折 40 个诊断批次；每批 16 个标注锚点 + 32 条隐藏 OOF + 32 条 ULB，共 80 节点；
- 每批隐藏样本不重复（32 unique），共 297,600 条 pair-行（5 折 × 40 批 × 496 对 × 3 维）；
- 邻居图固定使用同一特征空间；
- 隐藏标签只用于评价，不进入图构建；
- 未访问 TEST；未读取 ULB 隐藏 VAD 标签；未使用 DEV 调参。

> 说明：R1 使用当前 fold 的 JACF student raw 预测作为“模型排序”代理；未单独训练 EMA Teacher（`train_oof_fold.py` 未保存 teacher）。F1 特征来自同一 fold checkpoint 的 encoder。这是本轮协议的限制，但 R3 低于 R1 的结论不依赖 teacher 的具体形式。

---

## 2. 图与关系恢复

| 方法 | 说明 |
|---|---|
| R0 | 随机排序 |
| R1 | fold 模型 raw 预测排序（Teacher 代理） |
| R2 | CLSS：仅 64 个无标签节点，Fiedler 向量；方向不变评价 |
| R3 | GCLSS：80 节点混合图，16 个锚点逐维标准化，`r_U=-L_UU^+ L_UL r_L` |
| R3-LP | 与 R3 同一调和延拓公式，数值等价检查 |

图：`S=(1+cos)/2`；`L=D-S`。另做 kNN(k=10) 稀疏 + normalized Laplacian、32 锚点两个预先固定的敏感性变体。

---

## 3. 主结果（Pairwise Accuracy）

| 特征/图 | 维度 | R0 | R1 | R2 | R3 | R3 NetCorrection |
|---|---|---:|---:|---:|---:|---:|
| F0 dense | V | 0.464 | **0.705** | 0.455 | 0.541 | −16,203 |
| F0 dense | A | 0.464 | **0.618** | 0.461 | 0.516 | −10,127 |
| F0 dense | D | 0.457 | **0.558** | 0.450 | 0.443 | −11,344 |
| F1 dense | V | 0.464 | **0.705** | 0.457 | 0.675 | −2,892 |
| F1 dense | A | 0.464 | **0.618** | 0.458 | 0.595 | −2,294 |
| F1 dense | D | 0.457 | **0.558** | 0.451 | 0.490 | −6,685 |
| F1 kNN10+norm | V | 0.464 | **0.705** | 0.450 | 0.655 | −4,908 |
| F1 kNN10+norm | A | 0.464 | **0.618** | 0.472 | 0.589 | −2,860 |
| F1 kNN10+norm | D | 0.457 | **0.558** | 0.451 | 0.511 | −4,654 |
| F1 dense, 32 anchors | V | 0.464 | **0.705** | 0.457 | 0.672 | −3,191 |
| F1 dense, 32 anchors | A | 0.464 | **0.618** | 0.458 | 0.588 | −2,921 |
| F1 dense, 32 anchors | D | 0.457 | **0.558** | 0.451 | 0.506 | −5,157 |

- F1 特征明显优于 F0（R3 V 0.541→0.675），说明 fold encoder 确实包含更多 VAD 相关信息；
- 但即使最好的 F1 dense，R3 仍低于 R1 约 0.03–0.07，且 NetCorrection 为负；
- R2 始终接近随机，说明**全局谱排序**不适用于 VAD 多维关系。

### 3.1 CR / IR（F1 dense，R3 vs R1）

| 维度 | Teacher wrong | Corrected | CR | Introduced | IR | Net |
|---|---:|---:|---:|---:|---:|---:|
| V | 21,387 | 6,503 | 0.304 | 9,395 | 0.134 | **−2,892** |
| A | 30,948 | 11,749 | 0.380 | 14,043 | 0.229 | **−2,294** |
| D | 33,629 | 12,917 | 0.384 | 19,602 | 0.354 | **−6,685** |

R3 能纠正一部分 Teacher 错误关系，但引入的错误更多；净纠错为负。

### 3.2 3D 联合结构

| 特征 | 方法 | 3D Direction Exact | 3D Distance ρ |
|---|---|---:|---:|
| F0 | R1 | **0.279** | **0.334** |
| F0 | R3 | 0.137 | 0.037 |
| F1 | R1 | **0.279** | **0.334** |
| F1 | R3 | 0.227 | 0.237 |

R3 的联合三维关系也没有超过模型预测。

---

## 4. GO/NO-GO

### R0-GO 条件

| 条件 | 结果 |
|---|---|
| R3 优于随机 | 是（但幅度有限） |
| R3 相比 R1 提供额外关系信息 | **否**（所有维度、所有变体均低于 R1） |
| 至少两个维度配对排序改善且 CI 正向 | **否**（dacc CI 全为负） |
| 第三个维度没有稳定负向退化 | **否** |
| NetCorrection 为正且 CI 支持 | **否**（全为负） |
| 三种子/折稳定 | 仅 seed 0；但方向已一致为负 |
| 覆盖率 ≥95% | 是（批内 100% pair 覆盖） |
| 不依赖 OOF 标签确定方向 | 是（R3 锚点定向；R2 方向不变） |

→ **R0 NO-GO。**

### 特别说明

- R0-GO 只代表信息层通过，不代表 F/P 或 Student CCC 会提高；本轮连信息层都未通过。
- R3 的失败不是“三个维度被压成一个标量”——我们已经分别恢复 V/A/D；失败在于共享特征图 + 调和延拓本身没有提供超越 Teacher 的关系信息。
- 不能把 GCLSS 换个名字当作新方法；也不能因为 R3 能纠正部分错误就忽略其引入的更多错误。

---

## 5. 为什么失败（证据边界）

1. **全局 Fiedler 排序（R2）接近随机**：冻结/微调特征的余弦图主要反映语义相似，不直接编码 VAD 的全局顺序；
2. **GCLSS 的调和延拓在稠密余弦图上过度平滑**：16/32 个锚点不足以约束 64 个无标签节点的局部多维关系；
3. **Teacher 已经很强**：R1 的 Pairwise Accuracy 0.56–0.70，关系恢复需要非常精确的局部信息才能净提升；
4. **VAD 是多维轴向关系**：单一图上的三个标量调和延拓不能保证三维联合结构；3D Direction 仅 0.23。
5. **F1 特征有帮助但不够**：从 0.54 提升到 0.67（V），仍低于 R1 0.70。

---

## 6. 决策与下一步

| 问题 | 决策 |
|---|---|
| R0 是否通过？ | **NO-GO** |
| 是否进入 M1 学生训练？ | **不进入** |
| 是否继续调 GCLSS 图/锚点/特征？ | **不继续**；已做 F0/F1、dense/kNN、16/32 锚点，方向一致 |
| 组件二的信息基础 | 在当前共享特征图设计下不成立 |

### 如果还要继续组件二

必须换**信息源**，而不是继续改图：

1. **多标注者分歧/软标签**：已经做过 U2 试点（NO-GO），但可用更强的分布头/更多标注复核；
2. **跨视图/跨语言一致性**：旅游数据的中英文并行文本可作为额外关系约束；
3. **外部知识/情感词典**：作为锚点关系的先验，而不是从文本图恢复；
4. **Pairwise ranking distillation**：RankUp/SSR-RCUS 已覆盖，且本任务排序机制此前无稳定优势，不建议重复；
5. **多任务辅助**：如同时预测 V/A/D 的 ordinal 等级，但需证明不是已有 ordinal regression 的变体。

在没有新的信息源之前，**不建议继续投入 M1 组件二**；JACF 仍是唯一已验证的 MAE/R² 机制。

---

## 7. 复现

```bash
cd /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-avrrc
PY=/root/autodl-tmp/rankup_project/envs/rankup/bin/python

# F1 dense, 16 anchors
$PY scripts/vad_research/m1/relations/run_r0.py \
  --data-root /root/autodl-tmp/rankup_project/data/emobank_processed \
  --pretrain-path /root/autodl-tmp/rankup_project/pretrained/roberta-base \
  --manifest outputs/p0/fold_manifest.json --oof-dir outputs/oof \
  --out-dir outputs/m1_r0_f1 --seed 0 --feature-source f1 --device cuda

$PY scripts/vad_research/m1/relations/evaluate_r0.py \
  --pairs outputs/m1_r0_f1/pairs_seed0.parquet \
  --graph-diag outputs/m1_r0_f1/graph_diag_seed0.json \
  --out outputs/m1_r0_f1/eval_seed0.json --boot 1000
```

测试：`pytest -q tests_vad/test_m1_relations.py`（4 passed）。
