# P1/P2：TRAIN-only OOF 与 A-VRRC 可部署性验证报告（2026-11-02）

## 0. 结论：P2-A NO-GO，停止 A-VRRC，不进入 P3

在严格的 TRAIN-only 5-fold OOF 协议下，可部署的局部仿射校准**没有复现 DEV oracle 的增益**：

| DEV 校准 | Macro-MAE | Macro-CCC | 相对 VRRC 的配对 bootstrap |
|---|---:|---:|---|
| raw | 0.170725 | 0.4403 | — |
| VRRC-fixed（base） | **0.167875** | 0.4430 | — |
| Global Affine | 0.165494 | 0.4162 | MAE −0.000595（显著），CCC −0.0269（显著变差） |
| Local Intercept | 0.167751 | **0.4484** | MAE −0.000031（CI 跨 0），CCC +0.0054（显著） |
| Local Affine | 0.167539 | 0.4464 | MAE −0.000084（CI 跨 0），CCC +0.0034（显著） |

- Local Affine 相对 VRRC 的 MAE 改善只有 **0.000084**，95% CI [−0.000211, +0.000033]，**不显著**；
- Local Affine 与 Local Intercept 几乎相同（DEV MAE 0.167539 vs 0.167751；CCC 0.4464 vs 0.4484），**局部斜率没有独立价值**；
- 更关键的是，**OOF 交叉验证**（500 条，拟合与评价完全分离）没有复现任何增益：

| OOF cross（500 条） | Macro-MAE | Macro-CCC |
|---|---:|---:|
| raw | 0.181740 | 0.4267 |
| VRRC | **0.177537** | **0.4370** |
| Global Affine | 0.175006 | 0.4033 |
| Local Intercept | 0.178070 | 0.4350 |
| Local Affine | 0.177559 | 0.4362 |

Local Affine 在 OOF 上相对 VRRC：MAE +0.000022、CCC −0.0008，**等价于没有校准**。DEV 上的 CCC 提升没有迁移到 OOF，说明它是最终 500-label 模型几何/分布下的局部现象，不是可复现的校准机制。

→ 按预注册标准：**P2-A NO-GO；停止 A-VRRC；不进入 P3 训练侧伪标签实验。**

补充的表示瓶颈 probe（2.4 节）进一步显示：折 encoder 特征上的诚实线性 probe（0.181627/0.4328）与模型 raw（0.181740/0.4267）几乎相同，**回归头不是瓶颈**；剩余误差更可能来自表示/标签噪声，而不是校准或回归头。

---

## 1. P1：严格 OOF 是怎么生成的

### 1.1 协议

- 数据：EmoBank lb500_s0，500 条 TRAIN 标签；5 折 × 100，V 四分位分层，seed `20261101`；
- 每折：400 条训练 + 100 条折外；
- P1：RoBERTa + 3 维回归头，400 条监督，**固定 10 epochs**，无 DEV early stopping；
- JACF：固定 α=0.75，400 标签 + 7562 无标签，**固定 96 steps**，无 DEV 选模；
- 预测：折外 100 条，保存 raw + VRRC-fixed（折内 400 锚点库，fold-specific 冻结中心化）；
- 标签隔离：`train_ids ∩ holdout = 0`；锚点库只含 400 条训练标签；不读取 raw CSV / 隐藏无标号标签 / DEV / TEST。

### 1.2 OOF 结果（500 条）

| 方法 | Macro-MAE | Macro-CCC | Macro-R² | V/A/D MAE |
|---|---:|---:|---:|---|
| raw | 0.181740 | 0.4267 | 0.2157 | 0.2024 / 0.1828 / 0.1600 |
| VRRC-fixed | **0.177537** | **0.4370** | **0.2493** | 0.2013 / 0.1769 / 0.1543 |

逐折 raw MAE：0.17639 / 0.18230 / 0.16920 / 0.19801 / 0.18279；VRRC：0.16843 / 0.17883 / 0.16636 / 0.19190 / 0.18217。折间差异大（n=100），但 VRRC 在 5/5 折都优于 raw。

### 1.3 OOF 与 500-label 最终模型的分布差

| 维度 | OOF raw mean / std | DEV raw mean / std | 差值 mean |
|---|---|---|---:|
| V | 0.5069 / 0.0577 | 0.4943 / 0.0610 | +0.0126 |
| A | 0.5221 / 0.0378 | 0.5195 / 0.0379 | +0.0026 |
| D | 0.5270 / 0.0302 | 0.5188 / 0.0289 | +0.0082 |

OOF（400-label 折模型）整体比 500-label 最终模型更“正”、V 维方差更小。这是 400→500 标签分布差，也是局部校准映射不能直接迁移的风险来源。

---

## 2. P2：可部署校准的完整对照

### 2.1 实验设计

- Base predictor：`alpha_fixed075_s0`（500 标签，best step 96）；
- 校准数据：500 条 OOF 的 `(raw/VRRC, y)`；
- Base VRRC：DEV 用最终模型 encoder + 500 锚点；OOF 用**各折自己的 encoder** + 400 锚点（避免最终模型 encoder 看到折外标签）；
- 局部校准检索：统一使用**冻结预训练 RoBERTa**特征（无标签、跨折可比）；
- 校准器：
  - Global Affine：逐维 `y = a·q + b`，500 条 OOF 拟合；
  - Local Intercept：k=50 邻居，加权平均残差 `c`，`q_new = q + 0.5c`；
  - Local Affine：k=50 邻居，加权 ridge 拟合 `r = c + γq`，`q_new = q + 0.5(c + γq)`；
- 无 DEV 调参；k/τ/ridge/shrink 固定为 50 / 0.1 / 1.0 / 0.5。

### 2.2 DEV 结果

| 方法 | MAE | CCC | R² | V/A/D CCC |
|---|---:|---:|---:|---|
| raw | 0.170725 | 0.4403 | 0.2198 | 0.663 / 0.387 / 0.271 |
| VRRC | 0.167875 | 0.4430 | 0.2382 | 0.673 / 0.398 / 0.257 |
| Global Affine | **0.165494** | 0.4162 | **0.2599** | — |
| Local Intercept | 0.167751 | **0.4484** | 0.2412 | — |
| Local Affine | 0.167539 | 0.4464 | 0.2417 | — |

配对 bootstrap（2000 次，按样本重采样）：

| 对比 | ΔMAE vs VRRC [95% CI] | ΔCCC vs VRRC [95% CI] |
|---|---|---|
| Global Affine | −0.000595 [−0.000910, −0.000286] | −0.0269 [−0.0309, −0.0226] |
| Local Intercept | −0.000031 [−0.000236, +0.000194] | +0.0054 [+0.0026, +0.0081] |
| Local Affine | −0.000084 [−0.000211, +0.000033] | +0.0034 [+0.0019, +0.0050] |

DEV 上局部方法确实显著提升 CCC，但 MAE 改善不显著，且 Local Affine 没有优于 Local Intercept。

### 2.3 OOF 交叉验证（更严格的 deployable 估计）

| 方法 | MAE | CCC |
|---|---:|---:|
| raw | 0.181740 | 0.4267 |
| VRRC | 0.177537 | 0.4370 |
| Global Affine | 0.175006 | 0.4033 |
| Local Intercept | 0.178070 | 0.4350 |
| Local Affine | 0.177559 | 0.4362 |

- Local Affine 与 VRRC 等价（MAE +0.000022，CCC −0.0008）；
- Local Intercept 比 VRRC 更差（MAE +0.000533，CCC −0.0020）；
- Global Affine 降 MAE 但 CCC 大幅下降，重演“压缩换 MAE”的权衡。

**DEV 的 CCC 增益没有在 OOF 上复现**，因此不能作为可部署的独立组件。

### 2.4 表示瓶颈 probe：回归头不是瓶颈

为了判断“是否还有被 3 维回归头丢掉的表示信息”，在每个折模型**自己的 encoder 特征**上做诚实 OOF 线性 probe：用 400 条 bank 特征拟合 ridge，预测 100 条折外特征（折 encoder 没见过这 100 条标签）。

| OOF（500 条） | Macro-MAE | Macro-CCC | Macro-R² |
|---|---:|---:|---:|
| ridge alpha=1000 on fold-encoder features | **0.181627** | **0.4328** | **0.2394** |
| 当前模型 raw | 0.181740 | 0.4267 | 0.2157 |
| 当前模型 + VRRC | **0.177537** | **0.4370** | **0.2493** |

结论：

- 线性 probe 与模型 raw 几乎相同（MAE −0.00011，CCC +0.0061，R² +0.0237），**3 维 MLP 头不是主要瓶颈**；
- probe 没有明显超过模型，说明 encoder 特征中**没有大量被头丢掉的、可线性读出**的 VAD 信息；
- VRRC 仍比 probe 好（0.177537/0.4370），说明检索平滑提供了额外收益，但这是校准，不是新信息。

这支持“信息瓶颈在表示/标签，而不在回归头或后处理校准”的判断。

---

## 3. 为什么 oracle 的局部仿射没有迁移

1. **oracle 使用了 DEV 邻居的真实标签**：`local_lin` 的拟合样本与评价样本同分布，且邻居标签是 DEV 的；可部署版本只能用 TRAIN OOF，400-label 折模型的残差分布与 500-label 最终模型不同。
2. **特征几何不同**：oracle 用微调后的 encoder；本实验的局部校准用冻结预训练 encoder（为了跨折可比）。预训练特征的邻域质量更差，DEV base VRRC 的 CCC 也从 0.4427 降到 0.4430（base 接近），但局部邻域的可分性下降。
3. **局部斜率没有独立信息**：Local Affine ≈ Local Intercept，说明在 OOF 残差上估计 `γ` 没有带来额外收益；这与“局部变化压缩主要是不可预测噪声”一致。
4. **400 vs 500 标签的分布差**：OOF V 均值高 0.0126，局部残差映射迁移到 500-label DEV 时会系统性偏移。

---

## 4. 决策与下一步

| 问题 | 决策 |
|---|---|
| A-VRRC 是否作为 M1 的第二个机制？ | **NO-GO**；Local Affine ≈ Local Intercept ≈ VRRC，DEV 增益不迁移到 OOF |
| 是否进入 P3（训练侧伪标签校准）？ | **不进入**；P3 的前提（P2 有可部署独立收益）不成立 |
| 是否继续调 k / ridge / shrink / 加方差分支？ | **不继续**；会回到反复堆模块 |
| 下一步 | 诊断**预测表示的信息瓶颈**，而不是继续做校准 |

### 建议的信息瓶颈诊断（P2.5）

本报告 2.4 的诚实 OOF probe 已经给出第一层证据：**线性 probe ≈ 模型 raw**，回归头不是瓶颈。下一步应做：

1. **标签噪声上限**：用 EmoBank 的标注一致性/分半可靠性估计不可约噪声；如果标签本身噪声很大，预测向均值收缩是 MAE 最优，CCC 的损失是统计不可避免的；
2. **train/dev 拟合差**：比较模型在 500 TRAIN 与 DEV 上的残差方差、预测方差比，判断是过拟合还是欠拟合；
3. **冻结预训练特征的 ridge probe**：与 fold-encoder probe 对比，量化微调带来的表示增益；
4. **更强的表示学习路线**：对比 CLSS（对比 + 谱排序）、ICML 2026 谱特征回归、DKD 分布对齐；这些是“增加可预测信息”，不是后处理校准。

只有在 probe/标签噪声诊断显示“仍有可预测方差未被模型利用”时，才值得设计新的训练机制；否则应接受 MAE–CCC 权衡，把论文定位为“锚点校正 + 系统性诊断”，而不是强行增加第二个组件。

---

## 5. 复现

```bash
cd /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-avrrc
PY=/root/autodl-tmp/rankup_project/envs/rankup/bin/python

# P1: 5-fold OOF (fixed 10 epochs / 96 JACF steps, no DEV)
bash scripts/run_oof_all.sh

# aggregate
$PY scripts/vad_research/aggregate_oof.py --oof-dir outputs/oof --out-dir outputs/oof

# P2: honest deployable calibration
$PY scripts/vad_research/p2_calibration.py \
  --data-root /root/autodl-tmp/rankup_project/data/emobank_processed \
  --oof outputs/oof/oof_all.parquet \
  --fold-manifest outputs/p0/fold_manifest.json \
  --ckpt /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-source/saved_models/vad_research/alpha_fixed075_s0/model_best.pth \
  --pretrain-path /root/autodl-tmp/rankup_project/pretrained/roberta-base \
  --out-dir outputs/p2_honest --base-feature-source model --cal-feature-source pretrained --device cuda

# paired bootstrap
$PY scripts/vad_research/bootstrap_p2.py \
  --pred outputs/p2_honest/p2_dev_predictions.parquet \
  --data-root /root/autodl-tmp/rankup_project/data/emobank_processed \
  --out outputs/p2_honest/bootstrap.json
```

产物：

- `outputs/oof/oof_all.parquet`（500 条 OOF）
- `outputs/oof/oof_metrics.json`
- `outputs/p2_honest/p2_calibration.json`
- `outputs/p2_honest/p2_dev_predictions.parquet`
- `outputs/p2_honest/bootstrap.json`
- 每折：`outputs/oof/fold{0..4}/`（checkpoint、预测、特征、隔离审计）
