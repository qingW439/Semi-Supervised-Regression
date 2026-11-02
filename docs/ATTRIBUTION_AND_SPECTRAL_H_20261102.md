# 实验 A/B：JACF 训练收益归因 + 无标签谱信息互补性（2026-11-02）

## 0. 结论

**实验 A：JACF 的训练端 MAE 收益成立，且不是推理校准造成的。**

- C（JACF raw）优于 A（匹配监督 raw）：三种子全部为正，平均 **+0.005332**；
- D（JACF + VRRC-fixed）优于 B（监督 + VRRC-fixed）：三种子全部为正，平均 **+0.003066**；
- 逐种子配对 bootstrap 的 ΔMAE 95% CI 均不跨 0；
- 但 JACF 的 CCC 比监督低约 **0.020**，R² 高约 **0.04**——是“MAE/R² 换 CCC”的权衡，不是 Pareto 全面改进。

**实验 B：当前核谱实现 NO-GO。**

- H2（TRAIN+ULB 谱特征）**没有优于 H0**（原始冻结特征 Ridge）：ΔMAE +0.00433，ΔCCC −0.1256；
- H2 只优于 H1（仅标注谱特征，ΔMAE −0.00222）和 H3（等维随机 RBF 特征，ΔMAE −0.00230）；
- 因此“无标签谱信息有相对增益”成立，但当前谱表示仍不如原始 RoBERTa 特征，不能作为 M1 组件。

→ **保留 JACF 作为有效训练机制；停止当前核谱分支；M1 仍未完成**，缺一个能挽回 CCC 的结构保持机制。

---

## 1. 实验 A：JACF 训练收益归因

### 1.1 四组设计

| 组 | 训练 | 推理 |
|---|---|---|
| A | 匹配监督（λ_u=0，其余与 JACF 相同） | Raw |
| B | 匹配监督 | VRRC-fixed |
| C | JACF fixed α=0.75（λ_u=1） | Raw |
| D | JACF fixed α=0.75 | VRRC-fixed |

- 三种子 0/1/2；同一 seed 的 P1 初始化、标注 batch 顺序、优化器、160 steps、选模规则一致；
- 监督组 λ_u=0，仍保留同样的 teacher/前向计算，但无无标签梯度；
- VRRC-fixed：kNN α=0.4、k=20、c0=0.3、scale 1.2，中心化用各模型在 7562 条 ULB 上的预测均值（不使用 DEV 批均值）。

### 1.2 逐种子 DEV Macro-MAE

| seed | A sup raw | B sup VRRC | C JACF raw | D JACF VRRC |
|---|---:|---:|---:|---:|
| 0 | 0.173421 | 0.169740 | 0.170725 | 0.167875 |
| 1 | 0.175142 | 0.167947 | 0.169339 | 0.164377 |
| 2 | 0.176433 | 0.168372 | 0.168935 | 0.164608 |
| **mean** | **0.174999** | **0.168686** | **0.169666** | **0.165620** |

### 1.3 收益分解（正 = JACF 更好）

| 量 | seed 0 | seed 1 | seed 2 | mean |
|---|---:|---:|---:|---:|
| Δtrain_raw = A−C | +0.002695 | +0.005803 | +0.007498 | **+0.005332** |
| Δtrain_cal = B−D | +0.001864 | +0.003570 | +0.003763 | **+0.003066** |
| Δinfer_sup = A−B | +0.003681 | +0.007195 | +0.008062 | +0.006313 |
| Δinfer_jacf = C−D | +0.002850 | +0.004962 | +0.004327 | +0.004046 |

逐种子配对 bootstrap（2000 次，按 DEV 样本重采样，原始 [1,5] 尺度）：

| seed | Δtrain_raw 95% CI | Δtrain_cal 95% CI |
|---|---|---|
| 0 | [+0.00112, +0.00419] | [+0.00040, +0.00337] |
| 1 | [+0.00373, +0.00799] | [+0.00185, +0.00540] |
| 2 | [+0.00594, +0.00912] | [+0.00230, +0.00517] |

- 三种子全部为正，CI 均不跨 0；**JACF 训练端收益成立**。
- `Δtrain_cal > 0` 说明在**同一 VRRC** 下 JACF 仍更好，排除了“收益只来自推理校准”的解释。
- 交互项 `(B−A) − (D−C) ≈ +0.00227`：推理校准对监督组的增益更大，说明 JACF 已部分覆盖了 VRRC 所补的信息，两者有重叠但非替代。

### 1.4 CCC / R² 代价

| 组 | Macro-CCC | Macro-R² |
|---|---:|---:|
| A sup raw | **0.4493** | 0.1801 |
| B sup VRRC | **0.4555** | 0.2303 |
| C JACF raw | 0.4286 | **0.2235** |
| D JACF VRRC | 0.4351 | **0.2520** |

- JACF 的 MAE 和 R² 更好，但 CCC 低约 **0.020–0.021**；
- 逐种子 ΔCCC（A−C）约 +0.016 ~ +0.024，方向一致；
- 这与此前结构诊断一致：JACF 的锚点融合降低 MAE，但压缩了预测变化，CCC 受损。

**判定**：JACF 是有效的**半监督训练机制（以 MAE/R² 为主）**，但不是全面优于监督；不能声称“同时改善所有指标”。如果论文主指标是 MAE，它可以成立；如果要求 CCC 不退化，则还缺一个结构保持机制。

---

## 2. 实验 B：H0/H1/H2/H3 谱信息互补性

### 2.1 设计

- 冻结**未经当前 500 标签微调**的 RoBERTa 特征；
- 5 折 OOF；Ridge α 在 400 内部 4 折 CV 选择；
- H1/H2/H3 的 RBF 带宽 γ **只在 TRAIN-400 上估计一次**，三者共用，避免 H2 因不同 γ 获益；
- H2 的 landmark 来自 TRAIN-400 + ULB-7562；H1 只来自 TRAIN-400；H3 为等维（400）随机 RBF 特征。

### 2.2 结果

| 方法 | OOF MAE | OOF CCC | OOF R² |
|---|---:|---:|---:|
| H0 原始冻结特征 + Ridge | **0.182099** | **0.3714** | **0.2290** |
| H1 仅标注谱特征 | 0.188648 | 0.1662 | 0.1305 |
| H2 标注+无标签谱特征 | 0.186426 | 0.2458 | 0.1681 |
| H3 等维随机 RBF 特征 | 0.188722 | 0.2400 | 0.1444 |

差值：

| 对比 | ΔMAE | ΔCCC | ΔR² |
|---|---:|---:|---:|
| H2 − H0 | +0.004328（更差） | −0.1256 | −0.0610 |
| H2 − H1 | −0.002221（更好） | +0.0796 | +0.0376 |
| H2 − H3 | −0.002295（更好） | +0.0058 | +0.0237 |

- **H2 没有通过准入**：它既不如 H0 的 MAE，也远不如 H0 的 CCC；
- H2 相对 H1/H3 的增益说明“无标签谱结构”确实带来了一些信息，但该信息不足以弥补谱映射本身对原始表示的损失；
- 因此当前 Nyström 谱实现应判定为 **NO-GO**，不进入 M1。

---

## 3. 决策

| 问题 | 结论 |
|---|---|
| JACF 是否有独立训练收益？ | **有**（raw +0.00533；匹配 VRRC +0.00307；3/3 seeds） |
| JACF 是否是全面改进？ | **否**；CCC 低约 0.020，MAE/R² 更好 |
| 当前核谱实现是否可用？ | **NO-GO**；H2 不如 H0 |
| 是否组合 JACF + 谱特征？ | **不组合**；B 未通过，且组合不自动产生创新 |
| M1 是否完成？ | **未完成**；JACF 是有效机制，但缺 CCC/结构保持部分 |

### 下一步的三个选项

1. **接受 JACF 的 MAE–CCC 权衡，冻结 M1 为 JACF**：论文贡献为“锚点伪监督 + 系统性诊断”，但 novelty 与 SimRegMatch/UCRC 重叠，需要正面对比。
2. **补一个训练侧结构保持机制**：目标是在 JACF 基础上把 CCC 拉回监督水平，同时保留 MAE 收益；此前的 A-VRRC/局部仿射、分歧监督、谱特征都失败，需要新的机制，而不是继续调后处理。
3. **先做 α 的 Pareto 扫描**（α=0.55/0.60/0.65/0.70/0.75，3 seeds）：确认是否存在同时接近监督 CCC、又保留大部分 MAE 收益的 JACF 配置。这是低成本、可直接支持论文的补充，不是新方法。

我建议先做第 3 项（低成本、明确），再决定第 1 或第 2 项。

---

## 4. 复现

```bash
cd /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-avrrc
PY=/root/autodl-tmp/rankup_project/envs/rankup/bin/python

# 实验 A：训练 6 个模型（sup/jacf × 3 seeds）
bash scripts/run_attribution.sh

# 实验 A：冻结 VRRC 评价 + 分解
$PY scripts/vad_research/attribution_eval.py \
  --data-root /root/autodl-tmp/rankup_project/data/emobank_processed \
  --pretrain-path /root/autodl-tmp/rankup_project/pretrained/roberta-base \
  --ckpt-root /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-avrrc/saved_models/vad_research \
  --out-dir outputs/attribution --device cuda

# 实验 B：H0-H3
$PY scripts/vad_research/spectral_h_pilot.py \
  --data-root /root/autodl-tmp/rankup_project/data/emobank_processed \
  --oof-dir outputs/oof --manifest outputs/p0/fold_manifest.json \
  --pretrain-path /root/autodl-tmp/rankup_project/pretrained/roberta-base \
  --out outputs/diagnostics/spectral_h.json --device cuda
```
