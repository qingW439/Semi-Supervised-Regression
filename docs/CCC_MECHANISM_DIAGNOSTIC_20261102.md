# JACF CCC 下降的机制归因（D1–D4，2026-11-02）

## 0. 结论

**JACF 的 CCC 损失几乎全部来自偏差修正因子 \(C_b\)，而不是 Pearson 相关。**

- 三种子 step=96 的 Raw 结果：JACF 的 Pearson 与监督几乎相同（0.4807 vs 0.4789，甚至略高），但 \(C_b\) 从 0.9298 降到 0.8787，预测/真实 SD 比从 0.715 降到 0.617；
- 因此 CCC 从 0.4505 降到 0.4300，几乎完全由**预测方差收缩**解释，不是“连续关系/排序信息丢失”；
- 匹配 Mean Teacher 只造成很小的 CCC 损失（0.4473 vs 0.4505，−0.0032），JACF 的损失（−0.0205）明显更大，说明**压缩主要来自锚点目标，而不是普通半监督一致性训练**；
- 伪标签结构诊断显示：JACF target 的 MAE 最低（0.1661），但 SD 比只有 0.412，CCC 只有 0.3418；锚点目标本身是“更准但更平”的目标，学生继承了这种压缩（SD 比 0.631），但无法完全恢复。

→ **缺陷定位：锚点伪监督目标的方差压缩，传递到学生输出；不是编码器丢信息，也不是单纯的均值偏移。**
→ **不应做输出后处理缩放；如果要做机制，应在训练目标/训练约束层面处理方差，而不是再改推理校准。**

---

## 1. D1：CCC 分解（监督 vs JACF raw，step=96，三种子）

恒等式 \(CCC = r \times C_b\) 已验证（数值误差 <1e-6）。

| 维度 | 方法 | Pearson r | \(C_b\) | CCC | SD 比 | 均值偏差 |
|---|---|---:|---:|---:|---:|---:|
| V | 监督 | 0.6835 | 0.9756 | 0.6668 | 0.807 | −0.0112 |
| V | JACF | 0.6886 | 0.9435 | 0.6497 | 0.719 | +0.0143 |
| A | 监督 | 0.4706 | 0.9147 | 0.4305 | 0.685 | +0.0418 |
| A | JACF | 0.4672 | 0.8595 | 0.4015 | 0.588 | +0.0356 |
| D | 监督 | 0.2827 | 0.8991 | 0.2542 | 0.652 | +0.0204 |
| D | JACF | 0.2864 | 0.8332 | 0.2388 | 0.545 | +0.0171 |
| 宏平均 | 监督 | 0.4789 | 0.9298 | 0.4505 | 0.715 | +0.0170 |
| 宏平均 | JACF | **0.4807** | **0.8787** | 0.4300 | **0.617** | +0.0223 |

差值（JACF − 监督）：

| 维度 | ΔPearson | Δ\(C_b\) | ΔCCC | ΔSD 比 |
|---|---:|---:|---:|---:|
| V | +0.0051 | −0.0321 | −0.0171 | −0.088 |
| A | −0.0034 | −0.0553 | −0.0289 | −0.097 |
| D | +0.0036 | −0.0658 | −0.0154 | −0.107 |
| 宏平均 | **+0.0018** | **−0.0510** | **−0.0205** | **−0.098** |

- Pearson 没有下降；A 维略降 0.0034，但 V/D 上升。
- \(C_b\) 全面下降，D 维最严重（−0.0658）；SD 比全面下降 0.09–0.11。
- 均值偏差很小且没有一致方向，不是主因。

**判定：属于“Pearson 稳定、\(C_b\) 下降”的情况 → 数值分布/方差匹配问题，不是相关性信息丢失。**

---

## 2. D2：匹配 Mean Teacher 对照（step=96，三种子）

| 方法 | Raw MAE | Raw CCC | Pearson r | \(C_b\) | SD 比 | VRRC MAE | VRRC CCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| 监督 S | 0.175803 | **0.4505** | 0.4789 | 0.9298 | 0.715 | 0.169099 | **0.4569** |
| Mean Teacher T | 0.174191 | 0.4473 | 0.4853 | 0.9096 | 0.682 | 0.168323 | 0.4535 |
| JACF J（α=0.75） | **0.170472** | 0.4300 | 0.4807 | 0.8787 | 0.617 | **0.166261** | 0.4363 |

- 普通 Mean Teacher 相对监督：MAE −0.0016，CCC −0.0032，\(C_b\) −0.0202，SD 比 −0.033；
- JACF 相对监督：MAE −0.0053，CCC −0.0205，\(C_b\) −0.0510，SD 比 −0.098；
- **JACF 的压缩远大于普通 Mean Teacher**，因此不能把 CCC 损失全部归因于“半监督一致性训练的共同缺陷”；锚点目标是主要增量来源。
- T 的 Pearson 甚至高于 S（0.4853 vs 0.4789），再次说明半监督训练没有丢失相关性，损失在 \(C_b\)。

---

## 3. D3：编码器探针（本轮不执行）

按预注册条件，D3 只在“Pearson 明显下降或表征信息损失”时执行。D1 显示 Pearson 稳定/略升，因此**没有证据表明 JACF 编码器丢失了可线性读出的连续信息**。

此前 R0–R4 的诚实 OOF probe 也显示：折内 encoder 的 Ridge 与模型 raw 几乎相同（0.181627/0.4328 vs 0.181740/0.4267），回归头不是主要瓶颈。结合本轮 Pearson 稳定，**优先检查目标/输出尺度，而不是重新训练编码器或加表示模块**。

---

## 4. D4：Teacher / Anchor / JACF target / Student 的结构诊断（DEV，仅诊断）

| 信号 | MAE | CCC | Pearson r | \(C_b\) | SD 比 | 均值偏差 |
|---|---:|---:|---:|---:|---:|---:|
| Teacher（MC 均值） | 0.170172 | 0.4081 | 0.4704 | 0.8560 | 0.570 | −0.0061 |
| Anchor（kNN） | 0.169192 | 0.3076 | 0.4382 | 0.6614 | 0.387 | +0.0108 |
| JACF target（0.25T+0.75A） | **0.166062** | 0.3418 | 0.4694 | 0.6948 | 0.412 | +0.0066 |
| Student（训练后 raw） | 0.170725 | 0.4403 | 0.4881 | 0.8896 | 0.631 | +0.0252 |

- JACF target 的 MAE 最低（0.1661），但 SD 比只有 0.412，CCC 只有 0.3418；锚点目标更极端（SD 比 0.387）。
- 学生没有简单复制 target：它把 CCC 从 0.342 提升到 0.440、SD 比从 0.412 提升到 0.631；但相对监督仍压缩（0.631 vs 0.715），CCC 仍低 0.020。
- **链条清晰：anchor 目标更准但更平 → JACF target 继承了“低方差、高 MAE 准确度” → 学生学到更平的输出 → CCC 损失。**

---

## 5. 机制判定与决策矩阵

| 诊断结果 | 本轮证据 | 决策 |
|---|---|---|
| Pearson 稳定，\(C_b\) 降低 | **是**（Δr +0.0018，ΔCb −0.0510） | 优先处理训练目标的分布/尺度，不做输出后处理 |
| Pearson 下降，encoder 读出变差 | 否 | 不做表示保持模块 |
| Student 相关性低但 encoder 正常 | 否（student r 反而高于 sup） | 不重新训练大 encoder |
| Mean Teacher 已有相同 CCC 损失 | 否（MT 仅 −0.0032，JACF −0.0205） | JACF 锚点目标是主要增量来源 |
| 找不到稳定具体原因 | 否 | 可以进入“目标方差约束”的机制设计 |

**结论：缺陷是锚点目标的方差压缩，而不是相关性/表示丢失。**
这意味着下一个机制应当研究：

> 如何构造一个**保留锚点 MAE 收益、但恢复目标方差**的连续伪监督目标，并让这种目标在训练中转化为学生 raw 输出的方差/CCC 改善。

这是训练目标层面的机制，不是后处理；但**直接加 \(1-CCC\) 损失或全局方差缩放并不构成足够强的新创新**，且可能只是重复已有 CCC/variance regularization 工作。需要在“目标构造”上提出可验证的新形式，而不是简单加 loss。

---

## 6. 下一步建议（有终点）

1. **不再做 α、后处理缩放、A-VRRC、可靠性、分布头、谱特征**；这些均已 NO-GO。
2. 如果继续 M1，只做一个**方差保持的锚点目标**候选：
   - 目标：在锚点融合时，把锚点分量在局部/全局尺度上对齐到教师或标注分布，使 target 的 MAE 不退化、SD 比接近 teacher；
   - 只在训练侧使用，评价 student raw 的 MAE/CCC/方差比；
   - 预注册：raw MAE ≤ JACF（0.1705）且 CCC ≥ 0.4400（或至少 ≥ JACF +0.01），三种子。
3. 如果该目标机制不能同时保留 MAE 与恢复 CCC，则停止第二个组件；JACF 作为 MAE/R² 机制保留，论文如实报告权衡。
4. 无论结果如何，不再从 DEV 反复挑配置；最终需要独立确认性评价。

---

## 7. 复现

```bash
cd /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-avrrc
PY=/root/autodl-tmp/rankup_project/envs/rankup/bin/python

# D1
$PY scripts/vad_research/ccc_decomposition.py \
  --pred outputs/alpha_sweep/alpha_sweep_predictions.parquet \
  --out outputs/alpha_sweep/ccc_decomposition.json

# D2: matched Mean Teacher (3 seeds, step 96)
bash scripts/run_mt.sh
$PY scripts/vad_research/mt_eval.py \
  --data-root /root/autodl-tmp/rankup_project/data/emobank_processed \
  --pretrain-path /root/autodl-tmp/rankup_project/pretrained/roberta-base \
  --ckpt-root /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-avrrc/saved_models/vad_research/alpha_sweep \
  --out-dir outputs/alpha_sweep --device cuda

# D4
$PY scripts/vad_research/pseudo_target_diag.py \
  --data-root /root/autodl-tmp/rankup_project/data/emobank_processed \
  --pretrain-path /root/autodl-tmp/rankup_project/pretrained/roberta-base \
  --ckpt saved_models/vad_research/alpha_sweep/a0.75_s0/model_step96.pth \
  --out outputs/alpha_sweep/pseudo_target_diag.json --device cuda
```
