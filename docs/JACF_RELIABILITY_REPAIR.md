# JACF 可靠性修复准入报告（2026-10-30）

## 0. 结论：Q3 不通过（NO-GO）

保留 JACF 原融合公式，只用 TRAIN 内部真实误差重新校准可靠性 r。结果：

- Q3（校准后 r'）对真实误差的 Spearman 为 **+0.119**（方向错误，越高可靠性对应越高误差）；
- AURC 0.0454，**不如** Q1 原 r（0.0427）、Q2 锚点方差（0.0413）、甚至 Q0 均匀（0.0441）；
- 相对 Q1 的 bootstrap 差值：Spearman CI [-0.020, +0.200]，AURC CI [-0.0005, +0.0056]，均跨 0，**没有显著改善**；
- 相对 Q2 teacher：Spearman CI [-0.153, -0.003]，**显著更差**。

→ 按预注册标准：**不启动 R1/R2 训练；JACF 的可靠性模块应删除或视为无效。**

## 1. 协议修正与数据安排

按本轮要求，锚点库统一使用 **fit 320**，而不是 meta 80：

| 数据 | 用途 |
|---|---|
| fit 320 | 参考模型训练 + 标注锚点库 |
| meta 80 | 生成样本外伪标签与真实误差，拟合误差预测器 |
| probe 100 | 仅评价 Q0–Q3，不参与拟合 |

教师（参考模型）没有使用 meta/probe 标签训练；meta/probe 的锚点检索只来自 fit 320，因此真实误差是严格样本外的。

## 2. 误差预测器

\[
\widehat e_{i,d}=\operatorname{softplus}\left(b_d+w_T\log v^T_{i,d}+w_A\log v^A_{i,d}+w_D\log(1+D_{i,d})\right)
\]

三维共享 \(w_T,w_A,w_D\)，仅保留各维截距 \(b_d\)；L2 强度在 meta 内部 5 折 CV 选择（结果 \(\lambda=10^{-3}\)，CV MSE 0.00214）。

拟合结果：

- \(w_T=0.221\)、\(w_A=0.247\)、\(w_D=0.020\)；
- \(b=[0.0070, 0.0044, -0.0552]\)；
- \(e_0=[0.0361, 0.0373, 0.0278]\)；
- \(r'=e_0/(e_0+\widehat e)\)。

## 3. Probe 评价（100 条文本，配对 bootstrap 按文本重采样）

| 评分 | Spearman(r, 误差) | AURC ↓ | 高 r 高误差比例 | 四分位平均误差 |
|---|---|---|---|---|
| Q0 均匀 | N/A（无排序） | 0.0441（随机顺序） | N/A | 0.0414（常数） |
| Q1 原 JACF r | **+0.029** | 0.0427 | 0.227 | 0.0367 / 0.0523 / 0.0471 / 0.0400 |
| Q2 仅教师方差 | **+0.200** | 0.0473 | 0.320 | 0.0352 / 0.0477 / 0.0442 / 0.0490 |
| Q2 仅锚点方差 | -0.042 | **0.0413** | 0.200 | 0.0484 / 0.0470 / 0.0414 / 0.0394 |
| **Q3 校准 r'** | **+0.119** | 0.0454 | 0.293 | 0.0397 / 0.0489 / 0.0408 / 0.0467 |

要点：

1. **Q3 方向错误**：Spearman 为正，意味着 r' 越高、真实误差越大；校准器在 meta 上拟合出的误差排序没有迁移到 probe。
2. **Q1 原 r 也没有负相关**（+0.029），与上一轮 meta-as-bank 的 -0.080 符号不一致，说明该信号本身不稳定。
3. **Q2 教师方差是反预测**（+0.200）：教师越“自信”（方差小），误差反而越大，典型的 confidently-wrong。
4. **Q2 锚点方差**是唯一方向正确且 AURC 最低的评分，但 Spearman 只有 -0.042，几乎无区分力。
5. 所有评分的 AURC 都在 0.041–0.047 的窄区间，probe 误差本身非常集中（均值约 0.041），可靠性的排序空间很小。

### Bootstrap（Q3 − 基准）

| 对比 | Spearman 差值 95% CI | AURC 差值 95% CI | 判定 |
|---|---|---|---|
| Q3 − Q1 | [-0.020, +0.200] | [-0.0005, +0.0056] | 跨 0，无显著改善 |
| Q3 − Q2 teacher | [-0.153, -0.003] | [-0.0040, +0.0004] | Q3 方向优于 Q2（+0.119 vs +0.200），但两者都为正、且 Q3 未超过 Q1；不能算通过 |

解释修正：Spearman(H, 误差) 越负越好。Q3 的 +0.119 比 Q2 教师方差的 +0.200 更接近 0，因此 Q3 相对 Q2 **方向更好**，而不是“显著更差”；但 Q3 仍然为正，且 AURC 与 Q1 无显著差异，所以总体仍是 NO-GO。

## 4. 决策

按本轮预注册规则：

- Q3 相对原 r 没有明确排序优势，AURC 也没有改善；
- 方向甚至相反；
- 现有三类证据（教师方差、锚点方差、标准化分歧）在 probe 上没有表现出可用的误差预测能力。

→ **Q3 准入失败，不启动 R1/R2 训练。**

进一步结论：

- JACF 的可靠性模块没有可验证的作用，应与 A4（去可靠性）一致地删除；
- 原 JACF 的正结果（0.168069 / 0.4447）来自融合目标本身，不是可靠性；
- 但这与 A0–A4 消融（A1 仅锚点 MAE 更好、A4≈A3）一起说明：**JACF 的“联合估计 + 可靠性”叙事不成立**。

## 5. 下一步建议

1. **不再拟合新的可靠性公式**：80 条 meta 上拟合的误差模型无法迁移到 100 条 probe；换公式不会创造新信息。
2. M1 的选择回到上一轮结论：
   - 若以 MAE 为主：A1 仅锚点 + VRRC（0.166864），CCC 0.4327；
   - 若要求 MAE/CCC 双改善：JACF（0.168069 / 0.4447），但删除可靠性主张。
3. 研究重心转向 **M2：VRRC 的四级诊断**（无校准 / 固定 1.2 / 简单全局 / 局部仿射），先确认推理校准的独立贡献，再决定是否投入 OOF。
4. 跨预算与 TEST 在 M1 冻结后再做。

## 6. 复现

```bash
cd /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-relcal
PY=/root/autodl-tmp/rankup_project/envs/rankup/bin/python
$PY scripts/vad_research/reliability_calibration.py \
  --manifest /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-m1-validity-v2/docs/m1/M1_TARGET_UTILITY_VALIDITY_V2_RUN_MANIFEST.json \
  --ckpt /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-acvr/outputs/reference_320_model_only.pth \
  --data-root /root/autodl-tmp/rankup_project/data/emobank_processed \
  --pretrain-path /root/autodl-tmp/rankup_project/pretrained/roberta-base \
  --out outputs/reliability_calibration.json --k 20 --tau 0.1 --mc-passes 5 --bootstrap 1000
```
