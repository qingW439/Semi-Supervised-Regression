# 组件级 OOF 复核：JACF 与 ACLDP 两种参照（2026-11-03 20:10）

## 1. 为什么要重做

导师指出：若最终 E0\* = Ridge，而 ACLDP 的差异信息仍来自旧的监督 MLP，则**不能继承旧结论**，
必须把「参照模型」作为方法定义的一部分重新训练与比较。本文件即该复核的设计（在任何新结果之前写下）。

## 2. 差异参照的两种合法定义

ACLDP 的损失是

\[
\mathcal L_{\text{diff}}=\frac{1}{3|\mathcal P|}\sum_{(u,v)\in\mathcal P}\sum_d
\operatorname{Huber}\!\left(\frac{(s_u^d-s_v^d)-(r_u^d-r_v^d)}{\sigma_d}\right)
\]

其中 \(r\) 是差异参照模型的预测。参照模型必须与学生在**同一 400 条标注**上训练，
否则参照会"见过"折外标签并经由无标签样本把信息漏回学生。因此参照在折内构造：

| 参照 | 定义 | 信息量 |
|---|---|---|
| `m0` | 该折的监督 P1 网络（编码器+MLP 头，10 epochs）对无标签池的预测 | 400 标签 + 同折编码器 |
| `ridge` | 在该折 400 条的**同一特征**上闭式拟合的 ridge（α 由 400 内部 5 折 CV 选） | 同上，但归纳偏置为 MSE 最优 |

两者信息预算相同、归纳偏置不同，因此「ACLDP 是否依赖 MLP 特有的差异结构」成为一个可检验命题。

## 3. 实验矩阵

3 seeds × 5 folds × {`m0`, `ridge`, `ridge_shuffle`(负控制：批内打乱参照)}，
`λ_diff=0.05`（沿用既有已验证配置，不再扫参），JACF 目标固定 α=0.75。

对照臂（同一折、同一折外样本）：
- `mlp_p1`：匹配纯监督 Control
- `jacf`：组件一（λ_u=1，无差异项）
- `ridge`：E0\* 候选（闭式读出）

## 4. 预注册判据（组件二是否成立）

对每个参照分别判定，**不得混合**：

1. 存在性：`ACLDP(ref)` 的 OOF Macro-CCC ≥ `jacf` + **0.005**，且配对 Bootstrap 95% CI 下界 > 0。
2. 安全性：OOF Macro-MAE ≤ `jacf` + **0.002**；OOF Macro-R² ≥ `jacf` − **0.010**。
3. 负控制：`ridge_shuffle` 的 CCC 必须显著低于 `ridge`，否则说明收益来自"扩大离散度"而非差异信息。
4. 稳定性：三个 seed 的 ΔCCC 方向必须一致。
5. 逐维：任一维 R² 不得低于 `jacf` 同维 −0.05。
6. 若两条参照都通过 → 记录"参照不敏感"，可选信息更少者；
   若仅其一通过 → 方法定义必须绑定该参照，并在论文中说明；
   若都不通过 → 组件二在 OOF 协议下不成立，按任务书失败表换机制（不得继续调 λ/阈值）。

## 5. 与 M1 判定的关系

- 若组件二成立：M1 = JACF + ACLDP(ref)，其 OOF 结果即本矩阵对应臂，再与 E0\* 及匹配 Control 比。
- 必须同时报告 **Student Raw**（本矩阵 `raw_*`）与 **Student Final**（`vrr_*`）；
  不得用 Final 的改善声称训练端 M1 超过 E0\*。
- 若组件二不成立：M1 不能宣称完成，创新点二改写为"锚点伪监督的方差压缩诊断与边界"，
  并在论文中如实报告（任务书允许负结果，不允许虚构）。

## 6. 复现

```bash
R=/root/autodl-tmp/rankup_project/Semi-Supervised-Regression-comp2
cd $R && bash run_acldp_oof.sh            # 45 次折训练，约 45 min
python scripts/vad_research/audit_components_oof.py
```
