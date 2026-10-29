# ACVR-VAD：JACF 融合组件实验记录（2026-10-29）

## 0. 结论

**JACF（Joint Anchor-Consistency Fusion）GO。**

在 lb=500、三种子、raw + 原 VRRC 协议下：

- A1（JACF + VRRC）DEV Macro-MAE **0.168069**；
- 旧顺序组合 B1（无 ARC + GCF + LA-PLC + VRRC）**0.169769**；
- 旧最佳组合 B3（无 ARC + RDA + GCF + VRRC）**0.169665**；
- 强监督 P2 **0.175589**。

A1 相对 B1 三种子分别改善 **+0.002536 / +0.001006 / +0.001557**（全部为正），相对 B3 为 **+0.001779 / +0.001063 / +0.001945**；CCC 0.4447（B1 0.4403、B3 0.4058），R² 0.2370（B1 0.2252、B3 0.2161）。

这说明：**把 GCF 与 LA-PLC 从“先过滤、再校准”的顺序模块，改造成一个联合估计组件，确实带来了独立于推理后处理的训练侧增益。**

## 1. 第一阶段：ARC / RDA 减法消融

固定 GCF + LA-PLC/RDA + VRRC，lb=500，seed0：

| 配置 | ARC | 回归伪标签 | Macro-MAE | Macro-CCC | Macro-R² |
|---|---|---|---|---|---|
| B0 | 保留 | LA-PLC | 0.171306 | 0.4459 | 0.2149 |
| B1 | 删除 | LA-PLC | 0.170939 | 0.4470 | 0.2214 |
| B2 | 保留 | RDA | 0.170995 | — | — |
| **B3** | **删除** | **RDA** | **0.170182** | 0.4056 | 0.2170 |

结论：

1. **ARC 可以删**：B1 优于 B0、B3 优于 B2；ARC 对 MAE 是负担，对 CCC 也没有稳定贡献。
2. **RDA 与 LA-PLC 各有取舍**：RDA 的 MAE 略低但 CCC 明显差（0.4058）；LA-PLC 的 CCC 更好（0.4403）。最终 JACF 用“锚点 + 一致性”统一了这两类信息。

## 2. JACF：把 GCF 与 LA-PLC 融合成一个组件

对无标签文本的第 d 维，构造两组带不确定性的证据：

**教师多视图证据**

\[
\mu^T_{i,d}=\frac1K\sum_k p^{(k)}_{i,d},\qquad
v^T_{i,d}=\operatorname{Var}_k(p^{(k)}_{i,d})+\epsilon_T
\]

**标注锚点证据**（top-k 邻居）

\[
\mu^A_{i,d}=\sum_j w_{ij}y_{j,d},\qquad
v^A_{i,d}=\sum_j w_{ij}(y_{j,d}-\mu^A_{i,d})^2+\epsilon_A
\]

**联合融合**（不再有固定 α=0.4，也没有“先 GCF 后 LA-PLC”的顺序）

\[
\alpha_{i,d}=\frac{v^T_{i,d}}{v^T_{i,d}+v^A_{i,d}},\qquad
\tilde y_{i,d}=(1-\alpha_{i,d})\mu^T_{i,d}+\alpha_{i,d}\mu^A_{i,d}
\]

**冲突与可靠性**

\[
D_{i,d}=\frac{(\mu^T_{i,d}-\mu^A_{i,d})^2}{v^T_{i,d}+v^A_{i,d}},\qquad
c_{i,d}=e^{-D_{i,d}/2}
\]

\[
v^F_{i,d}=\frac{v^T_{i,d}v^A_{i,d}}{v^T_{i,d}+v^A_{i,d}},\qquad
r_{i,d}=c_{i,d}\frac{v_0}{v_0+v^F_{i,d}}
\]

**训练目标**

\[
\mathcal L=\frac1{3N_l}\sum_{i,d}|f_\theta(x_i)_d-y_{i,d}|
+\lambda_u\frac{\sum_{i,d}r_{i,d}|f_\theta(A_s(x_i))_d-\operatorname{sg}(\tilde y_{i,d})|}{\sum_{i,d}r_{i,d}+\epsilon}
\]

- 没有 ARC loss，没有 RDA loss；
- JACF 同时输出“伪标签 + 可靠性”，不是两个模块串联；
- 逐 V/A/D 独立，所以同一条文本对三个维度可以有不同的融合比例与可靠性。

实现：`scripts/vad_research/jacf.py`、`scripts/vad_research/train_acvr.py`、`tests_vad/test_jacf.py`（6 tests passed）。

## 3. 三种子结果（lb=500，raw + 原 VRRC）

| 模型 | Macro-MAE | Macro-CCC | Macro-R² |
|---|---|---|---|
| P2 强监督 | 0.175589 | 0.4418 | 0.1750 |
| B1 无 ARC + GCF + LA-PLC + VRRC | 0.169769 | 0.4403 | 0.2252 |
| B3 无 ARC + RDA + GCF + VRRC | 0.169665 | 0.4058 | 0.2161 |
| **A1 JACF + VRRC** | **0.168069** | **0.4447** | **0.2370** |

逐种子：

| seed | B1 | B3 | A1 |
|---|---|---|---|
| 0 | 0.170939 | 0.170182 | **0.168403** |
| 1 | 0.170562 | 0.170618 | **0.169556** |
| 2 | 0.167805 | 0.168193 | **0.166248** |

配对差值（B1 − A1）：+0.002536 / +0.001006 / +0.001557；B3 − A1：+0.001779 / +0.001063 / +0.001945。三种子全部为正。

## 4. 现在的模型结构

```
RoBERTa + 3D 回归头 + EMA 教师
        │
        ├── 监督 L1
        └── JACF（唯一训练侧组件）
              ├── 教师 K=5 多视图证据 (μT, vT)
              ├── 标注 bank 邻域证据 (μA, vA)
              └── 联合输出 (ỹ, r) → 可靠性加权 L1
        │
        └── VRRC（推理侧，暂保留原版）
```

不再有 ARC、RDA、GCF、LA-PLC 四个并列模块；RankUp 只作为基座来源和负迁移证据。

## 5. 待做

1. **A-VRRC（M2）**：用同一条邻域证据做局部仿射校准，替代固定 `s=1.2`；需要 OOF 预测库。
2. JACF 的消融：仅多视图、仅锚点、独立组合、联合估计（证明不是“加了两种信号”）。
3. 250/1000 预算验证。
4. 冻结后一次性 TEST。

## 6. 复现

```bash
cd /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-acvr
PY=/root/autodl-tmp/rankup_project/envs/rankup/bin/python

$PY -m pytest -q tests_vad/test_jacf.py

for s in 0 1 2; do
  $PY scripts/vad_research/train_acvr.py --budget 500 --seed $s --steps 160 \
    --p1-dir /root/autodl-tmp/rankup_project/Semi-Supervised-Regression/saved_models/vad_research/p1_roberta_lb500_s$s \
    --eval-knn-alpha 0.4 --eval-scale 1.2 --device cuda \
    --out saved_models/vad_research/jacf_a1_vrrc_s$s
done
```
