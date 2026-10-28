# EmoBank M0 核谱机制：最后一次 TRAIN 内准入

**判定：NO-GO。** 在预先锁定的五折 TRAIN-only 协议下，B2（标注＋7562 无标签核谱）的 OOF Macro-MAE 为 **0.185951**，高于 B1（仅标注核谱）的 **0.183908**，差值 B2−B1 为 **+0.002043** 个原始 VAD 单位。预设规则规定总体 OOF MAE 未低于 B1 即 NO-GO。因此停止当前核谱定义，不进入 DEV、seed 1/2 或新参数搜索。本结果不能说明所有核谱半监督方法均无效。

## 计算边界与锁定协议

- 工作树：`/root/autodl-tmp/rankup_project/Semi-Supervised-Regression-spectral-readiness`，分支 `vad-m0-spectral-readiness`。开始时 Git 工作树干净，HEAD 为 `e7db75f530cc3fafc08b9be28c2e64e200e464d2`。历史资产未覆盖。
- 计算前先写入 `docs/m0/SPECTRAL_TRAIN_ONLY_PROTOCOL.json`，SHA256 `cbba03833639a455f734ccec771aca5f1912c5988a925bad91819bdacf39128a`。只读取 seed 0 缓存中的 `lb_x`、`ulb_x`、`lb_y`、`lb_ids`、`ulb_ids`，并与 TRAIN parquet 的 ID 和归一化标签逐项比对。脚本没有读取 `dev_x`、`dev_y`、DEV parquet 或 TEST；没有启用 GPU。
- 五折 `KFold(n_splits=5, shuffle=True, random_state=0)`；每折 400 条拟合标签、100 条验证标签。每折 gamma 仅由该折 400 条**标注拟合特征**计算，B1/B2 共享。以种子 `20261028+fold` 从 L2 归一化表示抽取 100000 对索引，排除自配对与零平方距离，令 `gamma=ln(2)/median(d²)`。未来 500 条 TRAIN 全量拟合的 gamma 用同一规则和种子 `20261028` 计算，不涉及 DEV。
- Nyström 锚点数固定 256；每折用种子 `fold` 从该折拟合标注特征选取，B1/B2 锚点完全一致且顺序一致。B1 核谱基只见 400 条拟合标注特征；B2 在相同前缀之后加入全部 7562 条无标签 TRAIN 特征。验证折特征只用于投影，验证折标签只用于评分。Ridge 拟合仅用对应 400 个真实标签。
- 候选 rank 固定 `{8,16,32,64}`，alpha 固定 `{0.01,0.1,1,10,100,1000}`。先按 B1 五折平均原始尺度 Macro-MAE 选共享 rank（各 rank 的 alpha 也仅在 B1 TRAIN 折中选择；同分取较小值）。B2 不参与 rank 选择。固定 rank 后，B1/B2 分别以各自 TRAIN 折分数选 alpha，同分取较小值。没有扩充候选或依据结果修改协议。
- OOF 预测与参数选择使用同一五折，因此是**选择条件下的 TRAIN 折外估计**，并非独立的泛化验证。rank 只从 B1 评分选择，B2 没有利用自身分数选 rank；这一安排使 B2 对 B1 的比较相对保守。准入结论严格限于该预设门槛。

## 锁定参数与数值诊断

| 项目 | 结果 |
|---|---:|
| 500 条标注 TRAIN 非零平方距离中位数 | 0.097655563 |
| 未来全量拟合 gamma（本轮未拟合） | 7.097877070 |
| 五折 gamma | 7.049661、7.197340、7.150051、7.136718、6.992779 |
| B1 各 rank 最优 alpha | 8/16/32/64 均为 100 |
| B1 各 rank 五折 Macro-MAE | 0.199372、0.190255、0.187402、0.183908 |
| 共享 rank | **64** |
| 最终 B1 / B2 alpha | **100 / 100** |

所有距离中位数为正且有限，五折及 OOF 特征/预测均有限，500 个 ID 唯一且恰好每折 100 个；两组预测均无 [1,5] 越界值。256 锚点核矩阵最小特征值在两折接近零：第 0 折约 `9.07e-13`，第 3 折约 `1.20e-12`，分别有 2、1 个特征值被既有 `1e-8` 数值地板截断；其余三折没有截断。该事实说明存在近重复锚点方向，但没有观察到非有限输出。对每折前 128 个拟合协变量的 64 维投影核诊断，B1 相对 Frobenius 误差为 **3.67%–4.08%**，B2 为 **3.85%–4.29%**；这是固定探针上的近似误差，不能解释为全数据误差，也没有用于改动配置。

## TRAIN OOF 结果

原始 VAD [1,5] 尺度；`Δ` 均为 B2−B1，MAE 越小越好。

| 维度 | B1 MAE | B2 MAE | Δ MAE | B1 CCC | B2 CCC | B1 Pearson | B2 Pearson |
|---|---:|---:|---:|---:|---:|---:|---:|
| V | 0.227450 | 0.230920 | +0.003470 | 0.368468 | 0.374216 | 0.491831 | 0.486813 |
| A | 0.176684 | 0.179286 | +0.002602 | 0.268015 | 0.258329 | 0.398913 | 0.379040 |
| D | 0.147590 | 0.147647 | +0.000056 | 0.247224 | 0.261519 | 0.380957 | 0.385933 |
| Macro-MAE | **0.183908** | **0.185951** | **+0.002043** | — | — | — | — |

五折配对 MAE 差值（B2−B1）：

| 折 | V | A | D | Macro |
|---:|---:|---:|---:|---:|
| 0 | +0.004705 | −0.001109 | −0.003832 | −0.000072 |
| 1 | +0.007962 | +0.000100 | +0.000337 | +0.002800 |
| 2 | −0.001741 | +0.006697 | +0.004693 | +0.003217 |
| 3 | +0.003917 | +0.001844 | +0.003530 | +0.003097 |
| 4 | +0.002505 | +0.005459 | −0.004445 | +0.001172 |

只有 **1/5** 折 Macro-MAE 略微改善。V、A、D 三维的全量 OOF MAE 均未改善；D 的 CCC 增加约 0.0143，但 MAE 基本持平，不能抵消总体退化。B2/B1 预测标准差之比 V/A/D 分别为 **1.040/1.020/1.060**，没有预测收缩迹象；本次失败主要表现为误差未降，而非数值崩溃。每折 CCC 与 Pearson 的配对差值也保存在参数 JSON 中。

## 历史参照、复现与停止条件

原有 B0 结果文件保留原样，仅作为历史资产；本次**没有打开其 DEV 结果或重新拟合 B0**。同样未进行完整 B1/B2 DEV 拟合、seed 1/2 缓存提取或任何 TEST 操作。参数详表及全部候选 TRAIN 折分数见 `SPECTRAL_TRAIN_ONLY_PARAMS.json`；逐条真实值和 B1/B2 OOF 预测见 `SPECTRAL_TRAIN_ONLY_OOF.csv`。

从该工作树复现的命令（需已有 seed 0 TRAIN 缓存；已有输出路径会被脚本拒绝覆盖）：

```bash
cd /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-spectral-readiness
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /root/autodl-tmp/rankup_project/envs/rankup/bin/python scripts/vad_research/m0/run_spectral_train_gate.py \
  --feature-dir /hy-tmp/rankup_project/Semi-Supervised-Regression/saved_models/vad_research/m0/pretrained_features_lb500_s0 \
  --protocol docs/m0/SPECTRAL_TRAIN_ONLY_PROTOCOL.json \
  --out-json docs/m0/SPECTRAL_TRAIN_ONLY_PARAMS.json \
  --out-csv docs/m0/SPECTRAL_TRAIN_ONLY_OOF.csv
```

现有谱方法 CPU 单测：**8 passed**。此后停止当前核谱方案；不因 NO-GO 调 gamma、rank、alpha、锚点数或折划分来挽救结果。
