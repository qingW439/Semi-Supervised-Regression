# E0* 审计记录：污染排查与协议修正（2026-11-03 19:25）

## 1. 审计中发现的问题（必须先记录）

### 1.1 "冻结编码器" 实际上被动过

对 `p1_roberta_lb500_s0/model_best.pth` 与 `pretrained/roberta-base` 逐张量比对：

| 项 | 结果 |
|---|---|
| encoder 张量总数 | 199 |
| 完全一致 | 1 |
| **有差异** | **198** |
| 最大逐张量绝对差 | ≈2.4e-3（pooler 为 1.4e-1，属重新初始化） |

**结论：M0/P1 的编码器并未冻结**，在 500 条标注上被 AdamW（lr 2e-5, 10 epochs）轻微更新。
这不影响 "P1 在 dev 上的指标"（dev 从未参与训练），但**会污染任何"用 P1 checkpoint 抽特征 → 在同一 500 条上做 OOF"的做法**。

### 1.2 被作废的一组数字

用 P1-encoder 特征对 lb500_s0 做 5 折 OOF ridge，得到
`OOF MAE 0.0975 / r 0.862 / R² 0.741`，而同一批 fold 模型在 dev 上只有
`MAE 0.1678 / R² 0.240`。**巨大落差即为泄漏信号**：折外样本的标签已经被该编码器见过。

→ 该组 OOF 数字**作废，不得引用**。已改为下述协议。

## 2. 修正后的 OOF 协议（与预注册一致，仅补充实现细节）

1. **每折独立训练编码器**：`train_oof_fold.py` 对每个外层折用该折的 400 条从
   `pretrained/roberta-base` 重新训练 P1（10 epochs），再跑 JACF（96 步）。
   折外 100 条只在预测时使用（`label_isolation.json` 已记录 `holdout_labels_used_in_training=false`）。
2. **同一折的三种监督臂共享同一编码器特征**：
   - `mlp_p1`：该折 `p1_final.pth` 的头作用在该折 `features_holdout.npy`
   - `mlp_jacf`：该折 JACF student 的 `raw_*`（`oof_predictions.parquet`）
   - `ridge`：在该折 `features_labeled.npy`（400 条）上闭式拟合，α 与预处理由**该 400 条内部** 5 折 CV 选择
3. 编码器只从 base 权重出发，因此任何一折的折外样本都未被其编码器见过。
4. `train_oof_fold.py` 原有硬编码 `lb500_s0` 的缺陷已修复（新增 `--data-seed`），
   否则 seed 1/2 的 OOF 无法成立（会静默用错标签划分）。
5. seed 1/2 的 OOF 目录：`outputs/oof_seed{1,2}/fold{0..4}`；seed 0 沿用主线
   `outputs/oof/fold{0..4}`。

## 3. E0* 候选集与当前状态

| 候选 | OOF 来源 | 状态 |
|---|---|---|
| MLP-L1-160（P1 配方） | 每折 `p1_final.pth` | 审计中 |
| Ridge（同折特征） | 每折 `features_labeled/holdout.npy` | 审计中 |
| MLP-L1-640-bs64 | 需重跑 30 个折训练 | **未完成（pending）** |
| MLP-L1-1600-bs64 | 需重跑 30 个折训练 | **未完成（pending）** |

**因此 E0\\* 目前只能在 {MLP-160, Ridge} 之间做出临时判定**；在长训练 MLP 的 OOF 完成前，
不得宣布 E0\\* 已最终确定。这一点与导师意见一致（"暂缓认定 Ridge 为 E0*"）。

## 4. dev 与 OOF 的关系

- dev（1000）只用于**报告**与最终确认，不参与 α/预处理/折数/配方选择。
- 由于每折模型都会给出 dev 预测，dev 指标按 5 个折模型的预测取平均后报告，
  并明确标注这是"折模型集成"的 dev 表现，不是单一部署模型的 dev 表现。

## 5. 复现

```bash
R=/root/autodl-tmp/rankup_project/Semi-Supervised-Regression-comp2
PY=/root/autodl-tmp/rankup_project/envs/rankup/bin/python
cd $R
bash run_oof_seeds12.sh                 # seed1/2 每折 P1+JACF（已修复 --data-seed）
$PY scripts/vad_research/audit_e0_assemble.py   # 汇总三种臂的 OOF 与配对 bootstrap
```
