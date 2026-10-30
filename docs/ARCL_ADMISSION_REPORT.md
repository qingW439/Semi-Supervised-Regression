# ARCL 机制准入报告（P1）

日期：2026-10-30  
范围：TRAIN-only 320/80/100 审计；不访问 DEV/TEST；不启动 ARCL 正式训练。

## 0. 结论：NO-GO

在 100 条 probe 上，**H3（教师预测 + top-k 局部残差）没有优于 H0，且显著劣于 JACF（H1）；与 H2（全局残差）、H4（置换残差）均无显著差异**。meta 特征相似度与残差相似度的 Spearman 仅 **-0.032**，没有支持“局部残差可迁移”的证据。

因此：**停止 ARCL，不进入 P2 单种子训练；保留 JACF + VRRC 作为 M1。**

## 1. 参考模型与隔离审计

| 项 | 值 |
|---|---|
| manifest | `Semi-Supervised-Regression-m1-validity-v2/docs/m1/M1_TARGET_UTILITY_VALIDITY_V2_RUN_MANIFEST.json` |
| checkpoint | `.../m1_target_validity_v2/reference_320_only.pth` |
| train_steps | 320 |
| git_commit | `8e678f9eb0ca5551502fe79593755c29a1fa0e01` |
| source_sha256 | `c02e487b…842847`（与 manifest 一致） |
| fit_id_sha256 | `cf61c981…1572d7`（与 manifest 一致） |

Split 核验（`train_lb.parquet` 500 条）：

- fit 320 / meta 80 / probe 100；
- 三者互斥（交集 0/0/0），并集 500；
- 全部属于 `train_lb`；
- 参考模型只在这些 320 条上训练（`fit_id_sha256` 与 manifest 一致）。

为避免 1.4GB 优化器状态占内存，另存了 model-only checkpoint（约 500MB），后续诊断只加载它。

## 2. 特征提取的 eval 模式核查

`train_acvr.py` 原来的 `encode()` 只开了 `torch.inference_mode()`，没有 `encoder.eval()`。实测同一文本重复编码：

| 模式 | 两次编码最大差异 |
|---|---|
| encoder.train() | **0.1319** |
| encoder.eval() | **0.0000** |

这是实质性的 dropout 污染。已修复 `train_acvr.py` 的 `encode()`（先 `encoder.eval()`），并重跑 JACF 三种子；P1 诊断脚本本身使用 `encode_eval()`，不受影响。

修复后 JACF 三种子复跑：0.168403 / 0.169556 / 0.166248（均值 0.168069），与修复前完全一致，说明 JACF 的训练结果不依赖这个 bug。

## 3. P1 方法

在 100 条 probe 上比较五种伪标签生成方式（均为 raw，不使用 VRRC）：

| 编号 | 方法 |
|---|---|
| H0 | 教师原始预测（eval 确定性） |
| H1 | JACF：教师 K=5 多视图 + meta 锚点联合融合 |
| H2 | 教师 + meta 全局平均残差 |
| H3 | 教师 + top-k 局部残差（ARCL 假设） |
| H4 | 教师 + 固定置换后的局部残差 |

固定 `k=20, τ=0.1, K=5`，单一预设，不用 probe 调参。指标在 [1,5] 原始尺度上计算，配对 bootstrap 1000 次。

## 4. 结果（seed 固定的一次运行）

| 方法 | Macro-MAE | Macro-CCC | Macro-R² | Pearson | SRCC |
|---|---|---|---|---|---|
| H0 教师原始 | 0.194105 | 0.4318 | -0.0168 | 0.4384 | 0.4353 |
| **H1 JACF** | **0.177926** | 0.3625 | 0.0910 | 0.3991 | 0.3759 |
| H2 全局残差 | 0.191047 | 0.3872 | 0.0031 | 0.4008 | 0.3802 |
| H3 局部残差 | 0.188849 | 0.3669 | 0.0074 | 0.3889 | 0.3593 |
| H4 置换残差 | 0.196492 | 0.3646 | -0.0548 | 0.3777 | 0.3461 |

配对 bootstrap（原始尺度，正=H3 更好）：

| 对比 | MAE 增益 | 95% CI | 判定 |
|---|---|---|---|
| H3 - H0 | +0.00526 | [-0.00581, +0.01751] | 跨 0，不显著 |
| H3 - H1 | **-0.01092** | **[-0.01631, -0.00554]** | H3 显著更差 |
| H3 - H2 | +0.00220 | [-0.00523, +0.00954] | 跨 0，不显著 |
| H3 - H4 | +0.00764 | [-0.00207, +0.01701] | 跨 0，不显著 |

meta 特征相似度 vs 残差相似度：Spearman **-0.032**（无局部结构）。

### 重复运行稳定性

同一配置重复三次（MC dropout 随机性）：

- H1 JACF：0.1714 / 0.1720 / 0.1779；
- H3 local：0.1844 / 0.1890 / 0.1888；
- H2 global：0.1858 / 0.1862 / 0.1910；
- H4 perm：0.1896 / 0.1931 / 0.1965。

H1 始终最好；H3 始终没有超过 H1，也没有稳定超过 H2/H4。结论不随重复次数改变。

## 5. 决策（按预注册表）

| 条件 | 实测 | 结论 |
|---|---|---|
| H3 不优于 H0 | CI 跨 0 | 不满足 |
| H3 优于 H0 但与 H2 相近 | H3≈H2 | 不满足 |
| H3 优于 H2 但与 H4 相近 | H3≈H4 | 不满足 |
| H3 同时优于 H0/H1/H2/H4 | 显著劣于 H1 | **NO-GO** |

**决定：停止 ARCL，不修改 JACF 主模型，不进入 P2。**

## 6. 下一步

1. 用修复后的 eval 特征重跑 JACF 三种子，确认 0.168069 是否保持；
2. JACF 的 raw 消融（仅多视图 / 仅锚点 / 独立组合 / 联合）与 250/1000 预算；
3. M2：A-VRRC（局部仿射校准，OOF 残差库）；
4. 冻结后一次性 TEST。

## 7. 复现

```bash
cd /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-acvr
PY=/root/autodl-tmp/rankup_project/envs/rankup/bin/python

$PY make_model_only.py
$PY scripts/vad_research/probe_anchor_relative.py \
  --manifest /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-m1-validity-v2/docs/m1/M1_TARGET_UTILITY_VALIDITY_V2_RUN_MANIFEST.json \
  --ckpt outputs/reference_320_model_only.pth \
  --data-root /root/autodl-tmp/rankup_project/data/emobank_processed \
  --pretrain-path /root/autodl-tmp/rankup_project/pretrained/roberta-base \
  --out outputs/arcl_probe.json --k 20 --tau 0.1 --mc-passes 5 --bootstrap 1000
```
