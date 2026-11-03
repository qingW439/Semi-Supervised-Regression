# 成功实验总账（只记录通过预先规定准入标准的正式配置）

> 规则：单 seed 偶然提升、未通过准入、未完成必要重复实验的配置**一律不进入本表**；
> 失败与中间实验只进入 `outputs/auto_search/run_history.jsonl`。
> 最近更新：2026-11-03

**当前状态：M1 尚未完成。** 本表目前只收录“已验证的基线/组件”，M1 复合模型为空。

## 通用协议（所有条目共用）

| 项 | 值 |
|---|---|
| 数据集 | EmoBank（英文），仅 train+dev；**官方 TEST 从未访问** |
| 划分 | 固定文件 `data/emobank_processed/`；train 池 8062 / dev 1000 / 内部 test 1000 |
| lb500_s0 train_lb | sha256 `c7c67afd9c62b71c…` |
| lb500_s1 train_lb | sha256 `6236c82d38617869…` |
| lb500_s2 train_lb | sha256 `f2db1e75ae143363…` |
| dev_gold | sha256 `e8c39e896b067a3a…` |
| 编码器 | 冻结 RoBERTa-base（`pretrained/roberta-base`），mean-pool + L2 normalize |
| 主干 | `/root/autodl-tmp/rankup_project/Semi-Supervised-Regression-avrrc`，分支 `m1/avrrc-oof` |
| Git commit | `fd6712dea7ebec61e54ef34a2215f2995d643790` |
| 评价 | dev 1000，逐维 + 宏平均；MAE 为 1–5 原尺度 |

---

## E0　M0 匹配监督基线　【Admission: GO（基线）】

| 字段 | 内容 |
|---|---|
| Experiment ID | `E0-m0-frozen-lb500-l1-160` |
| Method | 冻结 RoBERTa + 3 维回归头，L1 目标，160 步，λ_u=0（纯监督，与 JACF 其余配置完全一致） |
| Research Point | 无（基线，供创新点一/二做匹配对照） |
| Dataset | EmoBank（train+dev） |
| Label Budget | 500 |
| Seeds | 0, 1, 2 |
| Config | `saved_models/vad_research/alpha_sweep/sup_s{0,1,2}/run_config.json` |
| Git Commit | `fd6712d`（运行时为 `f0d7e5b`，相关代码未变） |
| Checkpoint | `alpha_sweep/sup_s{0,1,2}/model_step96.pth`（501 MB ×3） |
| Metrics（三种子均值 ± SD） | MAE **0.17580** ± 0.00175；r 0.4789 ± 0.0096；γ 0.7147；CCC 0.4505 ± 0.0104；R² 0.1713 ± 0.0185 |
| Reproduction | 见下方命令块（`--lambda-u 0`） |

## E1　JACF 组件一　【Admission: GO（组件）/ M1 未完成】

| 字段 | 内容 |
|---|---|
| Experiment ID | `E1-jacf-a075-lb500` |
| Method | JACF：伪目标 = 0.75·kNN 锚点 + 0.25·EMA teacher（α=0.75, k=20, τ=0.1, MC=5），L1 点损失 |
| Research Point | 创新点一（无标签数据驱动）+ 二（锚点引导伪监督）的已验证组件 |
| Dataset | EmoBank（train+dev） |
| Label Budget | 500 |
| Seeds | 0, 1, 2 |
| Config | `saved_models/vad_research/alpha_sweep/a0.75_s{0,1,2}/run_config.json` |
| Git Commit | `fd6712d`（运行时为 `f0d7e5b`） |
| Checkpoint | `alpha_sweep/a0.75_s{0,1,2}/model_step96.pth` |
| Metrics（三种子均值 ± SD） | MAE **0.17047** ± 0.00117；r 0.4807 ± 0.0068；γ 0.6175；CCC 0.4300 ± 0.0097；R² **0.2174** ± 0.0023 |
| Admission 说明 | 相对 E0：MAE −0.0053、R² +0.0461，但 CCC −0.0205、γ −0.097 → **只可作为组件一，不足以单独构成 M1** |
| Reproduction | 见下方命令块（`--lambda-u 1 --alpha-fixed 0.75`） |

---

## 被拒绝进入总账的配置（仅登记，不建独立文档）

| 配置 | 结果 | 拒绝原因 |
|---|---|---|
| ACLDP λ=0.05（三种子） | MAE 0.17139 / CCC 0.4457 / R² 0.2116 | MAE 差 0.00039、R² 差 0.0058，未同时满足三项准入 |
| ACLDP shuffle（负控制） | CCC 0.3586 | 负控制按预期崩溃，用于论文对照，不作模型 |
| Centered-M0 对照 | CCC 0.4395 | 对照用，证明 ACLDP 非纯中心化蒸馏 |
| Mean Teacher（匹配预算） | MAE 0.17419 / CCC 0.4473 / R² 0.1894 | 半监督基线对照，未达准入 |
| RDA / LVDP / Selective-ACLDP / 双视角 ACLDP | 见 `docs/` 既有报告 | 机制审计不通过或收益不足 |

---

## 复现命令

```bash
R=/root/autodl-tmp/rankup_project/Semi-Supervised-Regression-avrrc
PY=/root/autodl-tmp/rankup_project/envs/rankup/bin/python
cd $R
# E0 (lambda_u=0) / E1 (lambda_u=1)
for S in 0 1 2; do
  $PY scripts/vad_research/train_acvr.py --budget 500 --seed $S --steps 160 \
    --mode fixed --alpha-fixed 0.75 --lambda-u 1.0 \
    --p1-dir /root/autodl-tmp/rankup_project/Semi-Supervised-Regression/saved_models/vad_research/p1_roberta_lb500_s$S \
    --eval-knn-alpha 0.4 --eval-scale 1.2 --device cuda --no-save-ckpt \
    --save-at-step 96 --out saved_models/vad_research/repro_jacf_s$S
done
```
