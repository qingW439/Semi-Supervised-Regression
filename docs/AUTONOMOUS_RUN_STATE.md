# 自主研究运行状态（断点恢复用）

> 本文件只记录当前阶段、commit、最佳模型、在跑任务、未解决问题与下一步，**不是**实验总账。
> 最近更新：2026-11-03 18:56 (Asia/Shanghai)

## 1. 当前研究阶段

**M1 / 组件二 —— Phase 1（诊断）基本完成，正在重建匹配强监督基线。**
M1 未完成；未进入 M2；旅游领域适配尚未开始。

## 2. 环境与代码

| 项 | 值 |
|---|---|
| 远程主机 | AutoDL `connect.westc.seetacloud.com:24569`（`autodl-container-10be4ca793-fcf61227`） |
| GPU | RTX 4090 24564 MiB（检查时独占空闲 1 MiB） |
| Python | `/root/autodl-tmp/rankup_project/envs/rankup/bin/python` → torch 2.4.1+cu121, CUDA True, tf 4.46.3 |
| 主线 worktree | `/root/autodl-tmp/rankup_project/Semi-Supervised-Regression-avrrc` |
| 主线分支 / HEAD | `m1/avrrc-oof` / **`9a23c3dceb0a3027e50350e2766cdeb5dc50ef24`**（文档提交） |
| 实验 worktree | `/root/autodl-tmp/rankup_project/Semi-Supervised-Regression-comp2`，分支 `m1/comp2`（起点 `fd6712d`，未提交改动：点损失开关 + 3 个诊断脚本 + 2 个批次脚本） |
| 磁盘 | `/` 27 G 可用；`/root/autodl-tmp` 可用 32 G |
| 单次 160 步实验 | ≈95 s |

## 3. 当前最佳有效模型（按指标）

| 角色 | 模型 | 三种子均值 |
|---|---|---|
| **强监督基线 E0′（新）** | 冻结特征 + 闭式 ridge（α 内部 CV） | MAE **0.16483** / r **0.5048** / CCC **0.4484** / R² **0.2670** |
| 原 E0 监督基线 | 冻结 RoBERTa + MLP 头，L1，160 步 | MAE 0.17580 / r 0.4789 / CCC 0.4505 / R² 0.1713 |
| 组件一 JACF | α=0.75 | MAE 0.17047 / r 0.4807 / CCC 0.4300 / R² 0.2174 |
| 组件二最强候选（未准入） | ACLDP λ=0.05 | MAE 0.17139 / r 0.4867 / CCC 0.4457 / R² 0.2116 |

## 4. 正在运行的任务

- `comp2/run_recipe_test.sh`（pid 11554，18:52 启动）：4 个纯监督配置
  `{640,1600 steps}×{lr 5e-6,1e-4}×{bs 16,64}`，seed0，检验 MLP 头能否在充分训练后追上闭式 ridge。
  日志 `comp2/outputs/auto_search/logs/recipe_test.log`。

## 5. 已完成阶段与关键结论（详见 `docs/PHASE1_DIAGNOSIS_20261103.md` 与附录 A）

1. 原准入三项在 JACF 预测上**任意逐维仿射均不可行**（68 921 组网格，0 组可行）；瓶颈是宏平均 Pearson（需 ≈0.49）。
2. 标签噪声上限很高（宏 R²_max 0.661、r_max 0.809），模型离上限很远，不存在“数据本身做不到”。
3. 瓶颈是**标签数量**而非冻结特征：闭式 ridge 的 r 从 500 标签的 0.4835 单调升到 8062 标签的 0.5828。
4. **纯监督闭式 ridge（同特征、同 500 标签）三种子均值已全部超过原准入线** → 原准入线是相对“欠训练基座”标定的，不能再作为 M1 门槛。
5. 闭式自训练对线性模型是**可证明的恒等变换**（实测逐位相同）；伪标签在神经模型上的作用来自平滑正则，这正是压缩 γ 与 CCC 的机制根源。
6. 核/谱/PCA/标注者方差加权均不优于线性 ridge（内部选择复核；此前 k=25 的收益是 dev 调参噪声）。
7. 唯一正收益：两个误差部分独立的估计器取平均（ridge×0.4 + JACF×0.6）使 seed0 r 达 0.4941、CCC 0.4439、MAS 0.16788、R² 0.2419 —— 仍差 CCC 0.0011，且权重是在 dev 上扫描的，只能算诊断上界。

## 6. 尚未解决的问题

1. **准入门槛需要重新标定**：M1 应判定为“相对强监督基线 E0′ 的增量”，否则纯基线即可通过，会犯“把基线提升当半监督创新”的错误。该修订需在总账中显式记录（不修改指标、不修改划分）。
2. 神经 MLP 头（≈59 万参数）在 500 标签下过参数化；若能追平 ridge，则可作为 M1 的神经基座，否则 M1 需以 ridge 为基座。
3. 组件二仍无通过准入的机制；`ridge+JACF` 组合是唯一接近的候选，需要“非 dev 选择”的权重与三种子复核。
4. 论文检索（Phase 2）只完成了 arXiv 侧的初步核验（含 ICML 2026 Noisy Proxy Covariates、NeurIPS 2025 Heteroscedastic Pseudo-Labels、DKD 等），尚未形成最终机制清单。

## 7. 下一项待执行任务

1. 读取 `recipe_test` 结果 → 确定 E0′ 的形式（神经追平则用神经，否则用 ridge 及其闭式初始化）。
2. 在 E0′ 之上重跑 JACF（matched），确认 JACF 在强基座上是否仍有增量。
3. 用**内部确定**的混合权重（OOF 拟合）复核 `ridge+JACF`，跑三种子；若通过则作为 M1 候选。
4. 若 JACF 在强基座上无增量，则重新设计组件二（新的无标签信息利用方式），并记录失败原因。

## 8. 复现命令

```bash
R=/root/autodl-tmp/rankup_project/Semi-Supervised-Regression-comp2
PY=/root/autodl-tmp/rankup_project/envs/rankup/bin/python
cd $R
# 强监督基线 E0′（闭式 ridge，三种子）
$PY scripts/vad_research/probe_label_scaling.py     # 标签预算曲线（含 500 标签点）
# JACF / 组件二
$PY scripts/vad_research/train_acvr.py --budget 500 --seed 0 --steps 160 \
  --mode fixed --alpha-fixed 0.75 --lambda-u 1.0 --sup-loss l1 \
  --p1-dir /root/autodl-tmp/rankup_project/Semi-Supervised-Regression/saved_models/vad_research/p1_roberta_lb500_s0 \
  --eval-knn-alpha 0.4 --eval-scale 1.2 --device cuda --no-save-ckpt \
  --out saved_models/vad_research/<name>
```
