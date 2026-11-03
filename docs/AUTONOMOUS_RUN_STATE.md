# 自主研究运行状态（断点恢复用）

> 本文件只记录当前阶段、commit、最佳模型、在跑任务、未解决问题与下一步，**不是**实验总账。
> 最近更新：2026-11-03 18:40 (Asia/Shanghai)

## 1. 当前研究阶段

**M1 / 组件二 —— Phase 1（诊断）已完成主要量化，Phase 2（检索）与 Phase 5（小规模准入）进行中。**
M0 监督基线保持冻结；M1 尚未完成，未进入 M2，未进入旅游域适配合格评价。

## 2. 环境与代码

| 项 | 值 |
|---|---|
| 远程主机 | AutoDL `connect.westc.seetacloud.com:24569`（容器 `autodl-container-10be4ca793-fcf61227`） |
| GPU | NVIDIA GeForce RTX 4090, 24564 MiB（实验时独占，空闲 1 MiB） |
| Python | `/root/autodl-tmp/rankup_project/envs/rankup/bin/python` → torch 2.4.1+cu121, CUDA True, transformers 4.46.3 |
| 主线 worktree | `/root/autodl-tmp/rankup_project/Semi-Supervised-Regression-avrrc` |
| 主线分支 / HEAD | `m1/avrrc-oof` / `fd6712dea7ebec61e54ef34a2215f2995d643790`（2026-11-02 21:23） |
| 实验 worktree | `/root/autodl-tmp/rankup_project/Semi-Supervised-Regression-comp2`，分支 `m1/comp2`，起点 `fd6712d` |
| 磁盘 | `/` 27 G 可用；`/root/autodl-tmp` 96 G 中 32 G 可用（清理事故后状态，见 `cleanup_20261103/`） |
| 单次 160 步实验耗时 | **≈95 s**（实测：JACF seed0 复现运行 18:32:20→18:33:58） |

## 3. 当前最佳有效模型

| 角色 | 路径 | 说明 |
|---|---|---|
| M0 匹配监督参照 | `…/alpha_sweep/sup_s{0,1,2}/model_step96.pth` | `lambda_u=0`，其余与 JACF 完全一致（已核对 run_config） |
| JACF（组件一，固定） | `…/alpha_sweep/a0.75_s{0,1,2}/model_step96.pth` | α=0.75，step 96，三种子 |
| 组件二最强候选（未准入） | `…/diff_s005_s0`、`…/diff_s5_s1`、`…/diff_s5_s2` | ACLDP λ=0.05；仅 s0 保留 checkpoint |

三种子 RAW 指标（dev=1000，MAE 在 1–5 原尺度）见 `docs/PHASE1_DIAGNOSIS_20261103.md` 表 1。

## 4. 正在运行的任务

- `…/Semi-Supervised-Regression-comp2/run_loss_ablation.sh`（pid 7800，18:38 启动）
  目的：判定“L1-only 目标”是否使神经基线劣于闭式 ridge（Phase 1 基座诊断）。
  6 个 run：`{l1,mse,huber} × {sup(lambda_u=0), jacf}`，seed0，输出 `saved_models/vad_research/abl_*_s0`。
  日志：`outputs/auto_search/logs/loss_ablation.log`。

## 5. 已完成阶段

- Phase 0：仓库/分支/HEAD、数据划分哈希、权重、环境、GPU、磁盘全部核对完毕（本文件第 2 节）。
- Phase 1（部分）：完成 4 项定量诊断（准入可行性、标签噪声上限、标签预算标定曲线、基座与 ridge 对比）。
- 代码基座：`m1/comp2` 分支新增可切换点损失 `--sup-loss/--unsup-loss/--huber-delta`，6 个单测通过，默认值与原目标逐位一致。

## 6. 尚未解决的问题

1. **约束瓶颈已定位为宏平均 Pearson r**：在 JACF 预测上的任意逐维仿射变换都不存在满足三项准入的点（表 2），需要 r 提升而非尺度调整。
2. 神经基座在 500 标签下弱于同特征闭式 ridge（表 3），需先校正基座，否则任何组件二收益都不可信。
3. 组件二仍无通过准入的机制；ACLDP 差 MAE 0.0004、R² 0.0058。
4. 文献检索尚未产出经核验的机制清单（子 Agent 检索接口不稳定，待主 Agent 亲自核验）。

## 7. 下一项待执行任务

1. 读取损失对照结果，若 mse/huber 明显优于 l1，则把基座校正写入状态并在 `m1/comp2` 上重跑 JACF（matched）与 E5。
2. 在**校正后的基座**上重跑 ACLDP，得到可比的组件二基线。
3. 完成 2024–2026 文献核验，产出 ≤6 个机制方向，选定 1 个实现。
4. seed0 小规模准入 + 负控制 → 三种子 → 五折 OOF。

## 8. 复现命令（当前基线）

```bash
R=/root/autodl-tmp/rankup_project/Semi-Supervised-Regression-avrrc
PY=/root/autodl-tmp/rankup_project/envs/rankup/bin/python
cd $R
$PY scripts/vad_research/train_acvr.py --budget 500 --seed 0 --steps 160 \
  --mode fixed --alpha-fixed 0.75 --lambda-u 1.0 \
  --p1-dir /root/autodl-tmp/rankup_project/Semi-Supervised-Regression/saved_models/vad_research/p1_roberta_lb500_s0 \
  --eval-knn-alpha 0.4 --eval-scale 1.2 --device cuda --no-save-ckpt \
  --out saved_models/vad_research/<name>
```
