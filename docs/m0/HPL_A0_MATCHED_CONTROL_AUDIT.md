# EmoBank VAD M0：HPL A0 准入与匹配 Control 只读审计

**结论：`A0_NOT_SSL`；就 A1 的纯监督对照而言，`MATCHED_CONTROL_EXISTS`。** A0 本身是 160-step 纯监督续训，不能作为 HPL 的半监督结果。A1 的无标签项确实反传，但其 DEV 最优检查点在 step 0，权重与 P1 逐张量完全相同；当前最佳模型没有接受任何 A1 更新。P2 和旧 `hetero_v2` 与 A0/A1 属于 `PROTOCOL_MISMATCH`，不能用其分数推断 HPL 的因果收益。未找到明确命名为 E0 的检查点或日志（`E0_NAMED_ASSET_MISSING`）；旧监督资产只能单列为候选，不能擅自认作 E0。

## 1. 范围、路径与工作树

- 优先路径 `/hy-tmp/rankup_project/Semi-Supervised-Regression` 存在，解析到 `/root/autodl-tmp/rankup_project/Semi-Supervised-Regression`；审计时分支 `vad-m1-v3-pseudolabel-calibration`，HEAD `e45575d642b301e0410642054cef6ee0999bdc7d`。没有切换分支或覆盖实验。
- 审计开始时已有 7 个未提交修改：`scripts/vad_research/m0/{hpl_core.py,run_spectral_ridge.py,spectral_core.py,train_hpl_vad.py}` 与 `tests_m0/{test_hpl_core.py,test_spectral_core.py,test_spectral_runner.py}`。本轮不改这些文件。A0/A1 原始 `run_config.json` 记录的代码提交为 `998c900329723660411213e86ea846a4be12a203`；当前未提交的 HPL 改动增加诊断和 `model_last.pth` 保存，不能以当前工作树状态代替原运行版本。后续 `hpl_a1_diagnostic_s0` 与 `hpl_a1_lambda2_0_s0` 的配置仍只写该 Git 提交号，无法由它单独证明工作树完全干净。
- 读取的资产仅为训练脚本、数据准备说明、TRAIN/DEV 配置与现存结果及 checkpoint 元数据；**没有打开 TEST 数据或运行评估**。数据读取路径在 `load_checked_splits` 中仅为 `train_lb.parquet`、`train_ulb.parquet`、`dev_gold.parquet`。
- `scripts/emobank/prepare_emobank.py` 使用官方 `corpus/emobank.csv`；官方 `corpus/README.md` 给出 Reader/Writer 合并式 `(N_R·Reader + N_W·Writer)/(N_R+N_W)`，按各视角有效标注人数加权，再四舍五入到两位。`read_gold` 核验原始 VAD 位于 `[1,5]` 且 `y_norm=(y−1)/4`；DEV 输出和 MAE 还原到原量纲。固定集合为 500 有标签 TRAIN、7,562 无标签 TRAIN、1,000 DEV。

## 2. A0 最终生效协议

| 项目 | 实际值与证据 |
|---|---|
| 运行目录 | `saved_models/vad_research/m0/hpl_a0_lb500_s0`，`mode=a0`，seed 0，CUDA |
| 初始化 | `saved_models/vad_research/p1_roberta_lb500_s0/model_best.pth`；实际 SHA256 与 A0 配置一致：`ded8403b94c616e4d187e11cbb34b25026d589b9a296468fdbb04645c99d0cb1`；P1 在自身 DEV 第 10 epoch 选出 |
| 网络/冻结 | RoBERTa-base → attention-mask mean pooling → Dropout(0.1) → Linear(768,768) → GELU → Linear(768,3)；**无显式冻结参数**。encoder/head 均进优化器；RoBERTa 未被头使用的 pooler 即使列在 encoder 参数中也无该损失梯度。A0 不创建 `VADUncertainty` |
| 优化 | encoder 与 head 各一个 PyTorch `Adam`，均 `lr=5e-6`、`weight_decay=1e-5`，无学习率调度；梯度范数裁剪 1.0。不同于 P2 的单个 AdamW |
| 数据流 | 标注 batch 16，独立 DataLoader 随机发生器 seed `17000+0`；最大长度 128。A0 不实例化无标签 DataLoader，配置中的 `ulb_batch_size=64`、`lambda_u=1`、`unc_lr=1e-4`、`meta_every=5`、`lambda2=1` 是共享 CLI 的**闲置参数** |
| 更新/评价 | 恰好 160 次优化器更新；step `0,32,64,96,128,160` 评价，DEV batch 32；选择原量纲 DEV Macro-MAE 最低者，包含 step 0 |
| 损失/RNG | `F.mse_loss(model(tokens), y_norm)`；每步监督前 `torch.manual_seed(seed·100000+step)`，监督路径保留训练态 Dropout；`total=supervised`，随后 `total.backward()` 与两个 Adam step |
| 伪标签与权重 | **无在线或离线伪标签、无无标签损失、无不确定度权重、无 meta 更新**。`training_history.csv` 所有评价点的无标签损失、无标签头梯度、访问数均为 0 |

因此 A0 的“读取 7,562 条无标签元数据并校验拆分”不等于半监督训练。A0 代码分支 `unlabeled = ... if mode=='a1' else None`，训练体中全部无标签逻辑只在 `if mode=='a1'` 下；A0 的主反传只有监督 MSE。

## 3. A1 具体新增机制与真实作用

A1 相对 A0 的两份原始 `run_config.json` 逐字段比较，**只有 `mode` 和 `out` 不同**。同 P1 SHA、标签流 seed、batch、160 步、监督 MSE、Dropout、两个 Adam 优化器、裁剪和 DEV 选模均相同。A1 额外创建共享的 `VADUncertainty`（每个 V/A/D 坐标输入 `[strong−weak,strong]`，2→128→1 MLP），其单独 Adam 为 `lr=1e-4, weight_decay=1e-5`；无标签 batch 64，独立 shuffle seed `19000+0`。

同一句无标签文本在 `model.eval()` 下得到在线 `weak=stopgrad(f(u))`，在 `model.train()` 下得到含 Dropout 的 `strong=f(u)`。不读取离线伪标签。每维 `s=uncertainty(strong−weak,strong)`，主损失为

`L_U=mean[0.5·exp(−stopgrad(s))·(strong−stopgrad(weak))²]`，`L=L_S+1.0·L_U`。

每 5 步，另一个仅来自 **TRAIN 500 标签**、独立 shuffle seed `23000+0` 的 meta batch，经 `higher` 的 head-only 虚拟内步与 `meta MSE−1.0·mean(meta logvar)` 更新不确定度网络；160 个主步对应 32 次不确定度优化器更新，不拿 DEV 标签做 meta 训练。虚拟内步的 encoder 特征停止梯度，**实际主无标签项中的 strong 分支仍反传至 encoder 和回归头**。A1 最后一行累计值证实：160 步共见过 7,562 个无标签 ID，平均无标签加权 MSE `0.000390874`，平均无标签头梯度范数 `0.0375942`，均非零。代码的 `total.backward()` 位于累加 `L_U` 之后。

但 A1 原始运行的最优 checkpoint 元数据为 **step 0**、DEV Macro-MAE **0.178264362**；其 `model` state_dict 与 P1 checkpoint **逐张量 bitwise 相等**。后续 32/64/96/128/160 步均没有更低 DEV MAE。诊断复跑与 `lambda2=0` 变体也在 step 0 选最优；它们的 `model_last.pth` 只证明完成了 160 步，不能当作 DEV 最优或 HPL 收益。

## 4. 匹配 Control 与不可直接比较项

| 对照要求 | A0 ↔ 原始 A1 | 结论 |
|---|---|---|
| 相同初始化 | P1 路径与实际 SHA256 一致；step 0 DEV MAE 一致 | 满足 |
| 相同可训练预测参数 | 同一 RoBERTa encoder 与非线性 head；均无冻结；A1 的额外 uncertainty 网络是处理机制专属 | 满足 |
| 相同标注 batch 流 | 同一 500 TRAIN、batch 16、shuffle seed 17000；A1 的 meta 与无标签 loader 使用独立发生器 | 满足 |
| 相同优化步数与监督损失 | 160 步、normalized MSE，监督前同一逐步手动 seed，训练态 Dropout(0.1) | 满足 |
| 相同预测模型优化器与评估 | encoder/head 分离 Adam、相同 LR/WD/裁剪，DEV 0+每 32 步、同一原尺度 Macro-MAE 选择 | 满足 |
| 处理机制差异 | A1 增加在线伪值、加权无标签 MSE、uncertainty/meta 内步及其额外优化器 | 预期差异 |

所以 **A0 正是 A1 所需的已完成 160-step 匹配纯监督 Control**；`smoke_a0_s0` 仅 1 步，`smoke_a0_s0_bs16` 仅 5 步，不是替代。没有缺失 Control 的拟执行命令，也不应重复训练 A0。

P2 seed0 虽同 P1 初始化、500/7,562/1000、batch 16、160 步、每 32 步 DEV，但使用 **normalized L1、单个 AdamW、weight decay 0.01**；监督 Dropout 没有 A0 的逐步 `torch.manual_seed`，因此即使标签 DataLoader seed 17000 相同也不能声称完全相同的随机流。P2 最优 step 128、MAE 0.175543304；A0 最优 step 96、MAE 0.175329182，两者微小数值差**不能解释为 HPL 效果**。P2 seeds 1/2 使用不同标签拆分与 P1 初始化，也不构成 seed0 A1 的匹配 Control。

明确命名的 E0 资产在当前 `saved_models`、`logs` 与脚本/记录搜索中未找到。旧 `saved_models/emobank/supervised_lb500_s0` 及同名日志存在，使用 `google/bert_uncased_L-4_H-512_A-8`、20 epoch 左右的监督训练，最优 epoch 14、MAE 0.205647906；**不能仅因它是旧监督模型就认定为 E0**。`hetero_v2_control_lb500_s0` 从该旧 BERT checkpoint 续训，normalized L1、8×64=512 步、weight decay 5e-4、最优 epoch 3、MAE 0.204819925；与本轮 RoBERTa/P1/MSE/160-step HPL 协议不匹配。它自己的 H0/H1 比较只适用于旧协议。

## 5. 完整 DEV 轨迹、检查点和实际文件

| 运行 | step 0 / 32 / 64 / 96 / 128 / 160 的 DEV Macro-MAE `[1,5]`（六位小数） | 最佳 checkpoint |
|---|---|---|
| A0 | .178264 / .190115 / .179241 / **.175329** / .179632 / .178818 | `model_best.pth`，元数据 step 96 |
| A1 | **.178264** / .195243 / .183194 / .179156 / .184864 / .188397 | `model_best.pth`，step 0，权重等于 P1 |
| A1 diagnostic | **.178264** / .195243 / .183194 / .179156 / .184864 / .188397 | `model_best.pth` step 0；另有 step 160 `model_last.pth` |
| A1 λ₂=0 | **.178264** / .195275 / .183054 / .178600 / .184314 / .187042 | `model_best.pth` step 0；另有 step 160 `model_last.pth` |
| P2 seed0 | .178264 / .182273 / .183077 / .184278 / **.175543** / .177198 | `model_best.pth` step 128；仅描述性参照 |

实际结果文件（目录均在 `saved_models/vad_research/` 下）：

- `m0/hpl_a0_lb500_s0/`、`m0/hpl_a1_lb500_s0/`：各有 `run_config.json`, `training_history.csv`, `dev_metrics_best.json`, `dev_predictions_best.csv`, `model_best.pth`；**没有** `model_last.pth` 或 `training_diagnostics.csv`，符合历史代码版本。
- `m0/hpl_a1_diagnostic_s0/`、`m0/hpl_a1_lambda2_0_s0/`：除上述五文件外，还有 `model_last.pth`, `training_diagnostics.csv`。两者不是新的纯监督 Control。
- `p2_control_lb500_s0/`：上述五文件；另有 `logs/p2_control_lb500_s0.log`。`p2_control_lb500_s1/s2/` 也各有同类五文件及相应日志，但标签 seed 不同。
- 旧 `saved_models/emobank/supervised_lb500_s0/` 有 `constant_baseline_dev.json`, `dev_metrics_best.json`, `dev_predictions_best.csv`, `model_best.pth`, `run_config.json`, `training_history.csv`，日志 `logs/emobank/supervised_lb500_s0.log`。`saved_models/emobank/hetero_v2_control_lb500_s0/` 有 `dev_metrics_best.json`, `dev_predictions_best.csv`, `model_best.pth`, `run_config.json`, `summary.json`, `training_history.csv`，日志 `logs/emobank/hetero_v2_control_lb500_s0.log`。
- 未发现 HPL A0/A1 原始运行的单独控制台 `.log`；其已存 `run_config.json` 和 `training_history.csv` 是可核的配置与完整评价轨迹。

逐实验机器可读清单见 `HPL_EXPERIMENT_MATRIX.csv`。本轮只新增这两份审计文档；**没有新模型、训练、搜索、代码修改或 TEST 访问**。
