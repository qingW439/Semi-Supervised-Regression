# M1 目标效用预测有效性：来源门槛审计

**结论：INCONCLUSIVE（未进入模型更新模拟）。** 审计日期：2026-10-29。按任务第 3 条，必须先确认 P3B 的 100 条留出 TRAIN 标签未用于参考 checkpoint 的训练或模型选择；本次能复核 400/100 划分、归档脚本意图和运行日志，但无法证明运行时实际执行的脚本与两天后入库的版本一致。因此在来源门槛停止，不能据此 checkpoint 检验 head-only 效用对完整 AdamW encoder+head 更新的预测力。此结论不等于机制被证伪。

## 检查范围与 Git 状态

复读了现有 `docs/m1/M1_UTILITY_FAILURE_AUDIT.md`、`M1_METHOD_SPEC.md` 与 `scripts/vad_research/m1/dimensional_utility.py`。在既有独立工作树 `/root/autodl-tmp/rankup_project/Semi-Supervised-Regression-m1-utility-audit` 操作，HEAD `e45575d642b301e0410642054cef6ee0999bdc7d`，起始仅有前次两份未跟踪的 `docs/m1`；本次未改模型或训练代码。主仓库仍有前次审计前已有的 7 个 M0 修改文件和 3 个 `docs/m0` 未跟踪文件，未覆盖。当前 `dimensional_utility.py` SHA256 为 `374747c5b0d2242a2ba3c4a16904de6555edd05c0f8402f62be7b6fe24258e66`。

此前 14:20 UTC 的 M1 日志虽含后来才提交的余弦字段，配置却标 `1b70ee2`；沿用上次审计结论，未把它当可复现的正式结果。旧 160 步结果标 `1b70ee2`，其代码可由 Git 查看，本次仅离线重放记录，不做训练。

## CPU 工程消融：旧回退与零门控

从 `hpl_diagonal_lb500_s0_20261027T133009025467Z` 与 `hpl_crossdim_lb500_s0_20261027T133109882515Z` 各自的完整 `utility_matrix_history.csv` 读取 32 个矩阵；按历史控制器的轴定义复算 `S_t=0.9S_{t-1}+0.1M_t`（首项直接赋值）。diagonal 取对角，crossdim 取 meta 行对各无标签列的和；均先截断负分数。

两模式 32/32 次均得到全非正分数：旧策略探测后恒为 `[1,1,1]`；候选“全非正→零门控”探测后恒为 `[0,0,0]`。首个探测在 step 5，故若仅按既有记录静态重放，step 1–4 保持全 1，step 5–160 共 156 个位置会改变权重。旧日志这些位置的无标签损失记录之和为 `0.06095944400294684`（两模式相同）。这只是**固定旧轨迹上的工程映射消融**；一旦权重真的改变，后续模型、效用矩阵与损失均会改变，不能把该数当反事实训练收益。未声称创新。

## P3B 400/100 资产及来源门槛

P3B 目录：主仓库 `saved_models/vad_research/p3b_probe400_100_lb500_s0/`。固定源数据 `train_lb.parquet` 的 SHA256 为 `c7c67afd9c62b71cc7dfd5a31e4958072805f157dc7bf92fa06d52f3ba9fb094`。用归档脚本所记的 `seed=0`、`split_seed=20261019`、先按 ID 排序再 NumPy permutation 的规则独立复算，400/100 ID 与 `teacher_train_ids.csv`、`probe_ids.csv` 完全一致，交集 0；排序 ID 的 SHA256 分别为 `2267918146a9632bd90c3d07b5215badb70dde1799a039ae190ca0f4aa000c6e` 与 `54d7fcfc5964d5a4e15bfa31d7c13ff2b0cff7428924ea826ec066ee2cdb18cb`。checkpoint `teacher_probe_only.pth` 的 SHA256 为 `f37a7aa1352e72e423a7f0dc1fae7dfdbfea0a15e7e2e29e49f3361aacb7e9fa`，内含 model、320 步计数、配置及相同 split manifest；没有 optimizer state。

归档的 `scripts/vad_research/probe_roberta_vad_p3b.py`（SHA256 `31bb14697bbef440612dc02393a4c42b81eed1d691c24b3afd0909cce43a60a8`）定义了：只在 400 条上从原始 RoBERTa 初始化，AdamW、固定 320 步、无训练期探针或提前停止；完成所有更新后才读取 100 条探针标签并保存唯一 checkpoint。运行日志 `logs/p3b_probe400_100_lb500_s0.log` 与 manifest 报告固定 320 步和一次事后探针评价，文件时间约为 2026-10-19 19:25（UTC+8）。这些是**支持归档流程意图的证据**。

关键缺口：运行工件记录的 `source_git_commit=23d54714c707ccd408f377e0ae68d54afda4ff9b`，其 Git 树**没有** `probe_roberta_vad_p3b.py`。该脚本直到 2026-10-21 的 `d2b5959` 才入库；没有运行时脚本哈希、不可变源码快照或可独立验证的训练命令/审计记录。日志里的“不使用 probe”是运行程序自身的声明；保存的 ID 和 checkpoint 元数据能证实分组一致，不能独立证明训练或 checkpoint 选择过程未查看这 100 条标签。checkpoint 也不含运行末尾 AdamW 动量，无法严格复现作为原训练续步的优化器状态。若以后将其用于局部比较，最多能从同一 checkpoint 以**四个相同的全新空 AdamW 状态**开始，并明确这是重置状态模拟。

因此任务第 3 条要求的严格确认未达到。按“若无法确认，停止并报告”，没有预先挑选用于效果分析的监督/无标签重复 batch，也没有调用参考模型前反向、没有任何监督或单维无标签的局部参数更新，更没有使用 100 条探针标签计算 MSE 变化。机器结果中的完整更新收益、效用符号一致率、排序一致率及不确定性均为 `null`，不可从旧权重日志推断。

## 待验证问题及文献边界

一旦有运行时不可变源码哈希或同等来源证明，并能核实训练与选择过程，才可事先固定有限 batch、同一 checkpoint、同一重置 AdamW 状态及相同监督随机流，分别比较监督与监督+V/A/D 无标签的**完整 encoder+head** 局部更新。增量收益应定义为 `Δ_{k,j}=[L_j(probe; θ_sup)-L_j(probe; θ_sup+u_k)]`，其中四支从相同参数和优化器状态出发；正值表示第 k 个无标签项相对纯监督更新改善第 j 维。head-only 余弦只能作为待检预测变量，不能冒充完整优化器收益。本次没有上述数据，故不能判断机制能否用于实际更新决策。

[HPL（NeurIPS 2025）](https://papers.neurips.cc/paper_files/paper/2025/file/30a129d6b8e4229d1ed6d90a65ee1a76-Paper-Conference.pdf) 已用有标注数据双层学习异方差伪标签权重；[Meta-Gradient SSL（AISTATS 2021）](https://proceedings.mlr.press/v130/xiao21a.html) 已用有标注反馈优化伪标签；[PCGrad（NeurIPS 2020）](https://papers.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html) 已处理多任务梯度冲突。现有 M1 的 head-only 效用与前两者共享有标注反馈思想，与 PCGrad 共享梯度方向诊断；零门控只是当前控制器的工程分支。此次来源门槛未过，既不能证明与上述工作的实质方法优势，也不声明新颖性。

**停止状态：INCONCLUSIVE。** 无 DEV/TEST 访问，无 160 步或 M1/M2 训练，无模型改动。
