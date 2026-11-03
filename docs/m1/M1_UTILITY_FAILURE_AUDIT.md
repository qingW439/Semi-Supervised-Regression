# M1 维度目标效用失效审计

审计日期：2026-10-29。范围为 EmoBank 500 条有标注 TRAIN、7562 条无标签 TRAIN、1000 条 DEV 的 V/A/D 连续回归。未打开 TEST，未运行完整训练，未重跑 A0/A1 等基线，未改训练代码。DEV 仅核对协议和既有轨迹，不用于机制选择。

## 结论

1. “权重始终为 1”只适用于早期结果。12:29/13:30 UTC 的 diagonal、crossdim 5 步及 160 步运行中权重均为 `[1,1,1]`；14:20 UTC 的两次 5 步诊断已产生 `[3,0,0]`。模块可以区分维度。
2. 早期 160 步轨迹相同的直接原因是：原始矩阵有维度差异与瞬时正值，但 0.9 动量平滑后的三个效用分数在 32 次探测中全负；截断为非负后和为零，控制器回退 `[1,1,1]`，主模型伪标签项与原 HPL 一样。配对运行的 160 行 `training_diagnostics.csv` 的 54 个共同字段逐项相同；`training_history.csv` 仅计算耗时字段不同。
3. 未发现效用差分的符号、损失约简或梯度断路错误。float64 原始 M 很小，主要由 `5e-6` 虚拟步长导致；小量级本身不导致全 1。主要失效点是把“没有正的 TRAIN 效用”映射成继续使用全部伪标签。负分数是否代表真实泛化伤害尚未证明。
4. 当前代码改用梯度余弦、反转矩阵轴并隔离 TRAIN meta batch。14:20 的结果含这些新字段，但配置仍写 `git_commit=1b70ee2`，而新字段在后续 `fef5516` 才提交；该结果无法仅凭所记 commit 复原，应标记来源不完整，不能混作早期 160 步实验。

## 仓库、数据与配置

主仓库 `/root/autodl-tmp/rankup_project/Semi-Supervised-Regression` 当前为 `e45575d`、分支 `vad-m1-v3-pseudolabel-calibration`。审计前已有 7 个已修改的 M0 代码/测试文件及 3 个未跟踪 `docs/m0` 文件，本次未触碰。隔离工作树 `/root/autodl-tmp/rankup_project/Semi-Supervised-Regression-m1-utility-audit` 从 `e45575d` detached HEAD 建立；建立时状态干净。仅在其 `docs/m1/` 新建本次文档，既有模型与历史结果未改写。

固定文件：`train_lb.parquet` 500 行，SHA256 `c7c67afd9c62b71cc7dfd5a31e4958072805f157dc7bf92fa06d52f3ba9fb094`；`train_ulb.parquet` 7562 行，SHA256 `2d85de9048c5f2de6f4b15220d81e2d77169fd265e4aa64eb336c2ddf2740bf0`；`dev_gold.parquet` 1000 行，SHA256 `e8c39e896b067a3afcd4ef25686eb8f7ad10bccf8c2afe98847111aa40560d9c`。三组 ID 交集均为 0；无标签文件仅有 `id,text`。标签归一化为 `(原始 V/A/D - 1)/4`。哈希与所有相关运行配置一致。

六次目标运行共同配置：`budget=500, seed=0, batch_size=16, ulb_batch_size=64, eval_batch_size=32, max_length=128, dropout=0.1, lr=5e-6, unc_lr=1e-4, weight_decay=1e-5, lambda_u=1, lambda2=1, clip_grad_norm=1, meta_every=5, eval_every=32, device=cuda`；P1 最佳权重 SHA256 `ded8403b94c616e4d187e11cbb34b25026d589b9a296468fdbb04645c99d0cb1`。监督项是归一化 MSE；弱/强视图为同文本 eval/train dropout；既有选择标准为含 step 0 的最低 DEV Macro-MAE。配置的 `ggr_mode=vlr, ggr_scope=head, ggr_surgery_alpha=1` 为共同元数据，diagonal/crossdim 模式**没有启用 GGR**。HPL 参考 blob `978f26892608e27f62451634dbc0501660c005c6`，GGR 参考 blob `1db09f7e76649816526e246d82b54c5f6ef811c8`。完整 JSON 位于各结果目录。

各次 `check_only=false`；`data_root=/hy-tmp/rankup_project/data/emobank_processed`，`pretrain_path=/hy-tmp/rankup_project/pretrained/roberta-base`，`p1_dir=/hy-tmp/rankup_project/Semi-Supervised-Regression/saved_models/vad_research/p1_roberta_lb500_s0`；`out` 是下表对应的绝对结果目录。控制器 `utility_momentum=0.9`，虚拟更新记为“stateless head SGD directional probe at head lr”。早期配置用键 `max_length`，后两次同样为 128；14:20 的新日志还记录 `utility_signal=cosine of unlabeled and TRAIN meta head gradients` 及 `matrix_orientation=rows unlabeled VAD; columns meta VAD`。

共同目录前缀为主仓库的 `saved_models/vad_research/m1/hpl_{mode}_lb500_s0_`；每个目录均核查 `run_config.json`、`utility_matrix_history.csv`、`training_diagnostics.csv`、`training_history.csv`。

| 模式、目录时间戳（UTC） | 配置 commit / steps | 效用行数 | 探测后的权重 |
|---|---:|---:|---|
| diagonal `20261027T122920179185Z` | `998c900` / 5 | 1 | `[1,1,1]`，该 commit 尚不含 M1 文件 |
| crossdim `20261027T123007718792Z` | `998c900` / 5 | 1 | `[1,1,1]`，同上 |
| diagonal `20261027T133009025467Z` | `1b70ee2` / 160 | 32 | 32/32 次 `[1,1,1]` |
| crossdim `20261027T133109882515Z` | `1b70ee2` / 160 | 32 | 32/32 次 `[1,1,1]` |
| diagonal `20261027T142010832641Z` | `1b70ee2` / 5 | 1 | step 5 `[3,0,0]`，运行时未提交改动 |
| crossdim `20261027T142024762235Z` | `1b70ee2` / 5 | 1 | step 5 `[3,0,0]`，同上 |

## 数学与梯度路径

无标签第 `k` 维项为 `U_k = mean_i[0.5 exp(-s_ik)(f_k(x_i,strong)-stopgrad(f_k(x_i,weak)))²]`。其中 `s` 是 HPL 不确定网络的 log variance，在主模型效用路径中 detach。三个维度再取均值；全 1 权重与 M0 的 `weighted_pseudo_mse` 严格同值。真实主训练中，监督项与三个无标签项向 head 和 encoder 反传；弱视图、log variance、控制器权重不沿此路径反传。

旧版 `1b70ee2`：`g_k=∇_head U_k`，仅替换 head 参数为 `h-ηg_k`，`η=5e-6`，以冻结的 TRAIN meta 特征和各维 MSE 定义 `M_{j,k}=L_meta,j(h)-L_meta,j(h-ηg_k)`。旧矩阵**行是 meta 维、列是无标签维**；diagonal 取 `M_{k,k}`，crossdim 对列求和。一阶近似为 `M_{j,k}=η〈∇_h L_meta,j,g_k〉+O(η²)`，正号表示虚拟步降低 meta MSE。探测不改真实参数或优化器状态；它是 head-only SGD 方向代理，主训练实际使用 Adam 且 encoder 也更新。旧版复用 HPL meta batch；当前 `disjoint_meta_batch` 已隔离当前监督 batch 并限定为 TRAIN。

当前 `e45575d`：`virtual_utility_probe` 保存原始差分、一阶值和 `N_{k,j}=cos(g_k,∇_h L_meta,j)`；**行是无标签维、列是 meta 维**，零范数时余弦置零。diagonal 分数 `N_{k,k}`，crossdim 分数 `mean_j N_{k,j}`。控制器做 `S_t=0.9S_{t-1}+0.1N_t`（首探测直接赋值），再令 `q_k=max(score_k,0)`；若 `sum q>0`，`w_k=3q_k/sum q`，否则回退 `w=[1,1,1]`。所以三维权重和为 3；负分数被截断。当前无额外的 M/N 阈值；`clip_grad_norm=1` 只在主训练 `total.backward()` 后裁剪模型梯度，不裁剪虚拟探测梯度。

160 步两组第一次探测的对角 M 为 `[-3.7656e-8,-8.4677e-8,-7.2522e-8]`；crossdim 列和为 `[-4.0004e-8,-8.1044e-8,-7.2514e-8]`。32 行中，对角正值分别出现 15、7、9 次，crossdim 列和正值为 15、8、10 次；但复算 EMA 后 diagonal 的分数最大值仍是 `[-3.72e-9,-2.42e-8,-1.98e-8]`，crossdim 最大值 `[-4.27e-9,-2.29e-8,-1.96e-8]`，全部小于零，故全程回退。早期 M 绝对值约 `1e-12–1.8e-7`。160 步主训练无标签 head 梯度范数 `0.0117–0.0675`，无标签损失 `2.62e-4–6.74e-4`，没有整体梯度消失；监督/无标签 head 梯度余弦中位数约 `-0.409`，提示冲突但不等于泛化伤害。

已有只读 GPU 报告 `/hy-tmp/rankup_project/m1_probe_report.json` 使用固定 P1 checkpoint、16 条有标注 TRAIN meta 和 64 条无标签 TRAIN：`η=5e-6` 时 head 相对虚拟变动 `2.8e-8–4.5e-8`；float64 M 与一阶项最大绝对差 `2.15e-11`，重要元素最大相对差约 `4.16e-4`；提高步长 10 倍后重要元素符号稳定。该 batch 的余弦对角 `[-0.634,-0.755,+0.612]`，两模式均得 `[0,0,3]`。这是既存工件，本次未再运行 GPU；单 batch 只能确认真实模型可出现正负并存。

## CPU 检验与故障分类

在隔离工作树使用 PyTorch 2.4.1 CPU：固定 TRAIN 前 16 条真实归一化标签为 meta 目标，核对前 64 条无标签 TRAIN ID；两份 batch ID 列表 SHA256 为 `28ca22ce917bc06f91c70644ead64f5c47385d193c60810b43db5ee626300a11f`、`bbb337382291bb49fb6e7cdaac9e4958646f7529e76a9db4497c33d4d5cfbddfc`。用 2→3 线性头和人工伪标签方向构造可解析效用：余弦对角 `[1,0.70710678,-1]`，非对角为零；原始差分对角约 `[+5.17185e-6,+5.16560e-6,-5.16409e-6]`，一阶项同符号；两模式输出 `[1.75735931,1.24264069,0]`。全负单位矩阵输入输出 `[1,1,1]` 且 `fallback=True`。全 1 权重损失与 HPL 原损失逐位相等；自动求导确认强视图有梯度，弱视图与 detached log variance 无梯度。合成特征不代表 RoBERTa 表征，仅验证公式和模块功能；TRAIN 标签未用于调参。

| 类别 | 判断 |
|---|---|
| 公式错误 | 未发现符号或约简错误；旧 M 与一阶方向一致，当前 N 是尺度无关方向代理。 |
| 实现错误 | 早期配置 commit 不完整；14:20 配置 commit 与日志字段不匹配。全非正回退是明确策略，不可误称为代码故障。 |
| 数值退化 | `5e-6` 使差分极小，float32 可能损失精度；旧探测用 float64，既有 GPU 报告支持重要符号可解析。余弦去掉步长量级，但不消除小 batch 梯度噪声。 |
| 真实信号不足 | 160 步平滑分数均负；可能来自伪标签负效用，也可能是 16 条 TRAIN meta 采样、head-only SGD 代理、动量滞后或 Adam/encoder 不匹配。单种子证据不能裁决因果。 |

## 文献覆盖及停止点

[HPL 原论文（NeurIPS 2025）](https://papers.neurips.cc/paper_files/paper/2025/file/30a129d6b8e4229d1ed6d90a65ee1a76-Paper-Conference.pdf) 的内层用监督与异方差加权伪标签损失更新回归器，外层用另一有标注 batch 的损失更新不确定网络；实用算法只对回归头展开二阶导数。当前 M0 已覆盖“TRAIN 标签指导伪标签可靠性”“双层优化”“每样本/每目标 log variance 加权”。M1 额外部分是维度级虚拟更新效用代理和权重映射。[AISTATS 2021 meta-gradient SSL](https://proceedings.mlr.press/v130/xiao21a.html)、[NeurIPS 2020 PCGrad](https://papers.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html) 已覆盖标签引导嵌套优化及梯度冲突处理。[NeurIPS 2025 UMPL](https://papers.nips.cc/paper_files/paper/2025/hash/6fc46679a7ba2ec82183cf01b80e5133-Abstract-Conference.html) 处理回归伪标签的不确定性反馈；[IJCNN 2025 MASSTER](https://imec-publications.be/entities/publication/3ccbff9b-428b-4303-ae6a-9bb47213f39e) 针对多目标回归选择 instance–target 对；[NeurIPS 2025 半监督多目标学习理论](https://papers.nips.cc/paper_files/paper/2025/hash/ba92c896779f1477b9fddf9f380ae3f3-Abstract-Conference.html) 已讨论伪标签与多目标样本复杂度。现有检索未证明 M1 维度效用公式或候选机制有论文级新颖性，不作创新声明。

唯一候选与最小消融见 `M1_METHOD_SPEC.md`。真实泛化效用和候选优劣仍待实验；本次在文档完成后停止。
