# EmoBank M0 核谱半监督回归准入准备

**结论：BLOCKED（协议准入）；seed 0 的代码、缓存与 CPU 检查已具备运行条件。本轮未运行 B0/B1/B2 完整拟合，未访问 TEST。**

## 1. 工作树与资产保护

- 主项目：`/hy-tmp/rankup_project/Semi-Supervised-Regression`，实际路径 `/root/autodl-tmp/rankup_project/Semi-Supervised-Regression`；HEAD `e45575d642b301e0410642054cef6ee0999bdc7d`。审计前已有 7 个未提交的 M0 代码/测试改动（4 个 spectral 文件、3 个 HPL 文件），以及未跟踪的 `docs/`。先读取了 `git status`、四个 spectral 文件的 diff、现有测试与旧输出。主项目上述代码和历史结果均未改动。
- 为保护这些改动，新建独立工作树 `/root/autodl-tmp/rankup_project/Semi-Supervised-Regression-spectral-readiness`，分支 `vad-m0-spectral-readiness`；只复制当前四个 spectral 文件，并给 `run_spectral_ridge.py` 添加必选 `--mode {b0,b1,b2}`，使一次命令只运行一组。未增加模型或模块，也未改 HPL、LACT 或已有模型。
- 已有输出 `saved_models/vad_research/m0/spectral_lb500_s0/` 与 `spectral_fixed_lb500_s0/` 均含 `comparison.json`、`dev_predictions.csv`。两份结果的 B0 DEV Macro-MAE 均为 0.177539，B1 均为 0.180401；B2 分别为 0.185342 与 0.182450。它们都已经看过 DEV，且结果文件没有代码哈希/提交号，不能用其差异反向选核参数或判定当前工作树的完整复现。下方拟执行命令使用新目录，拒绝覆盖旧输出。

## 2. 与 ICML 2026 原论文的对应关系

[Kim & Kim, *Semi-Supervised Learning with Noisy Proxy Covariates: Generalization Bounds and Distribution Regression*, ICML 2026](https://proceedings.mlr.press/v306/kim26ae.html) 的[全文](https://arxiv.org/pdf/2606.00512)定义先用全部 `N` 个代理协变量估计核算子

\[
\widehat L_N h=\frac1N\sum_{i=1}^{N}\kappa(\widehat X_i,\cdot)h(\widehat X_i),
\quad \widehat\Phi_s(x)=(\widehat\phi_1(x),\ldots,\widehat\phi_s(x)),
\]

再仅用 `n` 个真实标签求解

\[
\widehat\gamma=\arg\min_\gamma \frac1n\sum_{i=1}^{n}
(Y_i-\gamma^\top\widehat\Phi_s(\widehat X_i))^2+\xi\|\gamma\|_2^2.
\]

论文的单标量响应在此扩展为 V/A/D 三输出。B0 是冻结 RoBERTa 表示的普通 Ridge；B1 仅从 500 条标注 TRAIN 文本学核谱基；B2 在相同特征来源及相同 500 个标注锚点上，额外用 7562 条无标签 TRAIN 文本学核谱基。三组 Ridge 均只拟合 500 个真实 TRAIN 标签。此实验只能检验当前文本表示上的核谱机制，不能直接援引论文对*有噪代理协变量*的泛化界；RoBERTa 向量是否符合其测量误差、谱衰减等假设尚未验证。

**忠实的结构关系：** 核特征先由 TRAIN 协变量学习，标签仅进入 Ridge；B2 比 B1 多用无标签 TRAIN 协变量；DEV 不进入核基构造或参数选择。当前 `spectral_pair` 在每个 5 折内部仅用拟合折的标注文本选共同锚点，排除该折验证标签；B1/B2 的锚点、核函数、秩和 Ridge 选法一致。

**工程近似：** `spectral_core.py` 以 256 锚点 Nyström 近似替代论文的完整 `N×N` 核特征分解；`gamma=1` 的 RBF 核用于 L2 归一化的 768 维 RoBERTa 向量；只保留 64 个谱方向，并以 `1e-8` 截断近零锚点特征值。`StandardScaler()+Ridge` 改变各谱坐标的惩罚尺度；5 折以原始 [1,5] 尺度 MAE 选 `alpha∈{0.01,0.1,1,10,100,1000}`，论文估计器及验证分析使用平方损失。不能称其为论文完整复现；它是固定表示、有限秩的机制对照。论文在 SSDR 密度估计实验中以交叉验证选 RBF 带宽，并讨论从经验特征值衰减确定秩；并未为 RoBERTa 文本给出 `gamma=1`、`rank=64`、`landmarks=256` 的固定值。当前项目也没有记录这些值的 TRAIN 内部选择依据。

## 3. 特征缓存和数据边界

seed 0 可复用缓存：`/hy-tmp/rankup_project/Semi-Supervised-Regression/saved_models/vad_research/m0/pretrained_features_lb500_s0/`，压缩 NPZ 25,944,993 字节，元数据声明冻结预训练 RoBERTa-base、attention-mask mean pooling、`max_length=128`、不使用 P1 微调权重。实际预训练 `model.safetensors` SHA256 与元数据同为 `5bde1d28afb363d0103324efeb5afc8b2b397fe5e04beabb9b1ef355255ade81`。当前标注 TRAIN、无标签 TRAIN、DEV 文件哈希与元数据逐项一致（分别以 `c7c67afd…`, `2d85de90…`, `e8c39e89…` 开头）。缓存三个 ID 数组与对应源数据**顺序完全一致**；标注 TRAIN 和 DEV 的归一化标签逐值一致；形状分别为 `(500,768)`、`(7562,768)`、`(1000,768)`，值均有限，ID 无重叠。源读取器检查 Reader/Writer 加权后的 VAD 在 [1,5]，并验证 `y_norm=(y-1)/4`。脚本最后用现有 `metric_report` 还原原始尺度评价。

上述检查确立文件、ID 和标签来源；由于本轮未重新做 GPU 前向，不能从 NPZ 内容独立证明每个向量确由该权重产生。`checked_features` 自身只检查缓存内部一致性，不重新验模型哈希与源数据哈希，所以运行前应保留本次外部核验记录。当前仅发现 seed 0 的特征缓存；seed 1/2 缺失，不能用 seed 0 缓存冒充别的标注划分。

## 4. CPU 检查与资源

在原项目运行 `python -m pytest -q tests_m0/test_spectral_core.py tests_m0/test_spectral_runner.py`：**7 passed**；隔离工作树增加单组 CLI 检查后：**8 passed**（均仅有外部依赖 SQLAlchemy 弃用警告）。测试覆盖小矩阵核重建、训练基到 DEV 的纯投影、外部锚点复制、输入校验、DEV 不参与拟合、5 折拟合子集与 B1/B2 共享锚点。另做只读 CPU 数值探针：L2 范数最大误差 `2.22e-16`；B0/B1/B2 的训练/验证形状和同参重复输出完全一致，结果有限；近重复锚点的特征仍有限。隔离脚本 `--mode b0 --check-only` 成功读取缓存但未拟合/写文件。尚无真实 8062 样本的时间与峰值内存实测，也未量化 Nyström 相对完整核的近似误差。

完整 B2 的 (8062^2=64,995,844) 核元素若用 float64 存储约 520 MB，完整特征分解计算更重；现实现主要矩阵 (8062\times256) 约 16.5 MB，另有同级临时数组、Ridge CV 与 Python/BLAS 开销。256×256 分解和每折约 7962×256 核计算在 CPU 上进行；seed 0 缓存拟合不占 RTX 4090 显存。缺失缓存的 RoBERTa 提取才需要 GPU；提取命令如下，仅列出、不执行：

```bash
cd /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-spectral-readiness
/root/autodl-tmp/rankup_project/envs/rankup/bin/python scripts/vad_research/m0/extract_roberta_features.py --budget 500 --seed 1 --device cuda --batch-size 32 --max-length 128 --out saved_models/vad_research/m0/pretrained_features_lb500_s1_readiness
/root/autodl-tmp/rankup_project/envs/rankup/bin/python scripts/vad_research/m0/extract_roberta_features.py --budget 500 --seed 2 --device cuda --batch-size 32 --max-length 128 --out saved_models/vad_research/m0/pretrained_features_lb500_s2_readiness
```

## 5. B0/B1/B2 精确拟执行命令

以下三条是隔离工作树当前 CLI 的**可执行命令**，均指向经核验的同一 seed 0 特征缓存，且输出目录当前不存在。现阶段仅记录，不执行；在解除下节协议阻塞后才运行。B1/B2 参数完全相同；B0 虽接收核参数，但不会用它们构造特征。

```bash
cd /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-spectral-readiness
/root/autodl-tmp/rankup_project/envs/rankup/bin/python scripts/vad_research/m0/run_spectral_ridge.py --mode b0 --budget 500 --seed 0 --feature-dir /hy-tmp/rankup_project/Semi-Supervised-Regression/saved_models/vad_research/m0/pretrained_features_lb500_s0 --rank 64 --landmarks 256 --gamma 1.0 --out saved_models/vad_research/m0/spectral_b0_lb500_s0_readiness
/root/autodl-tmp/rankup_project/envs/rankup/bin/python scripts/vad_research/m0/run_spectral_ridge.py --mode b1 --budget 500 --seed 0 --feature-dir /hy-tmp/rankup_project/Semi-Supervised-Regression/saved_models/vad_research/m0/pretrained_features_lb500_s0 --rank 64 --landmarks 256 --gamma 1.0 --out saved_models/vad_research/m0/spectral_b1_lb500_s0_readiness
/root/autodl-tmp/rankup_project/envs/rankup/bin/python scripts/vad_research/m0/run_spectral_ridge.py --mode b2 --budget 500 --seed 0 --feature-dir /hy-tmp/rankup_project/Semi-Supervised-Regression/saved_models/vad_research/m0/pretrained_features_lb500_s0 --rank 64 --landmarks 256 --gamma 1.0 --out saved_models/vad_research/m0/spectral_b2_lb500_s0_readiness
```

## 6. 准入阻塞项

1. **核设置尚未满足来源约束。** `gamma=1`、`rank=64`、`landmarks=256` 是现有脚本默认值，未见论文中对应文本任务的固定设定或 TRAIN 内部验证记录。论文其他实验中的 RBF 带宽选择不能直接移用于本任务；旧 DEV 结果也不能作为选择依据。应先书面固定仅用 TRAIN 内部的核参数规则及 Nyström 近似精度判据，再执行三组；本轮不追加搜索。
2. **完整多种子准备不足。** 只有 seed 0 缓存；seed 1/2 若纳入后续正式比较，须按各自标注 TRAIN 划分重新提取同一预训练 RoBERTa 特征，并逐项核验 ID、权重及数据哈希。本轮不提取。
3. **旧结果可复现性有限。** 两份已完成 seed 0 输出没有代码哈希，且现有工作树有未提交的共享锚点改动。旧结果只作资产清点，不用于参数选择或当前版本的结论。

解除上述协议阻塞前结论保持 **BLOCKED**。这不否定代码可运行或 CPU 单测通过，也不构成 B2 改善 MAE/CCC 的证据。
