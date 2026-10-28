# LACT-VAD 数学修订与实现核对

**审查结论：数学形式可实现；本次不准入完整训练。** 本文替代先前 `M1_METHOD_REDESIGN.md` 中把 P2 头写成 `Wz+b` 的定义。计算仅用 EmoBank TRAIN 与已有模型；无优化器更新，未读取 DEV 或 TEST。

## 1. 真实 P2 与修订后的目标

现有 `RobertaVADRegressor` 是 RoBERTa-base → attention-mask mean pooling → `Dropout(0.1)` → `Linear(768,768)` → GELU → `Linear(768,3)`。记池化表示为 `zθ(x)`，确定性非线性头为 `hφ(z)=W₂ GELU(W₁z+b₁)+b₂`。监督训练仍调用原 P2：

`f_train(x)=hφ(Dropout₀.₁(zθ(x)))`，`L_S=mean_{i,d}|f_train,d(x_i)−y_i,d|`。

评估路径为 `f_eval(x)=hφ(zθ(x))`。LACT 正则只在确定性头上计算局部导数，不改监督路径、Dropout 概率或预测参数：

`C_zy=(1/n)Σ_i(z_i−z̄)(y_i−ȳ)^T`，`Q=orth(C_zy)`，`P=QQᵀ`，

`d_uv=zθ(v)−zθ(u)`，`r_uv=(I−stopgrad(P))d_uv`，

`L_U=mean_(u,v)∈E { ||J_h(zθ(u))r_uv||₂² / [||stopgrad(d_uv)||₂²+10⁻⁸] }`。

`J_h(z)r` 是 **Jacobian-vector product**，不是以 `W₂` 或一个新线性头代替真实头。`Q` 只由指定 TRAIN 拟合集标签与其表示计算；图的整数邻居索引固定；分母停止梯度。分子中的 `z_u,z_v` 不停止梯度，头的 `W₁,b₁,W₂` 和编码器均可得到梯度。`b₂` 在方向导数中严格抵消，因此 **无标签项对末层偏置梯度为零**，监督项仍更新它。这是结构性质，不能声称无标签项更新头的每一个参数。

总损失若进入后续训练，才是 `L=L_S+λL_U`；本审查只分别求梯度，没有合成损失并更新权重。原设计的一次性初始标定式 `λ=0.2||g_S,head||/(||g_U,head||+ε)` 保留作风险核对，不据审查结果调参。

## 2. 正确性证据

CPU 单测使用真实 `Linear→GELU→Linear` 头，在 `float64` 下对照中心有限差分：

`J_h(z)r ≈ [h(z+10⁻⁵r)−h(z−10⁻⁵r)]/(2·10⁻⁵)`，容差 `rtol=10⁻⁶, atol=10⁻⁸`。单测还检查 `z_u,z_v,W₁,b₁,W₂` 梯度非零且有限、`Q` 无梯度、分母冻结造成的梯度与未冻结版本不同、互近邻索引为静态整数、正则不消耗 Dropout RNG。GPU 固定 batch 的真实 RoBERTa 前向与梯度见 `LACT_GRADIENT_AUDIT.md`。

在已有 400/100 参考模型上重复构造 `Q`，投影矩阵最大绝对差为 **0**。这只证明相同模型/数据下重算稳定；本轮没有执行参数更新，故不能证明训练节点前后 `Q` 稳定。已有五折模型与固定坐标比较见子空间审计。

## 3. 实现边界

诊断代码位于提交中的 `scripts/vad_research/lact_audit/`，不是 M1-v4 训练脚本。`torch.func.jvp` 对真实头计算一阶方向导数，反传该损失涉及混合二阶导，额外占用前向、反向时间与显存。头无 Dropout；无标签编码器在该审查中用 `eval()` 消除随机性但保留 autograd，监督分支维持 P2 的训练模式。现有测试覆盖 RNG 状态不变，不代表未来多步训练的性能或稳定性。

完整机器可读数值见 `LACT_AUDIT_RAW.json`；代码与测试的真实提交记录见 `LACT_GO_NOGO.md`。
