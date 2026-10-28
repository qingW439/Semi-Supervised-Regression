# LACT-VAD 创新边界复核

本审查补充先前 `M1_LITERATURE_COMPARISON.md`，重点看 LACT 的**数学机制**是否已有直接先例。下表公式为便于对照的机制式，除注明原文表达外不冒称论文中的公式编号。

| 真实论文 | 已有机制与任务 | LACT 的关系 |
|---|---|---|
| [The Manifold Tangent Classifier, NeurIPS 2011](https://papers.nips.cc/paper_files/paper/2011/hash/d1f44e2f09dc172978a4d3151d11d63e-Abstract.html) | 从无标签流形方向构造 TangentProp 类惩罚，可写成 `||J_f(x)t_x||²`；目标为半监督**分类**，流形方向来自表示学习。 | 无标签切向不变性与 Jacobian 惩罚已有。LACT 的 `Q` 来自 VAD 标签协变、只压 `Q` 外方向，是任务适配；不能把 JVP 或切向正则列为原创。 |
| [Semi-supervised Regression via Parallel Field Regularization, NeurIPS 2011](https://proceedings.neurips.cc/paper/2011/hash/bc6dc48b743dc5d013b1abaebd2faed2-Abstract.html) | 已直接研究**半监督连续回归**；令函数梯度场尽量平行，原文的核心二阶正则为 `R₂(V)=∫_M||∇V||²_F`，联同标签误差与函数/向量场一致性优化。利用无标签点近似数据流形和局部几何。 | “无标签几何改善回归”“向量场/导数正则”都已有。PFR约束梯度场的变化，LACT约束 `Q` 外的一阶输出响应；数学上不同但同属流形导数正则家族。 |
| [Tangent-Normal Adversarial Regularization, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Yu_Tangent-Normal_Adversarial_Regularization_for_Semi-Supervised_Learning_CVPR_2019_paper.html) | 将 VAT 的局部一致性分解为流形**切向**与**法向**扰动，分别构造 TAR/NAR；半监督分类。 | 正交方向拆分和各方向不同约束已有。LACT 的 `Q` 是**标签协变方向**，不是无标签流形法空间；其余向量也不必然是流形切向，因此不能借 TNAR 理论直接称其安全。 |
| [ConR: Contrastive Regularizer for Deep Imbalanced Regression, ICLR 2024](https://openreview.net/pdf/f7eb5919b3cb64559ec4b21e055fb75e75e9b80b.pdf) | 在**有标签、标签不均衡的连续回归**中，用连续标签的局部/全局关系做对比正则，保护少数标签特征免被多数邻居挤压。 | 连续标签引导表示几何已有。LACT 不构造正负对或标签相似性权重，目标是无标签边的 `Q` 外头响应；但“标签锚定方向保护”这一宽表述不足以构成原创。 |

先前比较过的 [HPL, NeurIPS 2025](https://papers.neurips.cc/paper_files/paper/2025/hash/30a129d6b8e4229d1ed6d90a65ee1a76-Abstract-Conference.html)、[UMPL, NeurIPS 2025](https://papers.neurips.cc/paper_files/paper/2025/hash/6fc46679a7ba2ec82183cf01b80e5133-Abstract-Conference.html) 用无标签连续伪值/不确定度，[SSR-RCUS, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/39487) 和 [RankUp, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c26a8494fe31695db965ae8b7244b7c1-Abstract-Conference.html) 用排序关系，[Kim & Kim, ICML 2026](https://proceedings.mlr.press/v306/kim26ae.html) 用无标签谱特征加标签 Ridge，[YNU-HPCC, SemEval 2026](https://aclanthology.org/2026.semeval-1.43/) 用 SimCSE 与词典情绪邻域。LACT 没有这些具体目标，但**不使用已有方法不自动产生创新性**。

## 数学上究竟剩下什么

LACT 是 `C_zy` 的监督子空间投影与无标签图方向二阶矩的组合：在局部常 Jacobian 近似下，`L_U≈tr[J_hᵀJ_h (I−P) C_graph (I−P)]`。标签决定保护方向 `P`，无标签图决定 `C_graph` 的方向权重。两部分单独都不是原创。只有当 `C_graph` 与随机/各向同性对照显著不同，而且这种差异给出可用的回归梯度，才有**可检验的组合机制**。

本次看到真实方向二阶矩有效秩 147.94（`Q` 外最多 765），真实/随机图梯度余弦 .531/.628，故无标签图不是严格冗余的各向同性头惩罚。但没有同锚点等尺度的普通头正则梯度对照，也没有完成模型收益；“标签锚定方向保护”不能宣称独立的普适数学发明。尤其在当前 `λ≈148` 时，信号增益的稳定性未成立。

**允许的准确表述**：一个针对 EmoBank 三维 VAD、无需无标签数值目标的标签条件图方向 Jacobian 正则假设，使用真实 P2 非线性头。**不允许的表述**：首创切向正则、Jacobian、子空间投影、局部图、连续标签几何，或已证明优于 P2。当前审查为 NO-GO，故不能宣称其为成功的 M1 核心创新。
