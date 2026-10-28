# LACT-VAD 一次性机制准入决定

## 决定：**NO-GO**

截至 2026-10-28，本轮只完成真实 P2 架构下的数学修订、TRAIN 内子空间与无标签图审计、固定 batch 前向/梯度诊断。**停止当前 LACT 定义；不启动 160-step 或完整 M1 训练。** 这不是对一切无标签局部方向正则的否定，也不是 P2 性能比较；本轮没有训练候选模型，所以无法宣称超过 P2、改善 MAE/CCC 或形成论文创新。

| 预设准入条件 | 证据 | 判定 |
|---|---|---|
| 数学实现严格正确 | 保留 P2 `Dropout→Linear→GELU→Linear`；真实头 JVP 与有限差分一致；Q/图索引/分母停梯度；监督 RNG 不受正则影响 | 通过 |
| Q 有折外关联 | 5 折折外协变捕获 .363–.682；400→100 为 .572，高于随机/拟合集标签打乱；A/D 与固定坐标个别折较弱 | 初步通过，动态刷新未证实 |
| 无标签图有独立方向结构 | 8,984 互近邻边；真实方向有效秩 147.94，非各向同性；真实/随机图梯度不同 | 部分通过；23.76% 孤立点，未证实方向对 VAD 有益 |
| 无标签梯度有限、非退化 | 头 .135%、encoder .433% 的未缩放监督比；`λ₀=148.077` 后 encoder 达监督的 64.10%，两组与监督余弦仅 .066/.020 | **未通过稳定有效梯度门槛** |
| 与普通正则有可检验区别 | 图协方差与各向同性、随机图不同；但缺精确同锚点监督头正则梯度对照 | 未充分证明机制层独立性 |

这里的 NO-GO 来自**原定义下、一次性固定 batch 的异常梯度尺度**，以及普通头正则等价性仍未完全排除；不以调整 λ、K、秩或追加超参数搜索补救。`Q` 折外有关联、图非各向同性，是保留下来的诊断事实，不足以覆盖其余准入缺口。之前 P2 三种子 DEV Macro-MAE **0.175589±0.000676** 仅为已有基线，本轮没有新增 DEV 评价。

## 可复核范围与隔离

- 数据：EmoBank seed0 的 500 有标签 TRAIN、7,562 无标签 TRAIN；仅 400/100 与 320/80 **TRAIN 内**留出用于机制审计。数据 SHA256 `c7c67afd9c62b71cc7dfd5a31e4958072805f157dc7bf92fa06d52f3ba9fb094`（有标签 TRAIN）。
- 复用：既有五折和 400 参考 checkpoint，未重训；固定 P1 checkpoint 仅作无标签图表示，未把其 500 标签训练表示用于 Q 折外宣称。原始审计基础提交 `e45575d642b301e0410642054cef6ee0999bdc7d`。
- 计算：CPU 单测、GPU 前向/反向；**optimizer steps=0**；未读取 DEV/TEST。未改标签版本、历史 checkpoint 或原 checkout 的未提交 M0 文件。
- 工作区：独立 Git worktree `/root/autodl-tmp/rankup_project/Semi-Supervised-Regression-lact-audit`，分支 `vad-lact-mechanism-audit`；源 checkout 的未提交文件保持原状。机器可读结果 `LACT_AUDIT_RAW.json`。

## 提交与复现记录

审计源代码：`scripts/vad_research/lact_audit/{lact_core.py,audit.py,test_lact_core.py}`。本地交付还包含真实 `LACT_GIT_DIFF.patch` 与 `LACT_GIT_LOG.txt`；其中记录远端提交和文件差异。CPU 命令：

```bash
cd /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-lact-audit
PYTHONPATH=scripts/vad_research/lact_audit /root/autodl-tmp/rankup_project/envs/rankup/bin/python -m pytest -q scripts/vad_research/lact_audit/test_lact_core.py
```

GPU 审计命令与绝对输入路径见 `LACT_GIT_LOG.txt`。该审计只读已有资产并拒绝覆盖已有结果目录。原先的 `M1_METHOD_REDESIGN.md` 把 P2 头写成单层线性头，已由 `LACT_MATH_CORRECTION.md` 明确撤销；不能继续按旧式实现训练。
