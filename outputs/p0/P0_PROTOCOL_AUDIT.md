# P0 协议审计报告

## 1. 数据身份

- 计数：{'lb': 500, 'ulb': 7562, 'dev': 1000, 'test': 1000}
- train union = 8062（应 8062）：True
- 所有交集为 0：True；overlaps={'lb&ulb': 0, 'lb&dev': 0, 'ulb&dev': 0, 'lb&test': 0, 'ulb&test': 0, 'dev&test': 0}
- `train_ulb` 列严格为 id/text：True（实际 ['id', 'text']）

## 2. 标签与尺度

- lb: finite=True in[1,5]=True norm_consistent=True mean={'V': 2.982, 'A': 3.059, 'D': 3.062}
- dev: finite=True in[1,5]=True norm_consistent=True mean={'V': 2.962, 'A': 3.031, 'D': 3.061}
- test: finite=True in[1,5]=True norm_consistent=True mean={'V': 2.978, 'A': 3.034, 'D': 3.065}

## 3. 代码扫描

- `scripts/vad_research/train_acvr.py` (training): raw_csv=False oracle=True dev_mean=True model_encoder=True
- `scripts/vad_research/train_roberta_vad_supervised.py` (training): raw_csv=False oracle=False dev_mean=False model_encoder=False
- `scripts/vad_research/reliability_ceiling.py` (diagnostic): raw_csv=True oracle=False dev_mean=False model_encoder=True
- `scripts/vad_research/structure_diagnostic.py` (diagnostic): raw_csv=True oracle=False dev_mean=False model_encoder=True
- `scripts/vad_research/local_affine_oracle.py` (diagnostic): raw_csv=False oracle=False dev_mean=False model_encoder=True

## 4. 历史 checkpoint 选择

- p1_s0: exists=True best_ckpt=True selection_is_dev=True
- alpha_fixed075_s0: exists=True best_ckpt=True selection_is_dev=True

## 5. 必须修复/声明的风险

1. 历史 P1 与 alpha_fixed 均按 DEV MAE 选择 checkpoint，DEV 不是盲测集。
2. 旧 VRRC 的 `p.mean(0)` 是当前评估批次均值，不是可部署的单样本统计量。
3. JACF/VRRC 使用微调后的 encoder 计算邻域；不同 OOF fold 的 encoder 空间不一致。
4. 可靠性/结构诊断脚本读取过 7562 条无标号文本的隐藏标签；这些标签只能用于特权诊断。
5. TEST 已有历史评估记录，后续只能一次性使用，不能再参与选择。

## 6. 结论

- 数据 ID 与标签尺度审计通过。
- 进入 P1 前必须固定：OOF 折划分、P1/JACF 固定步数、统一检索特征、冻结中心化统计。
