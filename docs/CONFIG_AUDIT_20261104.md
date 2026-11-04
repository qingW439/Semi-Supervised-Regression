# 实验配置自审（2026-11-04 10:35）

针对"这次配置是否都正确"的逐项核查。结论：**发现 2 处我自己的配置/工程缺陷（已修）+ 1 处环境故障（需你处理）+ 1 个需要你确认的口径问题**。

## A. 环境故障（阻塞中，需要你操作）

- 今早 **10:18 AutoDL 容器重启**（jupyter/tensorboard 进程启动时间可证），重启后**没有 GPU**：
  `torch.cuda.is_available() = False`、`device_count = 0`、`nvidia-smi: Permission denied`。
- 后果：10:20 启动的强基座 M1 网格**一个结果都没产生**（`outputs/strong_b1600/*` 三个目录均为空）。
  我此前误读了日志（把昨晚配方日志里的 `FOLD ... 0.15417` 当成本次输出），已纠正。
- 数据无损：`/root/autodl-tmp` 全在，昨晚三组 E0\* 候选（各 15/15）与 ACLDP 45 次折训练全在。
- **需要你做**：在 AutoDL 控制台确认实例是否处于"无卡模式"或需要重启并挂载 GPU；
  恢复后告诉我（端口若变化也请告知），我立即续跑。

## B. 配置缺陷 1：强基座网格缺少必需的两个对照臂（已修）

原脚本只有 3 条臂（base / jacf / m1），**缺**：

| 缺失臂 | 为什么必须有 |
|---|---|
| `sup96`（base + 96 步纯监督，λ_u=0） | 预注册 §4 与任务书 E5 都要求"匹配纯监督 Control"，否则无法区分"M1 的收益来自无标签数据"还是"多训了 96 步" |
| `m1_shuf`（强基座上打乱差异参照） | 强基座上的负控制；弱基座做过（CCC 0.346 vs 0.446），强基座上必须重做 |

现已改为 **5 条臂共享同一 base 权重**：`base / sup96 / jacf / m1 / m1_shuf`。
脚本加了 `bash -n` 语法检查与 `torch.cuda.is_available()` 启动前断言。

## C. 配置缺陷 2：静默 CPU 回退（已修）

`audit_sup_recipe_fold.py` 原来是
`dev = torch.device(a.device if torch.cuda.is_available() else "cpu")`
——GPU 不在时会**静默降级到 CPU**继续跑（`train_oof_fold.py` 则是抛错）。
这正是今早第一批"看起来在跑"的假象来源。已改为显式抛错：
`RuntimeError: CUDA requested but not available (refusing to silently fall back to CPU)`。

## D. 逐项核查通过的部分

| 检查项 | 结论 |
|---|---|
| 是否用了 dev | OOF 脚本只读 `lb500_s{seed}`；dev 仅在最终报告。`label_isolation.json` 记录 `holdout_labels_used_in_training=false` |
| 折外泄漏 | 每折从 `roberta-base` 重训编码器（此前"用全局 P1 特征做 OOF"的污染已作废并记录在 `E0_AUDIT_CONTAMINATION_NOTE.md`） |
| 差异参照泄漏 | ACLDP 的参照在**折内**构造（该折 400 条的 P1 或闭式 ridge），不看折外标签 |
| 划分一致性 | 脚本断言 400/100、train∩holdout=∅、manifest 与 label id 集合相等 |
| 超参选择 | ridge 的 α/预处理只在 400 内部 5 折 CV 选；λ_diff 固定 0.05 不扫；M1 的配方集合预先写死 |
| 配对性 | 强基座 5 臂共享同一 base 权重；E0\* 冻结用的配方与 base 臂是同一脚本同参数同种子（确定性） |
| 指标口径 | 统一 `metric_report`/自写同式，MAE 按 1–5 原尺度；逐维与宏平均分离报告 |

## E. 需要你确认的口径问题：E0\* 的 lr 来源

E0\* 选中的 1600 步配方使用 **lr=1e-4**，而这个 lr 来自我早前**看过 dev** 的配方对照（已披露）。
"完全不含 dev 调参"的对照版是 640 步 / lr=2e-5 / bs16，其 OOF MAE 为 **0.17980**（比 0.17641 差 0.0034）。

我的处理建议（也是当前做法）：**E0\* 取更强的 lr=1e-4 版**，因为
**用更强的基线做参考只会让 M1 的判定更严格，不会放水**；同时在论文中把 lr=2e-5 版作为敏感性结果报告。
若你要求严格口径（E0\* 不得含任何 dev 信息），我改为 `640/lr2e-5`，M1 的通过门槛会更宽松（这一点我会在总账中注明）。

## F. 修复后的状态

- `run_strong_base_m1.sh` v2：5 臂、启动前 CUDA 断言、`bash -n` 通过。
- 失败留下的空目录已清理（未删除任何 parquet/特征/配置）。
- GPU 恢复后一条命令即可续跑：`BASE_STEPS=1600 BASE_LR=1e-4 BASE_BS=64 TAG=strong_b1600 nohup bash run_strong_base_m1.sh &`
