# 修正可学习性诊断（2026-10-22）

当前服务器已经完成 A/B、教师与 10 轮 source-only probe。下面的诊断复用该教师和 VDT 划分，从相同通用预训练与随机种子重新初始化四个 RGB 学生，比较训练目标。诊断代码独立于正式训练源码，不改变既有 A/B/教师的源码哈希，也不覆盖它们的报告或 checkpoint。

## 已观察到的结果

数据来自用户提供的 `correction_4gpu_ampfix/seed-20261012` 服务器输出，尚未在本地读取完整服务器文件。

| 指标 | 数值 |
|---|---:|
| 教师 RGB 关系误差 | 0.249303 |
| 教师 RGB/T/D 关系误差 | 0.219956 |
| 教师关系误差相对降低 | 11.77% |
| 正可靠边覆盖率 | 87.75% |
| 零修正的正可靠边 MSE | 0.00453602 |
| TRAIN 拟合常数的正可靠边 MSE | 0.00385588 |
| 学生第 5 轮正可靠边 MSE | 0.00500945 |
| 学生第 10 轮正可靠边 MSE | 0.00597874 |

第 10 轮学生 MSE 比零修正高 31.81%，比常数修正高 55.06%。训练监督损失从 1.88014 降至 0.52869（降低 71.88%），修正迁移损失从 0.18461 降至 0.17829（降低 3.42%）。10 轮各有 105 次有效更新，AMP 跳步均为零。这些结果不支持直接把原 probe 延长到 100 轮，也不能证明完整 COD 训练必然无效。

原训练目标是按可靠度加权、按区域平衡的 SmoothL1；筛选指标是正可靠边上未加权、跨尺度汇总的 MSE。两者优化目标不同，这是代码事实，不能单凭此差异断言实现错误。修正分支和共享适配层还收到分割及 GT 关系监督梯度，而教师修正量相对于教师自己的 RGB 基础预测定义；学生基础预测与之不同。联合监督是否造成干扰，需要对照实验，现有日志无法直接证明。

## 四组固定预算对照

| 诊断组 | 是否加入原 GT 分割/关系监督 | 修正损失 |
|---|---|---|
| `joint_balanced` | 是 | 原可靠度加权、区域平衡的 SmoothL1 |
| `joint_positive_mse` | 是 | 正可靠边 MSE × 5 |
| `isolated_balanced` | 否 | 原可靠度加权、区域平衡的 SmoothL1 |
| `isolated_positive_mse` | 否 | 正可靠边 MSE × 5 |

倍率 5 来自原 SmoothL1 在小误差区间的系数 `1 / (2 × beta)`，其中 `beta=0.1`。这只是一个预先固定的尺度选择，不意味着梯度大小完全匹配：可靠度、区域与跨尺度归约方式也已改变。新损失在每批次所有有效正可靠边上汇总平方误差，再除以这些边的总数；教师目标保持 detached，空支持返回可求导的零。

`isolated_*` 仅用蒸馏损失更新可达的学生适配层与修正头，不将 GT 监督损失加入优化目标；教师目标与可靠度仍使用 VDT TRAIN 标注。它不是无监督训练，不冻结随机初始化的适配层，也不保证与联合监督具有相同的特征训练轨迹。GT 损失可以继续记录用于诊断。

四组默认均训练 10 轮，使用相同的初始化、数据成员、shuffle 种子与其余原配置，固定取最后一轮。教师和两种 TRAIN 拟合常数保持不变。评价始终使用原 `positive_mse / positive_zero_mse / positive_constant_mse`，不会降低门槛、选择验证曲线中最好的一轮，或修改原 `all` 流程。源域可学习性通过也不能视为 COD 精度提升。

训练历史中的 `source_transfer` 仍记录原 SmoothL1 目标，`optimization_transfer` 记录当前组实际使用的蒸馏项，`raw_positive_mse` 为未乘 5 的 MSE。后者是使用增强数据、参数不断变化时的在线批次均值，并非固定 checkpoint 在完整训练集上的全局 MSE，不能直接拿它与验证值之差证明过拟合。

## 服务器执行

将诊断包解压到已有 AMP 修复版项目根目录。它只新增诊断脚本、测试和本说明，不需要再次下载 DINOv3，也不需要重训教师。

```bash
cd /data2/yyy/TriPRD
conda activate triprd-v3
export RUN_ROOT=/data2/yyy/TriPRD/results/optimization/correction_4gpu_ampfix/seed-20261012
export OUTPUT_ROOT=/data2/yyy/TriPRD/results/optimization/correction_probe_diagnostic_v1
export GPUS=2,3,4,5
export RESUME=0

# 核对计划，不启动训练。
DRY_RUN=1 bash tools/run_triprd_probe_diagnostics.sh

# 四组各一张卡，只训练 VDT source-only probe。
bash tools/run_triprd_probe_diagnostics.sh
```

若设置了 `CUDA_VISIBLE_DEVICES`，它必须包含所选的物理 GPU 编号。每个子进程只能看到分配给它的单张 GPU。配置、DINOv3 路径、教师权重和数据划分从已完成报告读取；路径失效、源码不匹配或教师校验失败时应修复相应路径/文件，不能伪造哈希绕过验证。

新输出目录包含各组的 `run.log`、`comparison.json`、`history.json`、`last.pt`，以及汇总 `summary.json`。这是独立诊断，完成后不会自动执行 `screen` 或 `final`，诊断 checkpoint 也不作为正式 COD 学生的初始化。

在相同诊断代码与配置下发生中断，可以运行 `RESUME=1 bash tools/run_triprd_probe_diagnostics.sh`。已完成组只校验并复用，未完成组从最近完整 epoch 的模型、优化器、scheduler、scaler 与随机数状态恢复。默认每组固定 10 轮；改变 `PROBE_EPOCHS` 或诊断代码后需使用新的诊断输出目录，不会给已有结果直接追加轮次。

## 如何使用结果

- 同一损失下，仅蒸馏明显优于联合监督：支持进一步检查联合目标的干扰，但不能单凭此结果声称测到了负梯度余弦。
- 同一监督方式下，正可靠边 MSE 优于原损失：支持调整损失归约与目标；应同时保留 GT 关系误差等辅助指标，避免只追求单项诊断分数。
- 只有“仅蒸馏 + 正可靠边 MSE”有效：两个改动可能需要共同作用，仍需单独验证其在正式 COD 联合训练中的用途。
- 四组均不通过：先检查训练集可拟合性、按尺度/区域的误差与教师残差信号，再决定改变修正表示或迁移目标。现有结果不能支持继续扩大正式训练预算。

所有对照都属于在看到首轮失败后新增的开发实验。后续若据此修改正式方法，应明确记录新的方法版本并做匹配的 COD 对照；不要把本诊断并入原先已冻结的筛选记录。
