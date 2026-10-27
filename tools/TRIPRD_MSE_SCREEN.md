# 联合 MSE 的 COD 验证筛选（2026-10-22）

用户提供的服务器诊断输出如下。本地没有服务器数据或 GPU，以下数值来自该输出；脚本运行时会重新校验实际报告和 checkpoint。

| 源域诊断 | 最后一轮正可靠边 MSE | 相对常数基线降低 | 通过两条基线 |
|---|---:|---:|---|
| 原 probe | 0.00597874 | −55.06% | 否 |
| joint_balanced | 0.00597618 | −54.99% | 否 |
| joint_positive_mse | 0.00330194 | 14.37% | 是 |
| isolated_balanced | 0.00374339 | 2.92% | 是 |
| isolated_positive_mse | 0.00251589 | 34.75% | 是 |

零修正为 0.00453602，VDT TRAIN 拟合的常数为 0.00385588。相同联合监督下，改用正可靠边 MSE 后能通过原门槛，说明当前学生能够学习有效的修正信号。两组仅蒸馏结果也支持继续研究联合目标的影响，但没有直接测量梯度冲突；MSE 归约与固定倍率同时改变，也不能把效果只归因于某一种权重。

先选 `joint_positive_mse` 进入 COD 对照，是因为它保留源域 GT 监督，能与已完成 B 和普通 KD 的 C 保持一致。仅蒸馏 MSE 的源域分数更低，可以留作后续消融；这个分数本身不能确定哪组 COD 分割最好。

## 这次运行的实验

新增入口 `tools/run_triprd_mse_screen.sh`，使用独立方法标识 `triprd_correction_positive_mse_screen_v1`。

| 组 | 动作 | 目标 |
|---|---|---|
| A | 校验并复用 | COD 监督 |
| B | 校验并复用 | COD 监督 + VDT RGB/GT 监督 |
| C | 从相同通用预训练重新训练 | B + 普通全关系 KD |
| D | 从相同通用预训练重新训练 | B + 正可靠边修正 MSE × 5 |

D 的总损失为：

```text
COD_supervised
+ source_weight * (VDT_supervised + kd_weight * 5 * pooled_positive_edge_MSE)
```

`COD_supervised`、`VDT_supervised` 均保留原分割、辅助分割和 GT 关系监督；倍率、source/kd/relation 权重与通过诊断的方案一致。正可靠边只由教师在 VDT TRAIN 标注上的关系误差改善确定。MSE 汇总当前批次两尺度全部正可靠边的平方误差，再除以边数；教师目标停止梯度，空支持返回可求导的零。倍率 5 是固定设计选择，不保证与旧平衡 SmoothL1 的梯度幅度匹配。

训练以 COD 批次为主，每步配一个循环取用的 VDT 批次。轮数、学习率、调度、AMP、增强、batch size、数据成员和验证间隔从原 A/B 报告读取；本次服务器配置应为 100 轮、3232 COD TRAIN / 808 内部 VAL、838 VDT TRAIN / 210 VAL。无法通过参数临时缩短或延长预算。C/D 初始状态必须与 A/B 和源域诊断一致，教师与 DINOv3 编码器保持冻结；诊断学生权重仅作为证据验证，不载入正式学生。

新脚本读取并校验原源码哈希、配置、划分、教师、A/B 的 best/last checkpoint 和已通过的联合 MSE 诊断。它不修改旧训练代码、失败 probe、原报告或诊断结果。若缺文件或哈希不一致，会在训练前报出错误；应恢复对应文件或路径，不能手改哈希。

## 在服务器运行

将补丁包在现有项目根目录解压。已有 AMP 修复版和诊断脚本是依赖；无需重新下载 DINOv3、重训 A/B 或教师。

```bash
cd /data2/yyy/TriPRD
conda activate triprd-v3
unzip -o /path/to/triprd_mse_screen_2026-10-22.zip -d .

export RUN_ROOT=/data2/yyy/TriPRD/results/optimization/correction_4gpu_ampfix/seed-20261012
export DIAGNOSTICS_ROOT=/data2/yyy/TriPRD/results/optimization/correction_probe_diagnostic_v1
export OUTPUT_ROOT=/data2/yyy/TriPRD/results/optimization/correction_mse_screen_v1
export GPUS=2,3
export RESUME=0

DRY_RUN=1 bash tools/run_triprd_mse_screen.sh
DRY_RUN=0 bash tools/run_triprd_mse_screen.sh
```

只需要两张卡：C、D 各一张，没有 DDP。设置 `GPUS=2,3,4,5` 时也只会启动两个训练任务。若设置了 `CUDA_VISIBLE_DEVICES`，所选物理 GPU 编号必须在其中。剩余显卡可用于其他任务。脚本继承原报告中的 DINOv3 源码及 `model.safetensors` 路径，不新增下载。

中断后使用相同目录与代码恢复：

```bash
RESUME=1 DRY_RUN=0 bash tools/run_triprd_mse_screen.sh
```

已完成的组只做校验；未完成的组恢复最近完整 epoch 的模型、优化器、学习率调度、scaler 和随机数状态。输出目录有独占锁；正常退出会清除锁。若进程被强制终止留下锁，先核实该 launcher 及子进程确已结束，再处理该锁。

## 查看结果

新输出包括 `C_full_kd/`、`D_joint_positive_mse/` 中的 `run.log`、`comparison.json`、`history.json`、`best.pt`、`last.pt`，以及根目录的 `manifest.json`、`summary.json`。

各组按相同间隔在内部 COD VAL 上选择最高 macro Sm 的 checkpoint，平分时取最早轮。汇总同时提供该轮各项指标、D−A/B/C 的 macro Sm 差值以及 `D_beats_all_ABC`。不要把不同轮次各自最好的指标拼在一行。D 的训练日志中 `source_transfer` 为实际优化的 5×MSE，`raw_positive_mse` 为未乘倍率的值，`original_source_transfer` 只作旧损失诊断；这些在线批次均值与固定 checkpoint 全验证集 MSE 的聚合范围不同。

本次只访问训练池及内部验证集，不加载官方测试样本，不自动进入完整 4040 张重训。`D_beats_all_ABC=True` 仅是单划分、单随机种子的开发证据，不代表显著性或论文 SOTA。通过后再为冻结的新方案准备完整训练池重训与 CamoFormer 一致的官方测试评测；本次分数不能直接填入 `previous_performance.tex`。新版本 checkpoint 也不能直接交给仅接受旧 v3 方法的评测器。

旧 `run_triprd_optimization.sh all` 继续保留旧 probe 门槛。新的实验使用上面的独立入口。

## 本地验证

PyTorch 2.8.0 CPU 环境：新增测试 12 项通过、8 个子测试通过。实际 tiny 模型验证了新 C 与旧 C 的最终权重及验证指标完全一致，D 的全部梯度与独立汇总 MSE 的计算一致，且同时保留 COD/SOD 监督梯度；还验证了精确断点恢复、完成后不重训、证据/初始化校验、禁止官方测试调用、真实子进程 GPU 环境与 Bash 参数传递。原 AMP 和诊断回归测试另有 28 项通过、30 个子测试通过，1 项 CUDA 专项因本地没有 CUDA 而跳过。未在本地执行服务器 DINOv3 全量训练；COD 精度改善仍需本次服务器筛选确认。
