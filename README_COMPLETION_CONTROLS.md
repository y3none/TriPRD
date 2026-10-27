# Completion v6 四组控制实验

这份源码建立一套新的 A/B/C/D 对照，回答源数据、普通蒸馏和真实额外模态分别带来什么。学生结构沿用 v6 对象条件补全器，测试只输入 RGB。原 v6 训练脚本、模型、损失和已有结果保持冻结；本实验使用新的 runner、教师接口和独立输出目录，不能把旧 v6 的 D 结果直接拼入本表，也不复用旧教师或源学生 checkpoint。

**本包交付的是实验实现，不是已经取得提升的分数。CPU/tiny 功能检查不能代替真实 DINOv3、CUDA、4090 显存和四卡并发验证。**

## 四组具体比较什么

| 组 | 训练路线 | 与前组比较的含义 |
|---|---|---|
| A | 通用 DINOv3 初始化 → COD | 同学生架构、同 COD 训练预算的直接监督基线 |
| B | RGB + GT 源训练 → COD | B−A：增加 VDT RGB/GT 源训练的作用 |
| C | RGB 教师 → RGB 源学生 → COD | C−B：在相同源阶段加入教师输出蒸馏的作用 |
| D | RGB/T/D 教师 → RGB 源学生 → COD | D−C：真实额外模态在本路线中的作用 |

C 与 D 的教师结构、参数容量和编码器调用次数匹配。C 把同一 RGB 输入三个槽并实际执行三次共享编码器；D 的槽输入为真实 RGB、热红外和深度。C 不读取 T/D 值作为教师输入，也不把 RGB 误用为需传感器归一化的辅助数据。每组内部，冻结教师接收该组学生实际生成的同一 RGB cue；C、D 学生训练后各自的 cue 数值不保证相同。两组源学生均有 GT 监督和同定义的输出 KD。

这四组隔离的是训练路线中的增益。A 仍然含 v6 的细节分支、定位器和补全器，因此不是“裸 DINOv3”或标准线性探针。单种子四组结果也不能代替重复种子、机制消融和统计稳定性分析。

## 服务器准备与启动

将新源码包解压到新目录，例如 `/data2/yyy/TriPRD_completion_controls`，或在干净 clone 中使用本套文件。打包和训练都不依赖历史 v4/v5/v6 ZIP。沿用已有可用的 CUDA/PyTorch 环境；如需要补齐 Python 依赖，可在该环境中安装 `requirements-completion-v6.txt`，先确认不会替换原有匹配 CUDA 的 PyTorch。

```bash
cd /data2/yyy/TriPRD_completion_controls
conda activate triprd-v3
export OUTPUT_ROOT="$PWD/results/completion_controls_v1_seed-20261012"
export GPUS=0,1,2,3
mkdir -p "$OUTPUT_ROOT"
nohup bash tools/run_triprd_completion_controls.sh all \
  > "$OUTPUT_ROOT/queue.log" 2>&1 &
```

`GPUS` 按 A、B、C、D 顺序指定四张不同的物理 GPU。默认四组并行，每组独占一张卡；每组内部的教师、源学生、COD、测试按依赖顺序运行。它不是把一个模型分到四张卡，也不会共享其他组的教师。C、D 的教师需要重新训练。

默认路径沿用旧服务器布局；不同环境请在启动前覆盖：

| 环境变量 | 默认值 |
|---|---|
| `SPLIT_MANIFEST` | `/data2/yyy/TriPRD_task_v4/results/seed-20261012/_data/split_manifest.json` |
| `DINO_REPO` | `/data2/yyy/TriPRD/dinov3` |
| `DINO_WEIGHTS` | `$DINO_REPO/facebook--dinov3-vitb16-pretrain-lvd1689m/snapshots/master/model.safetensors` |
| `VDT_ROOT` | `/data2/yyy/TriPRD/datasets/VDT-2048` |
| `COD_ROOT` | `/data2/yyy/TriPRD/datasets/COD_benchmark` |

`SPLIT_MANIFEST` 应使用原 `_data` 下含 VDT/COD 开发划分的共同文件，不要替换为某个 full 阶段重写的单域 manifest。A 仅训练 COD；B/C/D 的源阶段仅使用 VDT 源训练数据。DINO checkout、预训练权重、数据集和划分文件都不随小源码包分发。

## 固定设置与显存

| 项目 | 默认值 |
|---|---|
| 骨干与输入 | DINOv3 ViT-B/16，384 × 384 |
| 教师 / 源学生 / COD epochs | 100 / 20 / 100；A 跳过前两段，B 跳过教师 |
| 有效 batch | `BATCH_SIZE=8` |
| 单次前向 micro batch | `MICRO_BATCH_SIZE=2`；按有效 batch 累积梯度 |
| COD 仅定位器预热 | `LOCATOR_WARMUP_EPOCHS=0`，四组统一从第一轮联合适配 |
| 学习率预热 | 默认 5 epochs（`--warmup-epochs 5`），与定位器预热不同 |
| 主学习率 / 骨干倍率 | 1e-4 / 0.1 |
| 骨干训练范围 | 最后 4 个 Transformer blocks |
| seed / split seed | 20261012 / 20261012 |
| COD 训练模式 | `COD_TRAIN_MODE=full` |

micro batch 用于降低每张卡前向/反向时的峰值显存，8 张有效样本通常拆成 4 次 micro forward，再做一次参数更新。最后不足有效 batch 的数据按实际样本数归一化。梯度累积不保证和一次性大 batch 的所有随机运算逐位相同，需把 micro batch 作为实验协议的一部分记录。单卡 4090 是否足够仍需真实训练确认；若需改为 `MICRO_BATCH_SIZE=1`，应四组一致、使用新输出目录重新开始，不能在中断恢复时悄悄改训练配置。

新控制实验使用统一的定位器预热和学习率调度设置，旧 v6 的 3 轮定位器预热结果仅保留为历史结果。四组实验应在本套协议内相互比较。

## Dry run、进度与恢复

先只打印四组计划，不检查服务器资源，也不创建训练输出：

```bash
DRY_RUN=1 bash tools/run_triprd_completion_controls.sh all
```

正式启动后查看：

```bash
tail -F "$OUTPUT_ROOT/queue.log"
tail -F "$OUTPUT_ROOT/A/cod/run.log"
tail -F "$OUTPUT_ROOT/B/source/run.log"
tail -F "$OUTPUT_ROOT/C/teacher/run.log"
tail -F "$OUTPUT_ROOT/D/teacher/run.log"
```

尚未开始的阶段没有日志文件。各组依赖阶段完成后会继续进入 `source/run.log`、`cod/run.log` 和 `test.log`。根目录的 `completion_controls_queue.json` 保存运行计划，`queue_status.json` 保存进度。失败时先查看对应阶段日志与退出状态。

使用同一代码、路径、GPU 分配和超参数恢复：

```bash
RESUME=1 nohup bash tools/run_triprd_completion_controls.sh all \
  >> "$OUTPUT_ROOT/queue.log" 2>&1 &
```

恢复会让每个阶段由 runner 校验产物和协议，完成阶段通过校验后跳过，中断阶段从已有 epoch checkpoint 恢复；不会仅凭队列的 status 文本跳过训练。不要同时启动两个相同 `OUTPUT_ROOT` 的队列。若进程被强杀留下 `completion_controls_queue.lock` 或阶段目录中的 `task_run.lock`，先核对锁中的 PID 确认原进程已经结束，再处理失效锁。

也可以把训练和正式测试分开：

```bash
bash tools/run_triprd_completion_controls.sh train
bash tools/run_triprd_completion_controls.sh test
```

`all` 在 full 模式下自动完成四组正式测试并汇总；`train` 只训练，`test` 要求已有满足正式协议的四组 COD 产物。

## 正式结果和开发结果

正式 full 模式使用 COD10K3040 + CAMO1000 共 4040 张训练图，预先固定 100 epochs，使用最后一轮作为 COD checkpoint，不根据官方测试集挑选轮次。教师和源学生在共同 VDT 开发划分上选模。四个完整官方测试集是 COD10K2026、CAMO250、CHAMELEON76、NC4K4121；每组 RGB 单次前向，预测恢复原图尺寸，不使用 TTA、GT 提示或人工裁剪。

各组 `cod/` 下会生成 checkpoint、`comparison.json`、`history.json`、测试预测和 `benchmark.csv` / `benchmark.md`。Emean、Emax、Eadaptive 分列；不要把论文未注明口径的 E 指标直接与本表混比。汇总命令为：

```bash
python tools/summarize_triprd_completion_controls.py --root "$OUTPUT_ROOT"
```

正式汇总前需要四组完整结果和相容协议，不能用缺失组、部分测试图或不一致资源的结果代替。根目录输出 `controls_summary.json`、`controls_summary.csv`、`controls_summary.md`，包含各组指标、B−A、C−B、D−C、D−A 的描述性差值和训练成本。训练期间的损失下降不能直接当作三模态净增益。

开发迭代使用独立输出目录：

```bash
OUTPUT_ROOT="$PWD/results/completion_controls_dev" COD_TRAIN_MODE=split \
  bash tools/run_triprd_completion_controls.sh all
```

split 模式仅用训练集内部划分开发，不安排四个官方测试集。`all` 或 `train` 在四组训练完成后会生成 `scope=internal_validation` 的内部验证汇总；full 模式的正式汇总标记为 `scope=official_tests`。这两种分数不能直接相减。修改超参数或架构时先在开发协议中完成决定，再冻结正式训练设置。

## 从当前源码打包与校验

```bash
python tools/package_triprd_completion_controls.py
python tools/package_triprd_completion_controls.py \
  --verify deliverables/triprd_completion_controls_2026-10-27.zip
```

打包器直接收集当前模型、工具、评测、数据读取器和测试的源文件，以及依赖清单与必要文档；无需 Git，也不读取旧 ZIP。仅包含 `datasets/` 顶层 Python 读取代码，排除图像、数据压缩包、结果、checkpoint、权重、DINO checkout、虚拟环境与缓存。`MANIFEST_COMPLETION_CONTROLS.json` 记录每个 payload 文件的 SHA256 和字节数；manifest 自身不参与自哈希。ZIP 生成后会自动核对条目、CRC、文件 hash 和本地模块依赖。

本地功能测试可从以下测试开始：

```bash
python -m pytest tests/test_completion_controls.py \
  tests/test_completion_controls_queue.py tests/test_completion_controls_summary.py \
  tests/test_completion_controls_package.py -q
```

功能检查覆盖的是实现和协议，不会生成论文分数；正式训练资源、总训练成本和额外 VDT/T/D 监督需要随结果一并报告。
