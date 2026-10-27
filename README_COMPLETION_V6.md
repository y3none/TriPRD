# TriPRD Completion v6

这版实现“对象条件补全器迁移”：RGB 粗定位 → 软前景/背景条件 → 全图掩码补全 → 边界参与逐级细化。源阶段由真实 RGB/T/D 教师指导 RGB 学生；进入 COD 时保留整个补全器，只重新学习 COD 粗定位，再联合微调。测试只输入 RGB。

截至 2026-10-27，首轮 v6 已完成教师、源学生、COD 三阶段训练和四个官方测试集评测。当前主入口为本文件；v4/v5 保留作历史版本，其教师和学生 checkpoint 与本版不兼容。原来的数据、划分 manifest、通用 DINOv3 权重可以继续使用。

`TriPRD` 沿用为项目工作名。旧展开名“Tri-modal Privileged Relation Distillation”对应历史关系蒸馏路线；v6 实际蒸馏的是逐级 mask/边界输出，不能继续将关系矩阵蒸馏列为本版贡献。尚未确定新的论文题名，见 [method_name.md](method_name.md)。

## 已完成结果与证据边界

本轮使用 DINOv3 ViT-B/16、384 输入和全局 batch 8。教师/源学生使用相同的 VDT 838/210 训练验证划分，分别训练 100/20 轮并按源验证 GT loss 选第 29/16 轮。COD 使用全部 4040 张训练图像，预先固定训练 100 轮，四个测试集均使用第 100 轮的同一 checkpoint。

| 数据集 | N | Sm ↑ | Fw ↑ | Emean ↑ | MAE ↓ |
|---|---:|---:|---:|---:|---:|
| COD10K | 2026 | 0.895018 | 0.831094 | 0.952530 | 0.016974 |
| CAMO | 250 | 0.899703 | 0.871965 | 0.950531 | 0.032771 |
| NC4K | 4121 | 0.913445 | 0.882477 | 0.958109 | 0.022695 |
| CHAMELEON | 76 | 0.922565 | 0.882944 | 0.959777 | 0.018206 |

原始报告见 [benchmark.md](results/TriPRD_completion_v6/reports/cod/benchmark.md)，完整核验与分析见 [v6 结果分析](experiments/completion_v6_result_analysis_2026-10-27.md)。报告记录完整补全器的 340 个 tensor 已迁移，新 COD locator 未被覆盖。源输出 KD 收敛，说明源训练上的加权输出差异缩小，不能据此证明三模态知识在 COD 上带来独立增益。

本轮只有一条正式方法结果，尚无同架构、同训练协议的 v6 无源迁移或 RGB 教师正式对照。相对项目已有 E2Net 参考行，四集 Sm/Fw 均未超过；该参考不是本轮资源匹配重跑，不能用于判断 DINOv3 与 DINOv2 的因果差异，也不能证明本方法贡献。v4/v5 内部验证分数不能与这次官方测试分数直接相减。当前不能宣称三模态净收益、SOTA、域不变补全或论文创新已经验证。

本地已核对报告、代码和迁移来源记录；本地未收到训练 checkpoint 和预测 PNG，因此尚未独立重算真实权重摘要与测试指标。学生约 90.94M 参数、教师约 95.17M 参数；可以报告测试仅 RGB，不能仅据教师移除就宣称轻量模型或特定加速比。

## 服务器直接运行

把本版源码包解压到一个新目录，例如 `/data2/yyy/TriPRD_completion_v6`。继续使用已有 PyTorch CUDA 环境，依赖见 `requirements-completion-v6.txt`。

```bash
cd /data2/yyy/TriPRD_completion_v6
conda activate triprd-v3
export OUTPUT_ROOT="$PWD/results/completion_v6_seed-20261012"
export GPUS=2,3,0,1,4
mkdir -p "$OUTPUT_ROOT"
nohup bash tools/run_triprd_completion.sh all \
  > "$OUTPUT_ROOT/pipeline.log" 2>&1 &
```

`all` 依次运行一条方法路线：

1. `teacher`：VDT 训练新的三模态对象补全教师，默认 100 epochs。
2. `source`：VDT 训练 RGB 补全学生，默认 20 epochs；教师与学生使用同一 RGB 软提示。
3. `cod`：完整继承源学生补全器，用 COD10K3040 + CAMO1000 训练，默认 100 epochs；前 3 epochs 训练 COD 粗定位接口，之后联合微调。
4. `test`：完成固定训练后，单一 RGB 模型评测四个完整官方测试集并输出表格。

三个训练阶段有前后依赖，顺序执行。每个训练阶段支持把同一个模型的全局 batch 分配到五张可见 GPU；不是同时启动五组筛选实验。默认全局 batch 为 8，完整批次分配为 2/2/2/1/1，末尾不足五张卡的批次只使用所需设备。验证和正式测试使用第一张可见卡。跨卡复制与汇总有开销，不保证五倍加速。

多卡逻辑提供 CPU 路径与分配逻辑测试；本轮服务器报告记录已在 `2,3,0,1,4` 五张可见 GPU 上完成，教师/源学生/COD 墙钟训练时间分别约 121.23/30.58/323.07 分钟，测试另计。该记录不是单卡/多卡加速比实验，显存日志也仅记录主设备。启动示例显式使用本轮 GPU 列表，脚本的默认列表仍为 `2,3,0,1,6`。若主卡显存不足，可先用 `GPUS=2` 保持全局 batch=8；改变正式运行设置时使用新的 OUTPUT_ROOT，避免把不同训练配置拼成同一结果。

默认服务器资源位置：

| 环境变量 | 默认值 |
|---|---|
| `SPLIT_MANIFEST` | `/data2/yyy/TriPRD_task_v4/results/seed-20261012/_data/split_manifest.json` |
| `DINO_REPO` | `/data2/yyy/TriPRD/dinov3` |
| `DINO_WEIGHTS` | `$DINO_REPO/facebook--dinov3-vitb16-pretrain-lvd1689m/snapshots/master/model.safetensors` |
| `VDT_ROOT` | `/data2/yyy/TriPRD/datasets/VDT-2048` |
| `COD_ROOT` | `/data2/yyy/TriPRD/datasets/COD_benchmark` |

路径不同就在启动前 `export` 覆盖。`SPLIT_MANIFEST` 应当是原 `_data` 下同时包含 VDT/COD 开发划分的文件，不能用全训练阶段重写过的单域 manifest 代替。训练按阶段只读取所需域的数据，使用共同 manifest 不意味着源阶段读取 COD 图像。

## 进度、分步和恢复

```bash
tail -F "$OUTPUT_ROOT/pipeline.log"
tail -F "$OUTPUT_ROOT/teacher/run.log"
tail -F "$OUTPUT_ROOT/source/run.log"
tail -F "$OUTPUT_ROOT/cod/run.log"
```

各阶段目录在启动该阶段时创建。完成教师后，流水线自动进入源学生；源学生完成后自动进入 COD。`pipeline_status.json` 记录当前子进程 PID 和退出码。

也可以分开启动：

```bash
bash tools/run_triprd_completion.sh teacher
bash tools/run_triprd_completion.sh source
bash tools/run_triprd_completion.sh cod
bash tools/run_triprd_completion.sh test
```

恢复时使用相同路径、超参数、代码和 GPU 配置：

```bash
RESUME=1 nohup bash tools/run_triprd_completion.sh all \
  >> "$OUTPUT_ROOT/pipeline.log" 2>&1 &
```

完成的训练阶段会核对已有产物后跳过；中断阶段从已保存的 epoch 恢复模型、优化器、调度器、随机数和数据加载器状态。正常中断会释放锁；若进程被强杀，先检查锁内 PID 确认对应训练已经结束，再删除那个失效锁，不要在进程仍存活时启动第二份训练。

只查看将要执行的命令，不启动训练：

```bash
DRY_RUN=1 bash tools/run_triprd_completion.sh all
```

开发划分模式不会访问官方测试集：

```bash
OUTPUT_ROOT="$PWD/results/completion_v6_dev" COD_TRAIN_MODE=split \
  bash tools/run_triprd_completion.sh all
```

开发结果不能作为 full4040 论文主结果。默认启动方式直接执行一条正式训练路线，不安排额外排除法分组。

## 实际结构和损失

- RGB 输入经 DINOv3 ViT-B/16 语义编码与真实 stride4/8 RGB 细节分支；默认只训练 DINO 的末四个 block 与末层 norm，并非全骨干微调。
- `locators.sod` / `locators.cod` 输出自动粗定位；软掩码生成前景、背景描述，保留全图语义通路。
- 单个共享 `refiner` 在 stride16/8/4 更新完整掩码；前一级预测边界进入下一级掩码计算。
- 源提示只来自 RGB 预测，可施加缺块/误检/膨胀扰动；无 GT 提示、GT box 或测试时 oracle 信息。
- 三模态教师仅在源阶段工作。RGB/T/D 共用 DINO 编码器权重，T/D 各有独立适配器，逐尺度拼接后卷积融合到 RGB 路径。学生输出自己的实际 cue，冻结教师接收同一 cue；教师不能额外改写学生的定位提示。
- 所有阶段有 GT 分割和边界监督。源学生从通用预训练初始化，通过三级 mask/边界 KL 学习教师输出；按教师相对源 GT 的正确程度加权并以总像素数归约。不匹配完整融合隐藏特征或关系矩阵，不采用 RTD-R 优势归一化门控。
- COD 迁移保留源 RGB 特征接口、完整 refiner、mask/边界读出；只保留新初始化的 `locators.cod`。COD 优化器与 RNG 重新初始化。
- COD 自动生成的 cue 保留梯度，最终掩码损失能够反传到定位器；源蒸馏 cue 仍停止梯度。定位预热结束后，补全器接受 COD GT 更新，没有永久冻结的源结构分支或 COD 阶段的源数据混训。

## 可比评测

正式训练使用 COD10K3040 + CAMO1000 全部 4040 张，预先固定 100 epochs 并使用最后一个 epoch。正式训练阶段不使用 COD 内部验证选模，更不使用测试成绩选 checkpoint。

四测试集分别为 COD10K2026、CAMO250、CHAMELEON76、NC4K4121。导出原图分辨率 PNG，沿用已有冻结预测处理协议与 CamoFormer 指标实现；严格核对预测/GT 一一对应、完整样本数量，不接受部分覆盖。输入仅 RGB，固定一次模型前向，不使用 TTA、GT 提示或人工裁剪。

输出：

- `cod/best.pt`：正式 full 模式等于最后 epoch；开发 split 模式按内部 macro Sm 选模。
- `cod/comparison.json`：配置、代码/权重/划分摘要、迁移来源、训练曲线与完整测试结果。
- `cod/benchmark.md`、`cod/benchmark.csv`：逐数据集 Sm、Fw、Emean、Emax、Eadaptive、MAE。
- `cod/predictions/<dataset>/`：评测预测 PNG。

报告时披露 VDT 额外源数据、三模态教师、DINOv3 预训练资源、实际训练成本；不要把 Emean 与 Emax 混成同一列，不把内部宏平均与论文逐测试集指标直接相减。

单独重评已完成的正式模型：

```bash
CUDA_VISIBLE_DEVICES=2 python tools/evaluate_triprd_completion.py \
  --run "$OUTPUT_ROOT/cod" --device cuda \
  --cod-root "$COD_ROOT" --dino-repo "$DINO_REPO"
```

若父 shell 没有设置 COD_ROOT/DINO_REPO，先按上表 export，或继续使用 shell 入口 `bash tools/run_triprd_completion.sh test` 自动应用默认路径。

## 本地验证与源码包

```bash
python -m pytest tests/test_completion_model.py tests/test_completion_loss.py \
  tests/test_completion_runner.py tests/test_completion_protocol.py \
  tests/test_completion_parallel.py tests/test_completion_pipeline.py -q
python tools/package_triprd_completion.py
```

功能测试使用 tiny 编码器和合成数据验证真实前向、反向、优化、迁移、恢复和协议边界；不能代替 DINOv3 真实数据集训练，也不能证明三模态增益或 SOTA。
