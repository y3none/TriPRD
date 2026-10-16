# TriPRD 显式关系迁移实验

更新：2026-10-16。本页对应新入口 `run_triprd_relation.py` / `run_triprd_relation_queue.py`，与旧 `run_triprd_ab.py` 的 A/B/C/D 定义不同。首个 seed 的四组 100 epoch GPU 筛选已完成并归档为 E011–E014；多模态 D 未超过 A/C 的主指标 Sm，尚无 SOTA 或稳定收益证据。完整结果与证据边界见 [关系实验汇总](../experiments/relation_summary.md)；E001–E010 见 [上一版汇总](../experiments/optimization_summary.md)，研究边界见 [Paper Scope](../paper_scope.md)。

## 本轮结果与下一步

以下均为 `seed=split_seed=20261012`、`cod-train-mode=split` 的内部验证结果：COD 训练 3,232 张，验证 COD10K 608 张 + CAMO 200 张；两来源等权，四组均在第 70 epoch 取得最佳 Sm。每组完成 100 COD epoch，不等于使用全部 4,040 张训练图；四个测试集尚未评估。

| 记录 / 组别 | Sm ↑ | Fw ↑ | mean E ↑ | MAE ↓ |
|---|---:|---:|---:|---:|
| E011 / A_scratch | 0.808668 | 0.703867 | 0.868845 | 0.048991 |
| E012 / B_rgb_source | 0.807764 | 0.701407 | 0.869505 | 0.049096 |
| E013 / C_rgb_relation | 0.807881 | 0.701479 | 0.870593 | 0.048955 |
| E014 / D_multimodal_relation | 0.807632 | 0.703131 | 0.869142 | 0.048766 |

D−A 的 Sm 为 −0.001036，D−C 为 −0.000249；D 有小幅 MAE 优势，但没有赢得事先固定的主指标。这里只有一个训练 seed，不能将小差异解释为稳定提升或稳定退步。多模态教师的 VDT Dice 为 0.904641，高于 RGB 教师的 0.855075，但 D 源学生的 0.827633 低于 C 的 0.829510 和 B 的 0.830162。教师关系归一化熵 C/D 为 0.995166/0.990839，源学生最佳轮为 0.998245/0.996127；平均接近均匀提示需诊断，不证明所有位置的关系均无效。

优先用已有教师权重，在相同 VDT 验证样本上对比原关系、均匀关系、打乱关系和关闭关系残差，检查分割结果是否依赖该关系。当前教师其他尺度的均值融合和 T/D value 可能提供未进入关系蒸馏目标的信息；这是代码支持的待验证解释，不是已证实原因。若干预影响很小，再检验“T/D 仅影响关系权重，value 和解码输入保持 RGB”的设计。**这些干预脚本与新信息路径均未实现、未运行**；本次只归档当前版本，不默认扩大现有配方的多种子或全训练池运行。关系依赖诊断也不能替代固定错配 T/D 后重训的因果对照。

本地已同步四份 comparison；实际权重、划分成员和队列日志尚未独立核验。没有同步独立新关系 GPU smoke 结果，不能据此补写 smoke 已运行。下方 GPU 统一用 `0,1,2,3` 作可配置示例；这不是从 comparison 确认的服务器物理卡号。

## 本次实现的机制

教师在低分辨率特征上预测 RGB→Thermal、RGB→Depth 的局部关系分布，关系直接参与加权值聚合，再残差注入解码器。投影、关系及解码器共同接受源任务分割监督；其余教师多尺度特征仍使用均值融合。学生从 RGB 预测两路同形状关系，并在自己的分割前向中使用它们。默认使用 ResNet-50 stride-16 尺度、3×3 邻域（含中心）、32 维投影；图像边缘的无效邻居被屏蔽，避免全分辨率两两关系矩阵的内存开销。

VDT 学生阶段冻结 ImageNet 主干参数和 BatchNorm 统计，只训练关系分支及源任务解码头。B/C/D 到 COD 时只迁移完整关系分支的 query/key/value/output 等参数；COD 主干和解码头按同 seed 重新初始化，与 A 保持一致。随后全部组采用相同 COD 微调配方。C/D 额外使用 detached 教师局部关系分布的 KL；本轮没有新的边界可靠性、uncertainty、logit KD 或 COD prior loss。

| 新组 | VDT 学生训练 | 教师 | 迁移到 COD | 主要对照目的 |
|---|---|---|---|---|
| A_scratch | 无 | 无 | 无；同结构关系分支保留随机初始化 | 相同学生结构的目标任务基线 |
| B_rgb_source | RGB 分割监督，主干冻结 | 无 | 仅关系分支 | 控制额外源数据与学生训练预算 |
| C_rgb_relation | 与 B 相同的监督和预算，加关系 KL | 三个独立编码器均输入 RGB | 仅关系分支 | 控制教师容量与通用 KD |
| D_multimodal_relation | 与 C 相同 | 三个独立编码器输入 RGB、真实 T、真实 D | 仅关系分支 | 检验真实特权模态的增益 |

C/D 教师结构和参数数量相同：单通道 T/D 复制到三通道后进入同规格编码器。C/D 分别从头训练自身教师，不能复用旧均值融合教师；B/C/D 源学生的数据清单、初始化、步数、监督和目标训练配方匹配。A 不使用 VDT，源训练预算差异通过 B 对照解释。教师额外训练预算单独报告。

## 四卡 smoke（复现或改动后的验收入口）

在服务器仓库根目录执行；4 张 4090 分别运行独立任务，不是 DDP。先省略 `--execute` 检查完整计划，dry-run 不写目录、不训练：

```bash
python tools/run_triprd_relation_queue.py --gpus 0,1,2,3 \
  --epochs 2 --vdt-epochs 2 --max-samples 32 --val-metrics-every 1 \
  --output-root results/optimization/relation_smoke
```

确认计划后实际运行：

```bash
python tools/run_triprd_relation_queue.py --gpus 0,1,2,3 \
  --epochs 2 --vdt-epochs 2 --max-samples 32 --val-metrics-every 1 \
  --output-root results/optimization/relation_smoke --execute
```

本配置自动把 warmup 缩为 1 epoch。仍使用 ImageNet ResNet-50、352 输入、base channels 256、batch 8；首次运行可能加载/下载 torchvision 预训练权重。输出位置为 `results/optimization/relation_smoke/relation/seed-20261012/` 下的四个组目录。C/D 比 A 多训练教师和源学生，完成时间不同属于正常现象。

smoke 验收：四组退出码均为 0；各组 `comparison.json` 完整且 Sm/Fw/mean E/MAE 有限；迁移报告通过主干/BN、独立 COD 头和仅关系分支检查；C/D 的关系 KL 与裁剪前总梯度范数有限；记录时间及峰值显存。日志中的梯度范数针对模型总梯度，关系分支逐参数梯度由单元测试检查。另查看 `relation_entropy` / `teacher_relation_entropy`：它们是有效邻域归一化熵，接近 1 表示分布接近均匀；KL 很小而教师熵接近 1，不能自动判定学到了有用关系。两 epoch 小样本的组间排名不用于选方法。CPU tiny 只用于开发检查，不能代替此 GPU 验收。

输出目录已存在时队列会拒绝覆盖。失败后保留诊断材料，换一个 `--output-root` 重试，避免把新旧结果混到同一目录。

## 第一轮完整匹配筛选（已完成，历史复现）

以下保留本轮配置和复现入口，已有结果目录不能直接重跑覆盖；需要复现时另设 `--output-root`。默认是 100 COD epoch、100 VDT epoch、batch 8、encoder LR scale 0.1、cosine、5 epoch warmup，学习率/weight decay 均为 1e-4。关系 KL 权重和温度均为 1。使用固定 `split_seed=20261012`，每 5 epoch 做原尺寸、来源等权的正式验证，以宏 Sm 选 checkpoint；所有组默认跳过测试。

VDT 教师和源学生按 VDT validation Dice 选 checkpoint，使用相同的训练总预算与验证机会；实际选中 epoch 可能不同，需保留并报告。这里的 Sm 选模用于 COD 阶段，不将 VDT Dice 与 COD 的正式 Sm 混为同一指标。

```bash
python tools/run_triprd_relation_queue.py --gpus 0,1,2,3 \
  --seed 20261012 --split-seed 20261012 --execute
```

正式结果位于 `results/optimization/relation/seed-20261012/` 下的四组目录。来源分层的 COD 内部划分为训练 3,232 张、验证 COD10K 608 张 + CAMO 200 张（以实际数据和输出清单核验）。首次训练按固定 seed 生成清单；后续复验使用包含 **VDT 和 COD 全部 split** 的新 runner 清单。不要直接传入上一版 COD-only 筛选清单；新关系训练必须核对 VDT 源训练/验证成员。不要使用 smoke 的小样本清单进行正式训练。

先比较 D−C（真实 T/D 的额外价值）、D−B（关系 KD 的价值）、D−A（相同部署学生的总收益），同时报告四指标与两来源，不只挑有利的宏 Sm。由于各组最终 COD 模型相同，可报告同一推理结构的参数/FLOPs/延迟，但必须额外披露教师和源训练成本。

本轮首 seed 未支持净增益，暂不直接执行以下扩展。后续机制筛选支持收益并冻结配置后，才做配对复验；例如复用首轮 D 的完整清单：

```bash
python tools/run_triprd_relation_queue.py --gpus 0,1,2,3 \
  --seed 20261013 --split-seed 20261012 \
  --split-manifest results/optimization/relation/seed-20261012/D_multimodal_relation/split_manifest.json \
  --execute

python tools/run_triprd_relation_queue.py --gpus 0,1,2,3 \
  --seed 20261014 --split-seed 20261012 \
  --split-manifest results/optimization/relation/seed-20261012/D_multimodal_relation/split_manifest.json \
  --execute
```

两条命令顺序运行，每次最多占用四张指定 GPU。每个 seed 默认重新训练 C/D 教师和 B/C/D 源学生，评估全流程随机性；训练 seed 改变模型与采样随机性，split seed 和清单固定划分。不要同时启动两套占用同四卡的队列。

## 冻结配置后全量训练与最终测试（入口已实现，真实运行未完成）

`--cod-train-mode full` 要求完整 3,040 张 COD10K train + 1,000 张 CAMO train，且 `--max-samples 0`；不使用 COD validation、不按测试集选权重，固定采用事先约定的最后 epoch。必须在开始前冻结时长和 cosine horizon，而不是运行后从不同测试结果挑 epoch。下面是保留 100 epoch 预算的示例，实际预算应在验证筛选后明确记录：

```bash
python tools/run_triprd_relation_queue.py --gpus 0,1,2,3 \
  --seed 20261012 --split-seed 20261012 --epochs 100 --vdt-epochs 100 \
  --cod-train-mode full --output-root results/optimization/relation_full \
  --execute
```

全量 COD 权重为各组输出目录下 `<GROUP>/cod/last.pt`。默认仍跳过测试。配置及选模规则冻结后，单独测试指定权重；例如 D：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/run_triprd_relation.py \
  --groups D --model-variant resnet50 --pretrained \
  --image-size 352 --base-channels 256 --relation-dim 32 --relation-kernel 3 \
  --train-seed 20261012 --split-seed 20261012 --cod-train-mode full \
  --eval-only --no-skip-test --device cuda \
  --cod-checkpoint D=results/optimization/relation_full/relation/seed-20261012/D_multimodal_relation/D/cod/last.pt \
  --out results/optimization/relation_final_test/D_seed-20261012
```

同样独立评估 A/B/C 及其它已冻结 seed。正式报告保持 `--max-test-samples 0`：runner 检查所选测试集的图像/GT、预测 ID 与数量，预期 COD10K 2,026、CAMO 250、NC4K 4,121、CHAMELEON 76。部分样本测试必须标为调试，不能替代完整测试。E001 已查看过公开测试结果，不能把后续确认实验称为完全盲测。

## 如何判断运行状态与保留证据

```bash
nvidia-smi
tail -f results/optimization/relation/seed-20261012/D_multimodal_relation/run.log
cat results/optimization/relation/seed-20261012/exit_status.json
```

每组的 `run.log`、训练目录 `metrics.jsonl`、checkpoint 及 `comparison.json` 用于跟踪；队列最终 `exit_status.json` 汇总成功/失败。`finished with exit status 0` 表示该任务正常结束；四组均完成且输出齐全才表示整轮结束。每张卡一次只启动一个进程，任何失败停止后续队列派发，已经运行的任务继续收尾。

回传分析至少保留四份 `comparison.json`、四份完整 `split_manifest.json`、队列 `launch.json`/`exit_status.json`；异常附 `run.log`。原始 JSON、日志和权重留在 `results/optimization/`，不提交 Git。提交小型实验 Markdown 记录时引用其路径、SHA、代码版本和完整命令；本轮已使用 E011–E014，后续新实验编号从 E015 起，不能复用历史编号或将计划登记为已完成结果。

如果 D 不胜 C 或 B，不叠加更多损失掩盖无效关系；先查源关系预测、跨任务保持和错误配对。净增益成立后再实现 GT-only、旧 Gram KD、打乱目标、固定错配重训和关系只做正则等消融。当前这些机制消融以及第二主干仍未完成；是否达到 SOTA 和 CVPR 所需证据均需后续实验支持。

## 本地验收范围

2026-10-16 在 Windows、Python 3.11、PyTorch 2.8 CPU 环境运行全套测试：**120 passed、198 subtests passed，9.04 秒**。新四组 tiny 模型真实执行两 epoch 训练、权重复用、原尺寸 Sm 选模、eval-only 和全量训练不使用 validation 的集成链路；测试也核验冻结主干与 BN、仅关系分支迁移、独立 COD 初始化、关系前向/反向、邻域屏蔽和队列命令。

另用真实 ResNet-50 结构、base channels 256、relation dim 32、batch 2、64×64 随机输入完成 CPU 检查，未下载 ImageNet 权重：教师监督、冻结主干的学生关系 KD、解冻后的 COD 微调各执行一次 AdamW 参数更新，损失及梯度均有限。参数量为教师 **75,154,626**、学生 **28,138,562**，其中可迁移关系分支 **104,448**。这验证实现与优化链路，不替代 352×352 真实数据 GPU smoke，也不提供运行速度、显存需求或性能收益结论。
