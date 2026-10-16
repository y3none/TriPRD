# E001 后的优化与四卡实验

更新：2026-10-16。设备：4 张 RTX 4090。真实 GPU smoke、第一波 COD 配方筛选和第二波 encoder-only 迁移/第二种子 A 复验均已完成。A 的组合配方在两个训练种子上提高了来源等权 Sm，但第二种子 CAMO 的 Sm/MAE 退步；encoder-only 未消除 B/C 相对 A 的负迁移。尚无本轮新四测试集成绩，也未完成全流程三种子确认。

本文归档已实现的上一版优化入口、已完成结果和历史复现命令。结果及证据边界见 [优化实验汇总](../experiments/optimization_summary.md)；独立显式关系入口已完成首 seed 的四组 100 epoch GPU 内部验证筛选，D 的 Sm 未胜 A/C，详见 [关系实验汇总](../experiments/relation_summary.md) 和 [关系实验说明](TRIPRD_RELATION.md)。研究边界见 [Paper Scope](../paper_scope.md) 和 [CVPR 2027 计划](../cvpr2027_submission_plan.md)。当前 `--phase distill` 是上一版 KD 消融队列，已实现但未执行，不代表新的关系机制。

## 1. 本轮改动与依据

E001 中 D 的 Fw 和 mean E 在四个测试集均低于 A；边界 KD 的加权标量占总损失首轮约 78.95%，gate_mean 约 0.99942。这些事实提示需要隔离训练配方和蒸馏分项，不能直接认定是某一项梯度造成了下降。

| 改动 | 新入口与含义 |
|---|---|
| 正式验证选模 | `--selection-metric S_alpha` 自动启用原 GT 尺寸验证，报告 Sm/Fw/mean E/MAE。先各来源均值，再对 COD10K-val、CAMO-val 等权平均；避免大数据集掩盖小数据集退步 |
| 验证时点 | `--val-metrics-every 5`，另外强制验证最后一轮。只在这些时点竞争 Sm 最佳 checkpoint；快速 Dice 每轮记录 |
| 可复现划分 | `--split-seed` 固定划分，`--train-seed` 控制初始化、打乱和增强；`--split-manifest` 可复用精确清单 |
| COD 优化器 | `--cod-encoder-lr-scale 0.1 --cod-scheduler cosine --cod-warmup-epochs 5 --cod-min-lr-ratio 0.05`；解码器基础 lr 为 1e-4，编码器为 1e-5，先 warmup 再衰减 |
| 权重复用 | `--teacher-checkpoint`、`--vdt-checkpoint B=...`；检查 stage、结构、输入尺寸和 RGB 归一化，记录 hash 和来源元数据 |
| 迁移范围 | `--transfer-scope full` 或 `encoder`；后者重新初始化 decoder/head，使用与 A 同 seed 的初始化 |
| 边界 KD | `--boundary-kd-type prob_mse` 比较 sigmoid 概率，候选权重 `--lambda-boundary-kd 0.1`；保留原 logits MSE 对照 |
| 关系 KD | `--relation-sampling uniform` 使用互斥的内部/内边界/外边界/外围背景区域，均匀取不重复点，空区域不补全图，统一 Gram 包含跨区域关系 |
| 分项消融 | `--lambda-logit`、`--lambda-relation`、`--lambda-boundary-kd` 可设为 0，真正跳过对应计算；`--gate-mode none` 关闭门控 |
| 诊断与失败 | 记录加权 KD 分项、监督项、裁剪前总梯度范数、gate 分布、实际 lr、耗时、峰值 CUDA allocated 显存；非有限 loss/梯度立即失败 |
| 筛选和测试 | `--skip-test` 不加载测试集；`--eval-only --cod-checkpoint A=...` 仅评估冻结权重；正式评估失败保留错误并返回非零退出码 |

旧默认仍是 Dice 选模、固定 lr、legacy 关系、logits 边界 MSE 和原门控，便于复现。新选项只在显式开启或优化队列中生效。VDT 教师和 VDT 学生继续按 VDT validation Dice 选模；本轮正式 Sm 选模用于 COD 阶段。gate 的 std/分位数是每批最多 4096 个均匀采样点的统计，再按批样本数汇总，不是整轮精确分位数。梯度范数是总梯度，不代表各 loss 的梯度占比。

`uniform` 同时改变区域定义和关系矩阵，需要 legacy 对照，不能把它仅称为采样提速。当前三分支教师仍为特征均值融合；本轮没有实现新的关系教师、COD prior 或学习式不确定性模块。

## 2. 服务器同步与验收

同步本次代码、测试和文档，尤其 `models/triprd.py`、`evaluation/validation.py`、`tools/run_triprd_ab.py`、`tools/triprd_training.py`、`tools/run_triprd_optimization.py`、`tools/evaluate_cod.py`、`tools/record_experiment.py`。已有 `models/triprd_full.py` 和 `evaluation/` 指标实现继续使用。无需上传本地 `.venv`；服务器保留原 CUDA PyTorch 环境并确认 scipy/OpenCV/Pillow 可用。

队列依赖服务器旧实验文件：

```text
results/triprd_full/teacher/best.pt       # KD 队列
results/triprd_full/B/vdt/best.pt         # COD 队列
results/triprd_full/C/vdt/best.pt         # COD 队列
results/triprd_full/comparison.json      # 旧 checkpoint 预处理协议的补充来源
results/triprd_full/split_manifest.json  # 核验旧权重见过哪些样本
```

新 checkpoint 内嵌配置；E001 旧 checkpoint 没有协议时会查找相邻上级 `comparison.json`。不匹配直接失败；缺少元数据给出 warning 并记录未核验。**这些检查不等于已核验旧划分。** 恢复不了旧 split manifest 时，旧权重只用于工程筛选，确认实验重新训练。更改 KD loss 后必须重新训练 VDT 学生；复用旧 VDT 权重只会比较 COD 微调，不能验证新 KD。

先在 GPU 2 做真实数据 smoke（以下为 Bash；PowerShell 将换行符 `\` 换为反引号，并用 `$env:CUDA_VISIBLE_DEVICES="2"` 设置 GPU）：

```bash
CUDA_VISIBLE_DEVICES=2 python tools/run_triprd_ab.py \
  --groups A,B,C,D --model-variant resnet50 --pretrained \
  --image-size 352 --base-channels 256 --batch-size 8 --device cuda \
  --epochs 2 --vdt-epochs 2 --max-samples 32 \
  --train-seed 20261012 --split-seed 20261012 \
  --selection-metric S_alpha --val-metrics-every 1 \
  --cod-encoder-lr-scale 0.1 --cod-scheduler cosine --cod-warmup-epochs 1 \
  --relation-sampling uniform --boundary-kd-type prob_mse \
  --lambda-boundary-kd 0.1 --gate-mode none --skip-test \
  --out results/optimization/gpu_smoke
```

默认数据根为 `datasets/VDT-2048` 和 `datasets/COD_benchmark`；必要时给所有命令传 `--vdt-root` / `--cod-root`。首次 ImageNet 初始化需要缓存或下载 torchvision 权重。确认输出中的最佳 Sm、分来源样本数、lr、loss 和显存后，再跑 10 epoch 全量测速；smoke 数据和轮数不能用于论文。若 batch 8 OOM，所有可比任务一致减为 4，并记录预算变化。当前 ZIP 数据读取强制 `num_workers=0`，不能直接增加 workers；多卡 CPU/IO 竞争要实测。

## 3. 第一波：四卡 COD 配方筛选（已完成）

以下保留当时使用的命令，供复现和追溯；已有输出目录不可直接重跑覆盖。

先预览，不创建输出或启动训练：

```bash
python tools/run_triprd_optimization.py --phase cod --gpus 2,3,6,7
```

确认真实路径后执行：

```bash
python tools/run_triprd_optimization.py --phase cod --gpus 2,3,6,7 --execute
```

| GPU | 任务 | COD 配方 | VDT 来源 |
|---|---|---|---|
| 2 | A_control | encoder scale 1、无调度 | 无 |
| 3 | A_schedule | encoder scale 0.1、cosine + warmup | 无 |
| 6 | B_schedule | 同 A_schedule | 旧 B 的 RGB 监督预训练 |
| 7 | C_schedule | 同 A_schedule | 旧 C 的 logits KD 预训练 |

所有任务均为 352、ResNet-50、batch 8、100 COD epochs、同一 split/train seed、每 5 epoch Sm 选模、`--skip-test`。输出位于 `results/optimization/cod/seed-20261012/<任务名>/`。每卡独立进程，不是 DDP；指定较少 GPU 时按卡排队。已有输出目录会拒绝覆盖，重跑用新的 `--output-root`。队列保存完整命令、run.log 和 exit_status；遇到失败停止派发新任务，等待已运行任务结束。

比较所选 epoch 的来源等权 Sm，并同时看两来源 Sm/Fw/MAE，不能使用某个来源的单独最佳 epoch。A_control 已采用新选模协议，不能与 E001 直接归因比较。A_schedule 同时改变 lr 和调度，若要归因需再拆分。不要默认新配方胜出。

本轮 `seed-20261012` 已完成。第二波已沿用其中 B/C full-transfer 结果作对照，没有重复训练这两个相同配置。

## 4. 第二波：迁移范围与第二种子复验（已完成）

已核验四个 `results/optimization/cod/seed-20261012/<任务>/comparison.json`：每组 100 轮、每 5 轮正式验证；训练 3232 张，验证 COD10K 608 张、CAMO 200 张。四组划分文件哈希和核心源码哈希一致。当前本地只有 comparison，未独立读取服务器权重或划分成员。

| 已完成任务 | Sm 选中轮次 | 来源等权 Sm | Fw | mean E | MAE |
|---|---:|---:|---:|---:|---:|
| A_control | 10 | 0.798263 | 0.677055 | 0.848436 | 0.054341 |
| A_schedule | 95 | 0.807636 | 0.704436 | 0.868485 | 0.047933 |
| B_schedule | 40 | 0.781063 | 0.648989 | 0.836460 | 0.055411 |
| C_schedule | 60 | 0.776925 | 0.649248 | 0.837467 | 0.056724 |

A 的组合配方在第一波 seed12 上使两来源四项指标均改善；B/C 全模型迁移在同配方下仍落后。第二波据此检验 decoder/head 的迁移是否妨碍适应，并配对复验 A。以下为已完成第二波的历史命令，不表示当前应继续派发同一队列。

预览入口（不创建输出、不启动训练）：

```bash
python tools/run_triprd_optimization.py --phase transfer --gpus 2,3,6,7 --seed 20261012 --repeat-seed 20261013 --split-seed 20261012 --split-manifest results/optimization/cod/seed-20261012/A_control/split_manifest.json
```

执行同一队列：

```bash
python tools/run_triprd_optimization.py --phase transfer --gpus 2,3,6,7 --seed 20261012 --repeat-seed 20261013 --split-seed 20261012 --split-manifest results/optimization/cod/seed-20261012/A_control/split_manifest.json --execute
```

| GPU | 已完成任务 | train seed | 唯一比较变量 / 配对参照 |
|---|---|---:|---|
| 2 | B_encoder | 20261012 | 对比上一波 B_schedule，只把 full 改为 encoder 迁移 |
| 3 | C_encoder | 20261012 | 对比上一波 C_schedule，只把 full 改为 encoder 迁移 |
| 6 | A_control_repeat | 20261013 | 第二种子旧配方，供同种子的 A_schedule_repeat 配对 |
| 7 | A_schedule_repeat | 20261013 | 第二种子新配方；与上一行计算种子内差值 |

共同协议：ResNet-50/ImageNet、352、batch 8、100 COD epochs、全量样本、每 5 轮来源等权 Sm 选模、`--skip-test`、split seed 固定 20261012。B/C 和 A_schedule_repeat 保留 encoder lr scale=0.1、cosine、5 轮 warmup；A_control_repeat 保留 scale=1、无调度。B/C 只载入 VDT encoder，decoder/head 使用与 seed12 的 A 相同初始化；不重新训练教师或 VDT 学生。

默认复用服务器上 `results/triprd_full/B/vdt/best.pt`（epoch 61，SHA256 `ced9ce5aa36238da94f7675ecfaf75a916a7476d71be7bb00ed3e43d0f694b14`）及 `C/vdt/best.pt`（epoch 93，SHA256 `476b3ab6ca9244850485278fd34acaabedc522c0d18c10a631428d07136395a2`）。保持这两份文件与上一波一致；不要换成 smoke 或 COD 微调权重。脚本会检查文件存在、阶段和预处理协议，并在结果中记录实际 hash。

`--split-manifest` 指向服务器上一波完整 COD 清单；如果 A_control 清单不在，可使用该波其他任务中成员相同的清单。不能使用 smoke 的子集清单。dry-run 允许路径尚未存在，`--execute` 会在创建输出和启动任务前检查必需文件。runner 会进一步检查清单 seed、成员完整性和 train/val 无交集。新 manifest 含不同 train seed 和来源路径，因此整体文件 hash 可能变化；应核对 `cod_train` / `cod_val` 成员及顺序是否相同。

输出为 `results/optimization/transfer/seed-20261012-repeat-20261013/<任务名>/`，独立于上一波目录。每项保存 run.log、launch.json、comparison.json、split_manifest.json 和阶段权重；已有输出拒绝覆盖。可选 `--repeat-seed` 默认是 `--seed + 1`，必须与主 seed 不同，仅适用于 transfer。本轮四个 comparison 记录均为 100 COD epochs、`complete`、`test_status=skipped`，总历时约 4 小时 24 分钟；该耗时不代表下一版模型的训练预算。

预先固定分析顺序：

1. B_encoder 对比 B_schedule，C_encoder 对比 C_schedule：检验迁移范围；不要跨 seed 比较。
2. 两个 encoder-only 结果分别对比 seed12 的 A_schedule（Sm 0.807636）。只比 full-transfer 提高、仍落后 A，说明损害减轻，不能称为方法优于基线。
3. 分别计算 seed12 和 seed13 的 `A_schedule − A_control`，同时检查两来源 Sm/Fw/mean E/MAE；两个 seed 方向一致仅是初步复验，论文确认仍需第三个 seed。
4. 若 encoder-only 仍明显落后，可单独改变 encoder LR 倍率，保持 cosine/warmup 不变，并补同配方 A。这是旧实现的迁移诊断，不替代新机制的独立验证。

第二波所选 checkpoint 的结果如下；四项指标均来自同一个 Sm 选中轮次：

| 已完成任务 | train seed | Sm 选中轮次 | 来源等权 Sm | Fw | mean E | MAE |
|---|---:|---:|---:|---:|---:|---:|
| A_control_repeat | 20261013 | 10 | 0.807635 | 0.687991 | 0.861264 | 0.051713 |
| A_schedule_repeat | 20261013 | 35 | 0.810328 | 0.694426 | 0.873200 | 0.050660 |
| B_encoder | 20261012 | 95 | 0.780768 | 0.656656 | 0.836831 | 0.053569 |
| C_encoder | 20261012 | 85 | 0.776117 | 0.652425 | 0.839120 | 0.055049 |

两波共同支持的结论：

- A_schedule 相对同种子 A_control 的 Sm 提升为 seed12 的 `+0.009373` 和 seed13 的 `+0.002693`，平均配对提升 `+0.006033`。只有两个训练种子，不能据此声称统计显著或普遍稳定。第二种子中 COD10K 四项改善，但 CAMO 的 Sm `-0.002951`、MAE `+0.001992`。
- B_encoder/C_encoder 相对同种子 full-transfer 的 Sm 分别为 `-0.000296` / `-0.000808`；相对同种子 A_schedule 分别落后 `0.026868` / `0.031518`。Fw/MAE 有所改善，但重置 decoder/head 没有修复核心 Sm 差距。
- B/C 仍复用固定的旧 VDT 权重，只检验 COD 迁移，不是完整教师—学生流程的多种子复验，也没有检验新 KD 或下一版显式关系机制。

已同步的证据是两波八份 comparison；配置、样本数、源码 hash 和 B/C 来源权重 hash 可追溯。本地未独立读取实际权重或 split manifest 的成员。第二波 manifest 包含 train seed/来源路径等元数据，文件 hash 与第一波不同不能单独认定划分改变；需补同步清单后核对成员及顺序。此波没有评估测试集。

## 5. 已实现但未执行：上一版 VDT 学生 KD 消融

当前 `--phase distill` 仍是原均值融合教师及融合特征 Gram KD 的四项消融。下面保留可运行入口，尚未提交该队列的真实完整实验结果；不要把第二波 B/C 的迁移结果算作它的验证。

```bash
python tools/run_triprd_optimization.py --phase distill --gpus 0,1,2,3
# 历史执行入口：仅用于明确选择该旧配方的消融，不是下一版主线队列。
python tools/run_triprd_optimization.py --phase distill --gpus 0,1,2,3 --execute
```

| GPU | 任务 | KD 设置 |
|---|---|---|
| 0 | C_logit | logits；relation=0、boundary=0、无门控 |
| 1 | D_relation | logits + uniform relation；boundary=0、无门控 |
| 2 | D_boundary | 上一行 + 概率边界 MSE，权重 0.1；无门控 |
| 3 | D_gate | 与上一行相同，仅打开 disagreement gate |

四组共用旧教师来源，各自重训 100 epoch VDT 学生，再跑 100 epoch COD；只有新学生重训才能检验本轮 KD 改动。门控无稳定收益则删除，不为保留模块硬调参。增加 legacy relation 对照，区分新增关系目标与旧实现；本旧入口没有容量匹配 RGB 教师、随机配对重训、常规 feature KD 等完整控制，不能用这四组替代。新关系入口的 RGB 教师对照见下节。

**队列不会自动读取上一波并选择配方。** 如果第一波保留 A_control、encoder-only 或其他设置，使用 dry-run 输出的四条 `command`，一致替换 COD 优化器和 `--transfer-scope` 参数后分别执行 `run_triprd_ab.py`，不要执行默认 distill 队列。保留替换后的完整命令。

### 新独立入口：显式关系机制（首 seed GPU 筛选已完成）

独立的 `run_triprd_relation.py` 已实现显式跨模态空间关系教师、单尺度 RGB 关系预测及残差聚合、VDT 阶段冻结学生主干和 BN 状态、仅保存/迁移关系分支，并分离源任务与 COD 分割头。新边界可靠性、uncertainty 和 COD prior 未加入。E011–E014 已完成 `seed=split_seed=20261012`、100 COD epoch 的内部验证筛选：A/B/C/D 的来源等权 Sm 为 0.808668/0.807764/0.807881/0.807632，均选中第 70 epoch。COD 使用 3,232 张训练、808 张验证，测试跳过；这不是全 4,040 张训练池的最终结果。详细指标和证据边界见 [关系实验汇总](../experiments/relation_summary.md)。

新 `run_triprd_relation_queue.py` 的四组为：同结构学生无 VDT、VDT RGB 监督、容量匹配 RGB 教师关系 KD、真实 RGB/T/D 教师关系 KD，历史复现命令见 [关系实验说明](TRIPRD_RELATION.md)。它与旧 `--phase distill` 分开，组名不能混用。多模态教师的源任务优势尚未转成学生 Sm 收益，当前优先级是关系干预诊断；干预脚本与 T/D 仅影响关系权重的新信息路径均未实现。全 4040 张 COD 的固定最后 epoch 模式和完整测试覆盖校验入口已实现，真实运行待完成；强基线复现仍待完成。旧实现的 LR 排查仅作为诊断保留，不预设它能解决跨任务负迁移。

## 6. 三 seed 确认、冻结评估与记录

固定 split seed `20261012`；train seeds 用 `20261012/20261013/20261014`。同一数据、样本限制、划分清单、预算和 COD 配方下，A/B/C/胜出方法均重复，不能只重复 ours。下面是**候选 D_boundary 胜出时**完整重训的单 seed 示例，不代表预先选定它：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/run_triprd_ab.py \
  --groups A,B,C,D --model-variant resnet50 --pretrained \
  --image-size 352 --base-channels 256 --batch-size 8 --device cuda \
  --epochs 100 --vdt-epochs 100 --max-samples 0 \
  --train-seed 20261012 --split-seed 20261012 \
  --selection-metric S_alpha --val-metrics-every 5 \
  --cod-encoder-lr-scale 0.1 --cod-scheduler cosine --cod-warmup-epochs 5 \
  --relation-sampling uniform --boundary-kd-type prob_mse \
  --lambda-logit 1 --lambda-relation 1 --lambda-boundary-kd 0.1 \
  --gate-mode none --skip-test \
  --out results/optimization/confirm/seed-20261012
```

GPU 1/2 分别替换 train seed 和输出目录，固定 split seed；GPU 3 留给因果控制和复算。更严格复用清单时，从一次完整全量运行保存 `split_manifest.json`，给后续命令传 `--split-manifest <该文件>`。COD-only 筛选的 manifest 不含 VDT 清单，不能直接用于完整 VDT 重训。三个 seed 都重训各自教师，才是全流程训练随机性；复用同一个教师只能报告条件于该教师的学生方差。每个 seed 内 C/D 自动共享同一教师。

冻结方法、权重及报告规则后，再统一测试。示例对第一波 A_schedule 已选好的权重做评估（换其他组时同时改 `--groups` 与映射键）：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/run_triprd_ab.py \
  --eval-only --groups A \
  --cod-checkpoint A=results/optimization/cod/seed-20261012/A_schedule/A/cod/best.pt \
  --model-variant resnet50 --pretrained --image-size 352 --base-channels 256 \
  --device cuda --out results/optimization/final_eval/A_seed-20261012
```

此时 `--pretrained` 表示保持来源 ImageNet 归一化；完整权重来自指定 checkpoint，不会再次下载 ImageNet 权重。四测试集正式指标在 `comparison.json` 的 `test` 字段；任何一集失败都会返回非零码。最终报告各数据集三 seed 均值、样本标准差和逐 seed 差值，不能按测试集挑不同 checkpoint。E001 已查看过公开测试，本轮按验证筛选不等于这些数据成为完全盲测。

每个任务完成后立即记录。以下是未来独立重跑完成后的登记模板，运行目录和 ID 必须对应实际新实验，不可指向已归档运行；E002 已用于 GPU smoke：

```bash
python tools/record_experiment.py \
  --id E015_A_control_rerun \
  --run-dir results/optimization/cod_rerun/seed-20261012/A_control \
  --change "固定训练配方，启用原尺寸来源等权 Sm 选模" \
  --hypothesis "建立新验证协议的 COD 控制组" \
  --tags validation,cod,control
```

记录器已支持 validation-only：显示实际选中 epoch 的 Sm/Fw/mean E/MAE、各来源结果及 test skipped。冻结测试另用独立 ID 登记并关联训练记录。代码、文档和小型汇总提交 Git；原始 comparison/log、权重和预测留服务器，以路径和 hash 追溯，继续不提交大型原始文件。

## 7. 本地验收范围

2026-10-16 归档前在 Windows CPU 环境复验，Python 3.11 / PyTorch 2.8.0+cpu / torchvision 0.23.0+cpu：`python -m pytest tests -q --tb=short` 结果为 **80 passed、115 subtests passed，7.25 秒**。覆盖旧默认损失兼容、分项关闭、关系区域、优化器及调度、checkpoint 协议核验、正式验证选模、四卡队列调度和实验记录；transfer 测试检查逐任务训练种子、迁移范围、清单传递、缺失文件启动前失败及无重复参数。tiny 模型的真实 A/B/C/D 两轮集成覆盖训练、权重复用、skip-test 和 eval-only。中文 Windows 路径下的真实 PNG 正式评估也已验证。本次复验没有启动新的 GPU 训练。

2026-10-15 另做 ResNet-50 教师/学生、256 解码器通道、batch 2、64×64 合成输入、无 ImageNet 下载的 CPU 前向/学生反向及一次参数更新：输出形状与有限梯度通过，参数量为教师 75,043,781、学生 28,034,114。三队列共 12 条生成命令和当时文档的 10 条 CLI 示例均通过参数解析检查。这些是当时的本地检查事实，不替代真实 GPU 实验，也不构成下一版关系机制的验证。已完成的服务器 COD/transfer 两波见第 3、4 节；旧 distill 队列和新机制的执行状态见第 5 节。
