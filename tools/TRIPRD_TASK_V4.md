# TriPRD v4：任务表征迁移与可比 COD 实验

2026-10-22。新方法标识为 `triprd_task_v4`。本版本重新建立 RGB COD 主体，并让教师知识监督学生实际用于解码的任务特征和 mask 输出。旧版 correction/stratified 结果、checkpoint 和源码保留；新学生从同一通用预训练与随机初始化开始，不加载旧 COD 学生或源探针权重。

当前验证涵盖本地 CPU tiny 模型、独立损失与梯度 oracle、训练/评测基础设施。它不构成真实 DINOv3/CUDA 训练成功或 COD 精度提高的证据，也不保证达到 SOTA。

## 1. 方法改动

- DINOv3 ViT-B/16 默认允许末 4 个 block 和最终归一化层训练，主干学习率为其余参数的 0.1 倍；设为 0 时严格冻结整个主干。
- 可选的 RGB 卷积支路直接产生 stride 4/8 细节特征，与语义特征融合。其输入来自原 RGB，区别于仅把 patch 特征插值到细网格。
- 共享任务特征进入两个独立的 SOD/COD decoder；推理调用 `task="cod"`，仅输入 RGB。SOD 监督仍会影响共享表征，独立 decoder 不等于已经消除负迁移。
- 多模态教师在特征值上融合 RGB/T/D。容量匹配的 RGB 教师保留相同模态适配器和融合参数，但三个输入槽都使用 RGB 编码；两种教师都对 `R` 和 `RTD` 两个 coalition 等权监督。
- 源监督与 KD 的权重同步余弦衰减，默认从 0.5 降至 0.05；COD 监督权重始终为 1。衰减是本版明确记录的设计，是否有效仍需控制实验。

单任务监督为 weighted BCE + weighted IoU、两级辅助 mask 监督和边界监督。默认辅助权重 0.5，边界权重 0.2。边界标签采用 3×3 形态梯度，正负边界像素按图像内的非空类别等权，避免背景数量支配边界损失。

`ordinary` KD 是普通 soft Bernoulli 输出 KL。`task` KD 直接监督实际 mask logits 与共享 stride 8/16 特征，不回归教师自身的差值。可靠权重取源 TRAIN 像素上 `max(BCE(R, GT) - BCE(RTD, GT), 0)`；输出 KL 和归一化特征距离按前景内部、背景内部、边界的非空区域等权，再按非空图像和特征尺度等权。教师全程停止梯度；无可靠区域时返回可导零。特征网格上的区域质量由 `reliability × region` 联合面积平均得到，不将分别下采样的标签和权重相乘。

T2 中的 `R/RTD` 都是 RGB 信息，可靠性表示同一个 RGB 教师两种 coalition 的优势；T4 中才涉及 T/D 增量。T3→T4 同时改变可靠区域、归约和特征监督，应报告为组合方案对照，不能把差值单独归因于某一项。

## 2. 分阶段控制

### baseline：先检验 RGB 主体

四组都仅使用 COD RGB/GT，并使用同一 v4 decoder、训练预算和划分。R0 是新版冻结主体控制，不是历史 v3 A 的直接复现。

| 组 | 主干末 block 数 | RGB 细节支路 |
|---|---:|---|
| R0_frozen_no_detail | 0 | 关闭 |
| R1_blocks4_no_detail | 4 | 关闭 |
| R2_frozen_detail | 0 | 开启 |
| R3_blocks4_detail | 4 | 开启 |

按内部 COD 验证、逐来源表现及计算成本选择主体，再固定 `--train-backbone-blocks` 和 `--detail-branch/--no-detail-branch`。之后 teachers、screen、final 必须使用同一选定主体。默认值是末 4 block + detail，不表示 R3 已被实验选中。改变主体后不能复用不匹配的教师或学生结果。

### teachers：两个容量匹配教师

独立训练 `rgb` 和 `multimodal` 教师，只使用 VDT 官方 Train 内部划分的 838 TRAIN / 210 VAL，默认 100 轮，以预设源验证监督损失选 checkpoint。二者成员、参数结构、初始化规则、更新预算和选模规则匹配；不读取 VDT 官方 Test。

### screen：检查额外数据与多模态的独立贡献

| 组 | 训练目标 | 教师 |
|---|---|---|
| T0 | 仅 COD 监督 | 无 |
| T1 | COD + VDT RGB/GT 监督 | 无 |
| T2 | T1 + reliable output/task-feature KD | 容量匹配 RGB 教师 |
| T3 | T1 + 普通输出 KD | 多模态教师 |
| T4 | T1 + reliable output/task-feature KD | 与 T3 相同的多模态教师 |

所有学生从同一初始化开始；每个 COD batch 对应一个循环使用的 VDT batch（T0 没有源 batch）。源样本暴露和更新次数保存在训练记录中。screen 使用 COD 官方训练池的内部 3232 TRAIN / 808 VAL，默认每 5 轮验证一次，以来源等权 macro Sm 选择同一 checkpoint，平分保留先出现的 checkpoint。它不生成正式测试成绩。

先检查 T1−T0，再检查 T2/T3/T4 的差异。至少对保留方案和关键控制运行 3 个配对训练种子；split seed 保持固定，训练 seed 改变，每个 seed 使用独立输出根和匹配教师。不得把不同指标各自最好的 epoch 拼成一行，或根据结果改变宏平均权重。

### final：冻结方案后单独启动

final 是显式阶段，从通用初始化重新训练全部 4040 张 COD 训练图；不沿用 screen 学生权重，不再用原内部 VAL 选模，固定最后 epoch。默认预算仍为 100 轮；若开发后决定不同预算，先记录并冻结，再为全部匹配控制使用相同预算。

final 会训练 T0–T4，并在每个最终权重上评测完整 COD10K、CAMO、CHAMELEON、NC4K。没有 `all` 自动从诊断或筛选升级到 final 的入口。源教师仍按原先冻结的 VDT 协议复用，教师加载时校验来源划分、主体、训练资源、源码和 checkpoint 哈希。

## 3. 服务器运行

建议把代码包解压到独立的 `/data2/yyy/TriPRD_task_v4`，复用已有数据、环境和通用权重，保留旧项目。代码不替用户下载 DINOv3 权重；路径需指向实际保存的官方源码与权重。

```bash
cd /data2/yyy/TriPRD_task_v4
conda activate triprd-v3

export OUTPUT_ROOT=/data2/yyy/TriPRD_task_v4/results/seed-20261012
export DINO_REPO=/data2/yyy/TriPRD/dinov3
export DINO_WEIGHTS=/home/yyy/data/TriPRD/dinov3/facebook--dinov3-vitb16-pretrain-lvd1689m/snapshots/master/model.safetensors

COMMON=(
  --output-root "$OUTPUT_ROOT"
  --gpus 2,3
  --dino-repo "$DINO_REPO" --dino-weights "$DINO_WEIGHTS"
  --vdt-root /data2/yyy/TriPRD/datasets/VDT-2048
  --cod-root /data2/yyy/TriPRD/datasets/COD_benchmark
  --seed 20261012 --split-seed 20261012
  --image-size 384 --batch-size 8 --num-workers 4
  --epochs 100 --teacher-epochs 100
  --train-backbone-blocks 4 --detail-branch
  --augmentation crop_flip
)

# 默认只打印计划。确认路径和配方后，加 --execute 才开始该阶段。
python tools/run_triprd_task_queue.py --phase baseline "${COMMON[@]}"
python tools/run_triprd_task_queue.py --phase baseline "${COMMON[@]}" --execute
```

每张卡运行一个独立实验，没有 DDP。`--gpus` 为物理卡号；若进程已有 `CUDA_VISIBLE_DEVICES`，指定卡需在其可见范围内。部分主干解冻、教师多模态前向和多个任务分支会增加显存需求，**batch 8 不是已确认的显存保证**。先用实际硬件检查所需配置；若必须降低 batch size，应为同阶段的全部配对组统一更改，并记录新的训练步数，不能仅为某一组减小 batch 后仍称预算一致。不要在运行中修改计划以绕过恢复校验。

baseline 完成后，检查结果并把 `COMMON` 中主体参数冻结为选定配置。随后分别运行：

```bash
python tools/run_triprd_task_queue.py --phase teachers "${COMMON[@]}"
python tools/run_triprd_task_queue.py --phase teachers "${COMMON[@]}" --execute

python tools/run_triprd_task_queue.py --phase screen "${COMMON[@]}"
python tools/run_triprd_task_queue.py --phase screen "${COMMON[@]}" --execute
```

screen 的 T2 自动读取 `teachers/rgb/best.pt`；T3/T4 自动读取 `teachers/multimodal/best.pt`。缺少完成证据或合约不匹配时会报错，应补齐实际结果或使用新的独立实验目录，不能手改哈希。

完成配对验证、明确冻结方案后，才执行以下独立命令：

```bash
python tools/run_triprd_task_queue.py --phase final "${COMMON[@]}"
python tools/run_triprd_task_queue.py --phase final "${COMMON[@]}" --execute
```

shell 快捷入口为 `tools/run_triprd_task.sh`：`PHASE=baseline`、`GPUS=2,3`、`OUTPUT_ROOT`、`DINO_REPO`、`DINO_WEIGHTS` 等环境变量配置任务；`DRY_RUN=1` 预览，`RESUME=1` 恢复。shell 默认执行 baseline；需要只看计划时显式设置 `DRY_RUN=1`。以 Python `--help` 为完整参数来源，改变阶段不代表允许改变既有阶段的冻结参数。

## 4. 恢复与结果文件

同一阶段、同一配置和源码，追加 `--resume`：

```bash
python tools/run_triprd_task_queue.py --phase screen "${COMMON[@]}" --execute --resume
```

runner 每个完整 epoch 原子保存 `last.pt`，含模型、optimizer、scheduler、AMP scaler、随机数和 DataLoader generator 状态，以及训练 history。恢复从下一个完整 epoch 开始；未保存完的部分 epoch 不被宣称已恢复。已完成的运行校验 checkpoint 后跳过。独立进程和队列锁用于防止两个进程写同一目录；强制中断遗留锁时，先核实记录中的进程已经结束再处理。

各阶段有独立计划文件。`--resume` 拒绝改变同阶段参数、源码或教师哈希；baseline 后选定主体可以用于尚未运行的 teachers/screen/final，但教师合约仍必须匹配。修改方法、训练 seed、预算或已运行阶段配方时使用新输出根。

```text
OUTPUT_ROOT/
  _data/split_manifest.json
  _queue/{baseline,teachers,screen,final}/plan.json
  baseline/{R0_frozen_no_detail,R1_blocks4_no_detail,R2_frozen_detail,R3_blocks4_detail}/
  teachers/{rgb,multimodal}/
  screen/{T0,T1,T2,T3,T4}/
  final/{T0,T1,T2,T3,T4}/
```

运行目录包括 `comparison.json`、`history.json`、`split_manifest.json`、`best.pt`、`last.pt`，队列另保留进程日志。正式 final 测试输出 `predictions/<dataset>/*.png` 和 `test_metrics.json`。选模 checkpoint 的所有指标共同报告，训练过程中的 source/KD loss 只是优化诊断。

若最终训练已完成但测试中断，可在相同 final 阶段配置下恢复。也可单独评估已完成的完整训练运行：

```bash
python tools/run_triprd_task.py --out "$OUTPUT_ROOT/final/T4" \
  --eval-only --device cuda --cod-root /data2/yyy/TriPRD/datasets/COD_benchmark \
  --dino-repo "$DINO_REPO"
```

该入口读取已保存的训练配置与选中权重，不重新访问源训练数据，也不训练模型。tiny 或 split 开发运行不具备最终 benchmark 导出资格。

## 5. COD 成绩可比性

- 最终 COD 训练为 COD10K 3040 + CAMO 1000，共 4040。内部 3232/808 开发成绩不填入论文正式测试表。
- 四测试集完整计数为 COD10K 2026、CAMO 250、CHAMELEON 76、NC4K 4121。每个运行使用同一权重、RGB-only、单尺度推理；不按测试集换 epoch 或选择后处理。
- 新版预测协议先把 logits 双线性恢复到原 GT 尺寸，再 sigmoid、round 到 uint8 PNG。保存方式采用 [CamoFormer 官方 test.py](https://github.com/HVision-NKU/CamoFormer/blob/main/test.py) 的约定；min/max 归一化在其[指标实现](https://github.com/HVision-NKU/CamoFormer/blob/main/evaltools/metrics.py)中进行。验证与最终导出使用同一版本化函数；独立评测校验预测与 GT 的唯一 ID 集合及完整覆盖，防止缺图得到正常分数。
- 报告 Sm、weighted F、MAE，并明确区分 adaptive、mean、max E。不能把旧版 mean E 列直接当成采用 adaptive E 的对照方法同名列。
- 披露 DINOv3 通用预训练数据/权重、输入尺寸、总参数/可训练参数、训练更新与源图像暴露、VDT RGB/GT 和 T/D 教师，以及训练/推理成本。VDT 是额外数据与特权监督；其绝对测试分数可比较，但不能宣称与只使用 COD + ImageNet 的方法资源等价。
- 对外部方法优先使用官方预测图和同一指标实现复评；仅引用论文分数时标明来源及 E 类型。不根据正式测试反馈调主体、损失、选模、阈值或预测流程。

本地功能测试通过只证明新链路可以检查与复现。正式结论仍需服务器真实训练、配对控制、多 seed 证据和冻结后的完整四测试集结果。
