# TriPRD correction_v3

本版本是独立的 RGB 部署学生和共享四输入组合教师，入口为 `run_triprd_correction.py`。正式默认使用本地通用预训练 **DINOv3 ViT-B/16**、384 输入、batch 8、解码宽度 128；教师与各学生均训练 100 epochs。它没有复用已有 COD 方法的权重。此文说明已实现的运行协议，不代表 GPU 实验已经完成，也不承诺超越已有方法。

## 分阶段优化入口（2026-10-20）

针对 structure_v2 未显示稳定多模态增益的问题，推荐先执行下面的四阶段入口，再考虑后文的六组扩展。入口为 `tools/run_triprd_optimization_phases.py`，Linux 包装脚本为 `tools/run_triprd_optimization.sh`。这套优化使用 correction_v3；**v3 的 A/B/C/D 定义与 structure_v2 不同**，输出目录和表格名称均分开。

本次代码修正与新增功能：

- 教师四种输入组合的关系损失各占 `1/4`。此前基础 RGB 关系被反复纳入，实际是 R 占 `5/8`、RT/RD/RTD 各占 `1/8`。现在只在学生中保留基础关系与修正后关系的双头监督。
- 学生修正头末层零初始化，初始关系等于 RGB 基础关系；修正头和共享 RGB 适配层随后通过训练学习。各对照组保持相同结构和初始参数。
- 新增 `--stage probe`：仅用 VDT train 优化 RGB 学生，在 VDT val 测量修正可恢复性。它不索引或读取 COD、不执行正式测试，不根据探针挑选最佳轮次；固定保存末轮 `last.pt`，不生成 `best.pt`。诊断权重不用于正式学生初始化。
- 新入口用固定预算与数据成员清单串行执行；后续阶段检查前一阶段的配置、源码、通用权重和 checkpoint 哈希。最终表格继续使用 CamoFormer 指标，输出自适应 E，不用 mean E 替代论文的 alpha E。

先在服务器准备 CUDA PyTorch 和依赖，以及官方 DINOv3 本地代码和 ViT-B/16 通用预训练权重：

```bash
python -m pip install -r requirements-correction.txt
export DINO_REPO="$HOME/data/TriPRD/dinov3"
export DINO_WEIGHTS="$DINO_REPO/facebook--dinov3-vitb16-pretrain-lvd1689m/snapshots/master/model.safetensors"
python tools/check_dinov3_weights.py --device cuda
export CUDA_VISIBLE_DEVICES=0,1,2,3
export GPUS=0,1,2,3
export OUTPUT_ROOT=results/optimization/correction_4gpu

# 仅查看计划，不训练、不写入输出。
DRY_RUN=1 bash tools/run_triprd_optimization.sh baseline

# 按顺序运行全部四阶段；失败或未通过实验门槛就停止。
bash tools/run_triprd_optimization.sh all

# 也可单独运行一个阶段，例如：
# bash tools/run_triprd_optimization.sh baseline
```

| 阶段 | 执行内容 | 四卡调度 | 进入下一阶段的依据 |
|---|---|---|---|
| baseline | A：仅 COD RGB/GT；B：COD + VDT RGB/GT，各 100 epochs | A/B 各占一张卡并行 | 建立同结构、同初始化的两个基线 |
| probe | 共用教师 100 epochs；D 学生仅 VDT 训练 10 epochs | 单卡依次执行教师、探针 | 教师在 VDT val 的关系误差优于自己的 RGB 输入；学生在可靠修正边上同时优于零修正和 TRAIN 拟合常数 |
| screen | 复用并校验 A/B，训练 C：完整关系 KD；D：可靠修正 KD，各 100 epochs | 校验 A/B 后，C/D 各占一张卡并行 | 检查完整 A/B/C/D 的固定内部 COD 验证；D 的选中 checkpoint 宏 Sm 必须高于 A/B/C |
| final | 从通用预训练重新初始化 A/B/C/D，用完整 4040 张 COD 重训，各 100 epochs | A/B/C/D 各占一张卡并行 | 固定末轮，逐组评测四套官方测试集并导出表格 |

`GPUS` 是物理设备编号列表，可改为服务器实际编号，或完整 `GPU-...` 标识。若设置了 `CUDA_VISIBLE_DEVICES`，`GPUS` 必须是其子集；例如此前只导出 `CUDA_VISIBLE_DEVICES=0` 时，需要先按实际可用设备更新可见范围。每个子进程只看到分配给它的一张卡，内部设备为 `cuda:0`。这是独立实验并行，每组仍使用 batch 8，训练预算、初始化和数据划分不变；`NUM_WORKERS` 也按每个实验计算。

`all` 依次执行四阶段，范围是 A–D；E/F 扩展仍使用后文的独立队列。`baseline` 和 `probe` 不跨阶段重叠。每个任务的标准输出和错误写入其目录的 `run.log`；终端显示任务、GPU、PID 和日志路径。阶段记录还保存每个任务的运行状态。任何任务失败会停止启动余下任务并回收该阶段仍运行的子进程，检查点可在同配置下恢复。不要同时手动启动同一输出根目录的多个阶段。

`DRY_RUN=1 ... all` 会预览四阶段；缺少先前证据时相应阶段显示 blocked，最终返回 2，且不会训练或创建结果目录。

`probe` 的教师门槛是 `mean_relation_gain > 0` 且有效覆盖大于零；学生门槛是 `positive_mse` 同时小于 `positive_zero_mse` 和 `positive_constant_mse`，并具有非空有效边。D 只在正可靠性边上接受修正监督，因此全边 MSE 保留为诊断值；实际学生关系误差、修正后的教师 RGB 基础关系误差也分别记录，不能混称为学生分割精度。短诊断未通过只说明当前配置和预算下缺少可学习性证据，不证明所有配置都无效。通过门槛也不代表 COD 有收益，更不代表统计显著性。

常数对照分别在 VDT TRAIN 的全有效边和正可靠性边上按尺度/邻接偏移拟合，保存为教师 checkpoint 的 `constant_correction` 与 `positive_constant_correction`。正边门槛使用后者；前者在正边的误差另记为 `positive_all_edge_constant_mse`。两种常数在验证时都固定，不用验证数据重新拟合，避免把仅学到常数的学生当作学到了空间修正。

阶段未通过时返回非零退出码，并把原因写入 `seed-20261012/phases/<phase>.json`。`screen` 不读取正式测试成绩；`final` 才使用全训练池与 NC4K 4121、COD10K 2026、CAMO 250、CHAMELEON 76 张测试图。正式训练不会从开发 checkpoint 或 probe checkpoint 接着微调。额外 VDT 数据和通用主干预训练仍须在论文中披露；4040 张 COD 相同不代表所有训练资源相同。

默认输出结构：

```text
results/optimization/correction_phases/seed-20261012/
  optimization_recipe.json         # 固定配置、预算、源码及资源哈希
  _data/split_manifest.json         # 共用源域与 COD 开发分区
  phases/{baseline,probe,screen,final}.json
  teacher/best.pt
  probe/{last.pt,source_probe.json,history.json}
  A_cod_only/ B_rgb_joint/ C_full_kd/ D_correction/
  final/
    A_cod_only/ B_rgb_joint/ C_full_kd/ D_correction/
    summary.json
    table/metrics.csv
    table/previous_performance_rows.tex
```

表格顺序与 `previous_performance.tex` 一致：NC4K、COD10K、CAMO、CHAMELEON，每套为 Sm、自适应 E、加权 F、MAE。CSV 额外保留 mean E。生成表格前检查四套测试完整覆盖、checkpoint 哈希和 full 模式 4040 张/固定末轮协议，不自动修改论文表。

设置 `VDT_ROOT`、`COD_ROOT` 可改变数据位置；`BATCH_SIZE`、`NUM_WORKERS`、`EPOCHS`、`TEACHER_EPOCHS`、`PROBE_EPOCHS`、`IMAGE_SIZE`、`BASE_CHANNELS`、`LR`、`SEED`、`SPLIT_SEED` 可在首次启动前设置。shell 优先采用 `GPUS`，未设置则采用 `CUDA_VISIBLE_DEVICES`，两者都没有时使用 `GPU` 或默认卡 0。未测真实 DINOv3 显存；如需减小 batch，应从阶段开始前统一修改，保持后续阶段一致。不要通过各组不同的 batch 或预算消除比较的可解释性。

重复运行已启动的阶段，或在中断后恢复：

```bash
RESUME=1 bash tools/run_triprd_optimization.sh probe
# 或从整个流程恢复：已完成任务仍会核验，不重复训练。
RESUME=1 bash tools/run_triprd_optimization.sh all
```

进入新的阶段无需 `RESUME=1`；例如 `screen` 会自动校验并复用已完成的 A/B。仅调整 GPU 列表不改变训练方案，可以恢复；同目录修改训练配置、源码或预算会被拒绝，应使用新的 `OUTPUT_ROOT`。因此从旧版脚本升级到四卡调度时也使用新目录。**本次修正改变了训练目标，旧 v3 教师与学生 checkpoint 不作为新流程的续训起点。** 崩溃遗留 `optimization_lock.json` 时，先确认其中 PID 对应的进程已经退出，再处理锁文件。

直接 Python 入口默认只预览，显式 `--execute` 才运行；适用于 Windows 或需要更多参数的情况：

```bash
python tools/run_triprd_optimization_phases.py --phase baseline \
  --dino-repo "$DINO_REPO" --dino-weights "$DINO_WEIGHTS" --gpus 0,1,2,3 --execute
```

2026-10-20 联合验收：**98 tests passed、65 subtests passed**。覆盖 tiny CPU 的真实 baseline → probe → screen 优化、源域隔离、损失梯度、精确恢复、阶段门槛和表格导出；真实 DINOv3/CUDA 训练性能和提升幅度仍待服务器实验。

后续四卡调度验证使用真实 CPU 子进程模拟训练报告，覆盖并发重叠、物理卡号隔离、单卡轮转、依赖门槛、checkpoint 校验、恢复及训练/数据准备/汇总期间的中断回收；多 GPU 模块 19 项测试通过。Bash 入口另有 5 项测试，覆盖 `all` 顺序执行、失败停止、只预览、恢复参数与 SIGTERM 转发。这些验证不等同于实际四卡 CUDA 训练。

## 六组回答什么问题

| 组 | 配置 | 用途 |
|---|---|---|
| A_cod_only | 仅 COD RGB/GT | 同一学生架构的目标域基线 |
| B_rgb_joint | COD 与源域 RGB/GT 联合训练 | 分离额外源数据和 RGB 监督的收益 |
| C_full_kd | B 加完整教师关系蒸馏 | 普通多模态蒸馏控制 |
| D_correction | B 加教师相对 RGB 的关系修正 | 检查辅助观测提供的增量知识 |
| E_selected_correction | D 加 COD 训练梯度一致性筛选 | 检查修正与目标任务筛选的结合 |
| F_selected_full_kd | 对完整关系蒸馏施加相同筛选 | 排除收益仅来自普通 KD 重加权 |

B–F 的源 RGB/GT 监督保留全部样本；E/F 的筛选仅作用于迁移梯度。真实 T/D 只出现在 VDT 源教师，COD 学生训练和部署均只接收 RGB，不生成伪模态。教师在同一模型中处理 R、RT、RD、RTD 四种输入组合，六组共用同一次教师训练。关系监督表达二值标注是否相同，背景像素同标签不意味着同一物体；源 SOD 标注也不是通用物体标注。教师相对 RGB 的误差改善是可观测的迁移可靠性依据，不是已识别的因果效应。

默认冻结通用 DINOv3 权重，训练共享 RGB 投影/适配层、关系模块和解码器；没有第二套 RGB 参考编码器。教师四种输入组合在每个源 batch 上等权监督，特征提取按模态缓存一次。D/E 的源 GT 可靠性加权从教师概率差构造；E/F 的额外梯度选择为 **batch 级**，只用 COD train。学生预测 `tanh` 有符号修正用于 KD，再将 `base_probability + correction` 裁剪为合法传播概率，避免学生自身基准概率饱和使蒸馏目标不可达。去除/置换修正的干预不重选 checkpoint；置换保留原始修正分布，裁剪后的传播强度可能改变。

## 完整开发运行

Linux 上先准备官方 DINOv3 本地仓库和匹配的 ViT-B/16 通用预训练权重。脚本不自动下载、猜测权重或退回 tiny。数据默认位于 `datasets/VDT-2048` 与 `datasets/COD_benchmark`，可用 `VDT_ROOT`、`COD_ROOT` 改为实际路径。

依赖采用 PyTorch 2.8 或以上。先在服务器准备对应 CUDA 的 PyTorch/torchvision，然后安装其余依赖并确认 CUDA 可用：

```bash
python -m pip install -r requirements-correction.txt
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

通用主干来源为 [DINOv3 官方仓库](https://github.com/facebookresearch/dinov3)。支持官方通用预训练 `.pth` state dict，以及已经下载的 Hugging Face `facebook/dinov3-vitb16-pretrain-lvd1689m` 的 `model.safetensors`。HF 文件必须与 `config.json` 放在同一目录；保留现有 `snapshots/master/` 布局即可，无需重命名、重新下载或另存 `.pth`。`configuration.json` 不替代 `config.json`，特征预处理仍使用本项目 ImageNet 标准化。

`models/dinov3_weights.py` 在内存中将 HF 参数映射到官方架构：合并 Q/K/V，转换 mask token，重建确定性的 RoPE 和 K-bias mask。每个学习参数都必须存在、形状匹配且有限；最终 `strict=True` 加载，配置或权重不匹配会失败。支持旧版 `layer.*` 和新版 `model.layer.*` 键，HF 转换限 ViT-B/16；运行不依赖 Transformers。实现参照 [Hugging Face 官方转换脚本](https://github.com/huggingface/transformers/blob/main/src/transformers/models/dinov3_vit/convert_dinov3_vit_to_hf.py)。原始权重、配置与转换源码的 SHA256 纳入实验配置校验，报告中记录转换信息。更新代码后请使用新输出目录，不与旧源码的运行记录混合恢复。

未提供本地仓库或权重会明确失败，不下载替代权重；LVD-1689M 预训练资源不能标为 ImageNet-only。可先运行下列检查，只做严格加载和随机输入前向，不读取数据集。384 输入时应输出 `status: ok`、`source_layout: huggingface` 和四个 `[1, 768, 24, 24]` 特征；这确认格式兼容，不代表模型精度已经验证。

本地验证包括真实 safetensors 文件的严格映射/拒绝测试，以及完整官方 ViT-B/16 随机参数与独立 HF 前向计算的数值比较。后者使用两个矩形输入，覆盖四层特征与 mask token；FP32 最大绝对差约 `3.34e-6`。可用 `python tests/check_dinov3_hf_equivalence.py --dino-repo "$DINO_REPO" --check-cli` 复现，不下载预训练权重。此检查不证明服务器实际下载文件的完整性；上面的预检会在服务器读取该文件验证。

```bash
export DINO_REPO="$HOME/data/TriPRD/dinov3"
export DINO_WEIGHTS="$DINO_REPO/facebook--dinov3-vitb16-pretrain-lvd1689m/snapshots/master/model.safetensors"
python tools/check_dinov3_weights.py --device cuda
export GPUS=0,1,2,3,6,7
bash tools/run_triprd_correction_full.sh
```

该 shell 脚本**直接执行完整训练**，前台等待并返回整个队列的退出码。需要断开终端后继续时，在已经导出环境变量的终端运行：

```bash
nohup bash tools/run_triprd_correction_full.sh > correction_v3_queue.log 2>&1 &
echo $!
```

也可使用 tmux。队列不会把子进程变成不可追踪的 daemon。默认 teacher 在 GPU 7，A/B 在 GPU 0/1 先运行；C–F 只有在教师完成并通过探针检查后启动。六个学生分别固定使用 0、1、2、3、6、7，同一 GPU 同时最多一个队列任务。`CUDA_VISIBLE_DEVICES` 保留这些实际编号；每个子进程内的 CUDA 设备自然重编号为 0。

若只想审阅计划，Python 队列入口默认不启动任务、不写文件：

```bash
python tools/run_triprd_correction_queue.py \
  --dino-repo "$DINO_REPO" --dino-weights "$DINO_WEIGHTS"
```

增加 `--execute` 才执行。正式输出为 `results/optimization/correction_v3/seed-20261012/`。默认训练 seed 和分区 seed 均为 20261012；`SEED` 与 `SPLIT_SEED` 可分别设置，但比较组必须匹配。`NUM_WORKERS` 默认 4，`PYTHON` 可指定 Python 可执行文件。

## 分区、选模与正式测试

启动前只准备一次 `_data/split_manifest.json`，所有任务读取同一成员清单。`SPLIT_MANIFEST`/`--split-manifest` 可指定已有源域与 COD 完整分区清单，准备步骤验证并生成此次运行的清单。所有组记录相同训练预算、优化器参数及验证频率；A 不训练源数据，B–F 的源域训练成本应单列。

默认 `cod_train_mode=split`、每 5 epochs 验证，按宏 Sm 选中一个 checkpoint，再报告这个 checkpoint 的各项指标；开发默认不读取正式测试成绩。单个 seed 的差值不能证明统计显著，内部验证也不能直接与论文的官方测试表排名比较。

确定方法、预算和配置之后，可在完整 COD 训练池上固定训练最后一轮，并显式评估官方测试集。必须指定独立输出根目录，不能复用开发目录：

```bash
COD_TRAIN_MODE=full SKIP_TEST=0 OUTPUT_ROOT=results/final/correction_v3 \
  bash tools/run_triprd_correction_full.sh
```

full 模式必须 `max_samples=0`，禁用内部 COD 验证选模，固定采用最后 epoch。官方测试只用于冻结方案的最终报告，不用于选择分组、阈值或训练长度。请按报告中指标的明确定义核对 E-measure；不能把 mean E 当作 adaptive E。

## 完成、失败与恢复

共享教师的 `best.pt`、`comparison.json` 与 `teacher_probe.json` 必须齐全；报告须声明 `status=complete`，配置必须与计划一致，两个探针文件的 contract 和数值必须一致。`mean_relation_gain` 和 `positive_coverage` 必须有限，覆盖率须在 [0,1]。默认要求平均关系改善为正、正改善覆盖率大于零，否则 C–F 阻塞。进程返回 0 而缺少完整报告或权重仍算失败。A/B 独立运行，某个学生失败也不会取消其他独立学生；队列最终返回非零退出码。

若研究目的明确需要测量负增益教师，可显式 `ALLOW_NONPOSITIVE_PROBE=1`（Python 参数 `--allow-nonpositive-probe`）。这会记录警告并放行**有限**的非正探针，不会接受 NaN、缺失文件或配置不匹配。这种结果应按负对照解释。

已有目录默认拒绝覆盖。使用完全相同的配置加 `RESUME=1`（或 `--resume`）恢复；源码、命令、输入清单或协议发生变化时应使用新输出根目录。队列即便发现已有 complete，也会再次调用 runner 的 resume 校验，核对成员、源码、权重和训练 contract，不凭文件存在静默成功。教师失败后允许用同一计划的显式非正探针覆盖开关重新核验并继续；这不会改变训练参数。

根目录 `launch.json` 保存计划、命令、源码哈希与 GPU 映射；`exit_status.json` 持续记录 PID、当前状态、退出码、警告和清单哈希。各任务有自己的 `launch.json`、`run.log`、`exit_status.json` 和训练报告。`queue_lock.json` 防止同一目录双重启动；意外断电留下锁时，先确认记录的 PID 及对应任务确已退出，再人工移除锁，不能盲目重启重叠训练。

每个 epoch 的 `last.pt` 保存模型、优化器、scheduler、AMP scaler、Python/NumPy/Torch/CUDA RNG 及各 DataLoader generator 状态；恢复从下一个 epoch 开始。`best.pt` 保存同一完整状态，若在 last 与 best 的两次原子写入之间中断，先重建最佳状态再续训。源常数修正基线仅从 VDT train 估计，并存入教师 best.pt；源 val 的恢复误差同时报告零修正、常数修正和实际学生关系误差。

读取六组已选择 checkpoint 的结果，禁止跨 epoch 拼接最好单项：

```bash
python tools/summarize_triprd_correction.py \
  results/optimization/correction_v3/seed-20261012 \
  --output results/optimization/correction_v3/seed-20261012/summary.json
```

该命令检查学生初始化、训练成员、预算、源码/通用权重和 C–F 教师权重匹配。只生成汇总，不启动训练。

已冻结训练方案的 RGB checkpoint 可独立评估；这条路径不读取 VDT 或 COD train，不需要教师或原通用权重文件，只需官方 DINOv3 架构源码和完整学生 checkpoint。示例保持默认 384/128/ViT-B 的训练配置：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/run_triprd_correction.py \
  --stage student --eval-only --no-skip-test --dino-repo "$DINO_REPO" \
  --cod-checkpoint /absolute/path/to/student/best.pt \
  --cod-root "${COD_ROOT:-datasets/COD_benchmark}" \
  --out results/final/correction_rgb_eval
```

若训练时修改了输入尺寸、通道数或精度选项，评估时传入一致配置。不要用正式测试结果反复选择方法。

## AMP 梯度溢出修复（2026-10-21）

旧版在 `scaler.unscale_(optimizer)` 后调用带 `error_if_nonfinite=True` 的梯度裁剪，导致 FP16 梯度偶发溢出时先抛错，GradScaler 无法跳过更新和降低倍率。新版先计算梯度范数：有限时正常裁剪；梯度含 NaN/Inf 且启用 scaler 时，不修改梯度，让 `scaler.step()` 跳过整个优化器更新，再由 `scaler.update()` 降低倍率。这遵循 [PyTorch 2.8 AMP 的跳步语义](https://docs.pytorch.org/docs/2.8/notes/amp_examples.html#typical-mixed-precision-training)。

损失非有限、未启用 scaler 时梯度非有限、梯度元素有限但范数计算溢出，仍立即报错。每轮连续 8 次跳步或整轮没有任何有效更新也会报错，避免持续数值异常被掩盖。单次梯度异常不能据此认定只是缩放溢出；若修复后仍持续失败，日志中的参数名和样本编号用于进一步定位。

每轮日志新增 `updates`、`amp_skips`、`scale`；`history.json` / checkpoint 同步记录 `optimizer_steps`、`amp_skipped_steps`、`amp_scale` 和 `amp_overflow_events`（批次、样本、前后倍率、非有限梯度参数）。`steps` 为处理过的批次数，损失均值涵盖这些批次；`grad_norm` 仅在实际更新的批次上取均值。失败批次不会更新模型、AdamW 动量或权重衰减；后续批次使用降低后的倍率。原 scaler checkpoint 保存/恢复逻辑保持有效。

**此补丁改变了训练源码哈希，旧运行不能直接用 `RESUME=1` 续训。** 保留旧结果，上传补丁中的 `tools/run_triprd_correction.py`，从项目根目录以新输出目录启动同一套 A–D 实验；不修改旧 checkpoint、manifest 或配置哈希。以下路径对应当前服务器，沿用已经下载的权重：

```bash
cd /data2/yyy/TriPRD
conda activate triprd-v3
export DINO_REPO=/data2/yyy/TriPRD/dinov3
export DINO_WEIGHTS=/home/yyy/data/TriPRD/dinov3/facebook--dinov3-vitb16-pretrain-lvd1689m/snapshots/master/model.safetensors
export GPUS=0,1,2,3
export OUTPUT_ROOT=/data2/yyy/TriPRD/results/optimization/correction_4gpu_ampfix
export RESUME=0
bash tools/run_triprd_optimization.sh all
```

新目录以后发生普通中断且源码/配置不变，才可设置 `RESUME=1` 恢复。`all` 仍会检查 teacher/probe/screen 的证据条件，条件不满足时不会执行后续正式实验。可选验证命令 `python -m pytest -p no:cacheprovider tests/test_correction_amp.py -q` 使用 CPU 上真实的 GradScaler 和 FP16 反传验证溢出跳步与恢复；CUDA 专项仅在 CUDA 可用时运行。本地 CPU 测试不能替代服务器完整训练。

## 本地验证的边界

`tests/test_correction_queue.py` 使用真实 CPU 子进程测试依赖调度、进程退出与报告核验、教师失败阻塞、负增益覆盖、独立学生失败及恢复限制。测试子进程的权重文件只是夹具，不是模型训练结果。tiny 仅用于检查实际模型训练链路，不属于正式方法配置。完整 DINOv3 的真实显存、吞吐、收敛和性能仍须由服务器实验报告确认。

2026-10-19 本地验收：完整回归 **219 passed、225 subtests passed**；新版本专项 **55 passed、15 subtests passed**。另用真实数据完成 tiny/32 输入、2 epochs 的共享教师及六组 CPU 队列并成功原配置恢复；真实压缩包 COD 数据的 2-worker 训练完成；RGB-only checkpoint 在未提供 VDT 路径时完成 2 张 COD10K 的有限评估。完整成员清单为 VDT 838/210、COD 3232/808。上述小样本检查不是正式精度实验，不能据其差值比较方法或声称 SOTA。
