# TriPRD 结构迁移第二版

更新：2026-10-17。入口为 `run_triprd_structure.py` 与 `run_triprd_structure_queue.py`，方法标识 `triprd_structure_v2`。这是已实现、待验证真实 GPU 效果的新研究方案；旧 `relation_v1` 代码、权重协议与 E011–E014 结果保留，不能混用组别或 checkpoint。

## 为什么改变框架

旧版同 seed 的四组宏 Sm 为 A 0.808668、B 0.807764、C 0.807881、D 0.807632，三模态迁移没有改善预先约定的主指标。D 的源教师 Dice 比 C 高 0.049566，但教师关系熵 0.990839、源学生关系熵 0.996127，接近均匀。源教师可从 T/D 值和多尺度均值融合直接获益，而学生蒸馏的是局部关系权重；两者并不是同一个信息通路。COD 后期训练继续改善、验证回落，也不支持优先延长原训练。

完整数字、证据范围与真实数据检查见[首轮诊断](../experiments/relation_v1_diagnosis_2026-10-16.md)。旁路造成多少损失、源知识是否被擦除，尚无旧权重的干预实验，仍是机制假设。

第二版的研究问题是：**真实辅助观测能否教会 RGB 预测有用的局部结构，并由 COD 自己选择这些结构的用途？** 结构预测、保存和使用的路径都作了修改。

## 已实现的计算路径

```text
源训练
真实 RGB/T/D ─→ 教师三编码器 ─→ 两尺度三类边
                   RGB 特征 ─→ 按边传播 RGB 值 ─→ SOD 预测
真实观测与源掩码 ─→ 独立结构监督 ────────────┘

RGB ─→ 固定参考编码器 ─→ 学生结构头 ─→ 观测边监督及可选教师 KD

COD 训练与推理
RGB ─→ 完整固定的参考编码器及结构头 ─→ 三类候选边 ─┐
RGB ─→ 可训练 COD 主干 ─→ 语义兼容边和上下文门控 ─┼→ 两尺度传播 → 逐级细化 → COD
                                                    └→ 可选择 identity
```

- **教师信息路径。** T/D 只决定边，分割传播的值、解码器特征及门控输入都来自 RGB。没有 T/D 多尺度均值融合旁路。教师的 RGB 与多模态控制具有相同参数布局和初始化。
- **两尺度结构。** ResNet-50 的 stride 8/16 分别预测热图外观、相对深度和源标注兼容性。每条边是独立 sigmoid 概率，不是 9 邻域 categorical softmax。TinyEncoder 仅作测试，其对应索引的实际 stride 为 2/4。
- **完整函数迁移。** 学生结构分支包含独立的参考编码器与结构头。参考编码器从 ImageNet 初始化并在源阶段固定；整个结构分支在 COD 阶段固定，包括 BatchNorm 状态。COD 主干、传播模块、语义图和解码头重新同 seed 初始化后训练。仅冻结头而改变其输入特征不算保留源函数。
- **目标任务使用结构。** COD 根据 RGB 上下文和粗预测选择 identity、三种结构候选或独立语义候选。源候选与 COD 语义兼容性结合后传播 RGB 值；identity 允许关闭传播。固定单位自环使弱边不会被强制放大。两个尺度都进入逐级细化，并接受粗预测监督。
- **目标语义保持独立。** COD GT 监督可训练语义图；不会将固定深度或热图关系重新命名为 foreground/background。源掩码的同类关系也不等于同一实例或场景所有对象的 objectness。

模块来源思路参见[方法构建文献](../literature/sod_to_cod_method_design_2026-10-16.md)，尤其 PAP、VSCode、Spider 与 AffinityNet。已有图传播、条件读出和蒸馏算子不是本项目发明；贡献仍须由跨任务实验建立。

## 真实数据决定结构标签的含义

VDT 的深度图虽常为 `I;16`，有效数值并未铺满 65535。抽查的多数图中位数约 2000；直接用归一化满量程上的固定差值尺度会得到几乎全为 1 的深度边。热图抽查为伪彩色 RGB，不能把灰度亮度称为物理温度。

本版处理为：

1. 深度保留单通道；零深度、非有限值与范围外读数不参与监督。热图保留 RGB 颜色，灰度热图也统一为三通道；热图零响应可以有效。
2. 在 resize 前建立有效性，用有效值加权插值，避免缺失深度污染邻域。网络输入中的无效值置零；结构目标在 feature 网格上再按有效覆盖率池化，要求覆盖率至少 0.99。
3. 对每张图的有效观测计算 q95−q05 的相对范围；颜色按通道 RMS 合并。分位数使用最多 4096 个确定性抽取的有效像素。范围退化时用完整范围，完全常数图使用 1 以保持有限结果。
4. 边目标为 `exp(−邻点观测差的通道RMS / (相对范围 × 0.1))`。该定义对深度单位缩放具有稳定性，表达相对观测连续性；热图表达颜色外观连续性。二者都不直接代表物体归属。
5. padding 和自环不参与结构损失。标注边按前景—前景、背景—背景、跨标注边界分别平衡；观测边按高低连续性分组，避免数量最多的平滑背景主导训练。

同一批 32 张官方 Train 图像的检查中，修正后的深度目标在 stride 8/16 上低于 0.5 的比例约为 15.44%/24.06%，此前均为 0；标准差由约 0.005 增至 0.228/0.272。它只说明目标不再近乎常数，不说明边界正确或 COD 已提高。

预处理审计保存每个源样本的原始模式、归一化除数和输出通道。固定标签定义和预处理协议进入 checkpoint 校验。若整个源 train 或 validation 的必需模态没有有效边，运行会失败，不会把无监督的分支当作成功的三模态训练。

## 五组实验各回答什么

以下 A–E 专属于 `structure_v2`，与旧版 A–D 含义不同。

| 组 | 源学生监督 | 教师 | 要检验的作用 |
|---|---|---|---|
| A_no_source | 无；同架构随机结构头固定，COD 语义路径正常训练 | 无 | 新架构本身的目标任务基线 |
| B_mask_structure | RGB/SOD 与源标注兼容边 | 无 | 源图像和掩码结构预训练 |
| C_direct_observed | 与 B 相同，并增加真实 T/D 观测边 | 无 | 直接特权结构监督 |
| D_rgb_teacher | 与 C 相同，加结构 KD | RGB/RGB/RGB，教师也有真实观测边标签 | 教师提炼相对直接监督的价值 |
| E_multimodal_teacher | 与 D 相同 | RGB/T/D，标签及教师容量与 D 匹配 | 真实多模态教师输入的额外作用 |

**D 已经使用特权标签**，不能称为完全不使用 T/D 信息。E−D 隔离教师输入，D−C 检验教师提炼，C−B 检验额外观测目标，E−A 是同部署结构上的总体收益。B 的另外两类头可经源分割梯度训练，但没有物理观测标签锚定。

训练中的三个源结构项各固定占 1/3 权重：增加 T/D 目标不会降低原标注兼容项的系数。默认总目标为最终 BCE+IoU、0.5 倍粗预测损失、1 倍 COD/源语义兼容性，以及 1 倍上述结构组合；D/E 另加 1 倍 detached Bernoulli KL。缺失的结构类型不会通过重新归一化改变其他训练项权重。

源模型按 validation `structure_supervision` 选模，即实际受监督类型的平均结构损失，越小越好；SOD Dice 仍记录但不作结构选模标准。COD 继续按每 5 epoch 原 GT 尺寸、COD10K/CAMO 等权宏 Sm 选模。B–E 共享源学生训练预算，D/E 另用相同的教师训练预算；所有组共享 COD 预算、划分和相应初始化。D−C 同时引入教师及其额外训练成本。新参考函数增加了模型成本，不能只与旧小模型比较后把全部收益归因于特权信息。

## 运行方式

在服务器仓库根目录执行。默认 ResNet-50 ImageNet、352 输入、width 256、batch 8，COD 与 VDT 各 100 epoch。先检查 dry-run；不带 `--execute` 不写实验目录、不启动训练：

```bash
python tools/run_triprd_structure_queue.py --gpus 0,1,2,3 \
  --epochs 2 --vdt-epochs 2 --max-samples 32 --val-metrics-every 1 \
  --output-root results/optimization/structure_smoke
```

实际 smoke：

```bash
python tools/run_triprd_structure_queue.py --gpus 0,1,2,3 \
  --epochs 2 --vdt-epochs 2 --max-samples 32 --val-metrics-every 1 \
  --output-root results/optimization/structure_smoke --execute
```

五个任务在四张卡上排队，同一张卡不会同时运行两个任务。smoke 用于确认输出完整、损失有限、结构有效覆盖和冻结检查；不按两轮小样本排名选择方法。双路径和两尺度图的真实显存、吞吐须在这一步测量，不能用 CPU 测试代替。

完成匹配筛选：

```bash
python tools/run_triprd_structure_queue.py --gpus 0,1,2,3 \
  --seed 20261012 --split-seed 20261012 \
  --output-root results/optimization --execute
```

新结果为 `results/optimization/structure_v2/seed-20261012/` 下五个组目录，不覆盖旧 `relation/`。服务器保存有旧轮完整源/目标清单时，可加 `--split-manifest results/optimization/relation/seed-20261012/D_multimodal_relation/split_manifest.json` 固定相同成员；本地只收到 comparison，不能冒称已独立核验了那份清单。

队列可用 `--groups A,B,C,E` 先检查直接监督和多模态方案，但若要判断教师多模态输入的独立价值，仍需 D；另起不冲突的输出根目录运行补充组。不要事后通过减少 A 的预算或改变选模指标放大组间差距。

单任务入口支持 `--teacher-checkpoint GROUP=PATH`、`--structure-checkpoint GROUP=PATH` 与 eval-only。结构复用必须提供含完整 VDT train/val 成员的清单，即使不重新读取源图像，也要核验源 seed、分区成员哈希、epoch、batch、优化器参数及损失权重。默认不接受把不同源划分或预算训练的教师包装成匹配实验。旧版 relation/full checkpoint 会被拒绝。

`--cod-train-mode full` 只使用完整 4040 张 COD 训练图像、禁用内部验证选模，固定采用最后 epoch；需先冻结训练时长。正式测试须显式 `--no-skip-test`，eval-only 须传每组 COD checkpoint。输入数据覆盖检查沿用正式测试数量要求，不能用测试集选择图类型或模型。

## 现有 checkpoint 的四测试集评测

使用 [`evaluate_structure_v2.py`](evaluate_structure_v2.py) 评测现有 A–E 五组 COD `best.pt`。默认读取 `results/optimization/structure_v2/seed-20261012`，从各组原始 `comparison.json` 恢复模型与输入配置，并核验权重 SHA256；只加载已选中的 COD checkpoint，不重新训练、不按测试分数重新选模。脚本调用现有 structure runner 的 `--eval-only --no-skip-test` 路径。

在保存权重的服务器仓库根目录、原训练 Python 环境中执行：

```bash
# 仅查看路径、参数及缺失的 checkpoint，不启动推理、不写结果。
python tools/evaluate_structure_v2.py --dry-run

# 将 0 换为空闲 GPU。默认 A–E 顺序评测，输出到独立的时间戳目录。
CUDA_VISIBLE_DEVICES=0 python -u tools/evaluate_structure_v2.py

# 自定义输入与输出位置；输出目录须为空或尚不存在。
CUDA_VISIBLE_DEVICES=0 python -u tools/evaluate_structure_v2.py \
  --run-dir results/optimization/structure_v2/seed-20261012 \
  --cod-root datasets/COD_benchmark \
  --out results/optimization/structure_v2_test/seed-20261012
```

用 `--groups A,E` 可只评测指定组。每组始终评测 NC4K 4121、COD10K-Test 2026、CAMO-Test 250、CHAMELEON 76，共 6473 张；不提供小样本测试选项。预测图保存在各组 `predictions/<dataset>/`，正式指标位于新输出目录的 `comparison.json`。覆盖检查和全部指标通过后，另导出 `metrics.csv`、`previous_performance_rows.tex` 和 `evaluation_manifest.json`。原始训练目录和 `previous_performance.tex` 不会被覆盖。

评测使用本仓库保留的 CamoFormer 指标类及 `tools/evaluate_cod.py`，预测和 GT 均通过 OpenCV 灰度解码；尺寸不一致时使用 OpenCV 默认的双线性插值将预测恢复到原始 GT 尺寸。指标类保留原实现的逐图预测 min–max 归一化、`GT > 128`、adaptive 阈值、256 阈值曲线及逐图平均，不改为固定阈值或像素池化。推理保持 checkpoint 原有的 RGB 归一化、输入大小和预测导出规则；不会为迎合测试成绩改变这些设置。CamoFormer **eval** 和模型各自的 **inference** 是不同环节，不能把另一模型的输入/输出处理直接替换到当前权重上。

`previous_performance.tex` 中四组数据集列顺序为 NC4K、COD10K、CAMO、CHAMELEON，每组按以下字段导出，显示三位小数，JSON/CSV 保留精度：

| 表格列 | 指标字段 | 定义 |
|---|---|---|
| Sm | `S_alpha` | S-measure |
| αE | `E_phi_adaptive` | adaptive E-measure |
| Fw | `F_beta_w` | weighted F-measure |
| M | `MAE` | mean absolute error |

[CamoFormer 原论文](https://arxiv.org/pdf/2212.06570) §4.1 明确将 αE 定义为 adaptive E，且表中对应的 CamoFormer 数字与其 Table 1/7 一致。[官方 eval.py](https://github.com/HVision-NKU/CamoFormer/blob/main/evaltools/eval.py) 同时计算 adaptive/mean/max E；其默认日志使用 mean E，不应据此将论文 αE 列替换为 mean E。本脚本明确读取 `E_phi_adaptive`，并另存 mean E；本轮之前生成的 JSON 里 `alpha_E` 曾指向 mean E，不能直接复制该旧别名填表。已有训练验证的 `E_phi` 保持 mean E 定义。

一致性核验针对本脚本导出的无损 PNG：官方 eval 在尺寸不匹配时会缩放、写回、再读取，本地在内存缩放，PNG 像素结果一致。不要将这个结论扩展到会重新压缩的 JPEG。2026-10-19 获取的官方 `metrics.py` 与本地 vendor 的规范化 Python AST 一致；[`test_camoformer_evaluation.py`](../tests/test_camoformer_evaluation.py) 固化了独立运行官方代码得到的数值，覆盖 Sm、Fw、mean/adaptive/max E、MAE 等指标、异尺寸预测、空 GT 及解码边界。测试本身无需联网。

输出会记录每组 COD 训练图像数、选中轮次和源数据使用情况。现有 seed-20261012 权重使用 3232 张 COD 训练图像及 808 张内部验证，B–E 还使用 VDT 源数据；补测不改变训练协议。与采用 4040 张 COD 图像训练的方法比较时必须注明此差异，不能标成同训练规模实验。本地若只有 `comparison.json` 而没有 `.pt`，只能检查计划，真实评测须在持有 checkpoint 的服务器执行。

## 如何读新结果

每组 `comparison.json` 包含 source/teacher/COD history、源结构误差、有效边比例、冻结哈希验证与默认后选模干预。首先看：

- `structure_thermal/depth/annotation`、各自 `*_mae` 和 `*_valid_fraction`：确认源监督存在且能被 RGB 学习。低损失但没有有效边不代表成功；预测标准差也不能单独代表结构质量。
- 同一 Sm 选中 checkpoint 的全部四项宏指标与两个数据来源；不要拼接不同 epoch 的最好单项。
- `intervention_validation.metrics` 与 `delta_from_normal`：`no_source` 只关闭固定源候选，保留 COD 语义图；`uniform_source` 保留每类候选平均强度但消除结构；`shuffle_source` 打乱源边空间对应。`identity` 关闭整个图传播，包括 COD 语义图，因此它不能单独证明源知识的价值。

这些干预不重新选择 checkpoint。若 E 比 A 高，但 E 与 C/D 无差别，就不能把改善归因于多模态教师。若去除或打乱源候选几乎不影响预测，应承认模型未有效利用源结构，不能仅凭 KD 非零继续宣传迁移成功。新 A 与旧 A 的提升主要检验解码器和部署结构，需与组间特权贡献分开报告。

## 成本与本地验收范围

ResNet-50、width 256、relation dim 32 下：学生总参数 **38,826,263**，COD 可训练 **30,227,409**；完整结构分支 **8,598,854**，其中固定参考编码器 **8,543,296**、源结构头 **55,558**。教师为 **47,467,863** 参数。参考编码器和教师辅助编码器中未消费的最后 stage 已裁除；图输入特征与未裁版本逐元素一致。固定参数仍产生推理计算，不能按可训练参数宣传部署成本。

本地环境为 PyTorch 2.8 CPU，无可用 CUDA。完整回归为 **164 passed、210 subtests passed**。测试覆盖五组 tiny 模型两轮真实优化、源结构选模、完整函数转移、COD 冻结、严格复用、全量模式与 eval-only、教师旁路干预、观测尺度和颜色处理。旧测试默认系统临时目录受到沙箱访问限制，授权在沙箱外重跑后完整通过。

另完成真实 ResNet-50 的三阶段 CPU 单步（width 256、relation dim 32、batch 2、64×64、不加载预训练权重）：教师 observed、源学生 observed+KD、fresh COD 各自反向并优化，损失与全部已产生梯度均有限；源头和 COD 主干/语义头发生更新，COD 固定分支包含 BN buffer 的哈希严格不变，固定 RGB 下源结构输出逐元素不变。该检查不代表 352 输入的 GPU 显存或收敛测试。真实 VDT 图像仅用于数据定义检查；未执行新版本完整 GPU 训练，也没有新 COD 性能结果。

代码入口：[模型](../models/triprd_structure.py)、[结构目标](../models/triprd_structure_loss.py)、[训练与数据协议](triprd_structure_training.py)、[运行器](run_triprd_structure.py)、[队列](run_triprd_structure_queue.py)。
