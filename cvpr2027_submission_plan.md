# CVPR 2027：Privileged Tri-Modal Relation Distillation for RGB-only COD

> 审核修订版：2026-10-13。本文档是唯一执行版本；原文件中两套重复计划已合并。本版把“研究假设”“可复现实验”“补充实验”分开，并为每个实验注明依据。论文截止时间以 CVPR 2027 官方页面为准：注册 2026-11-10 AoE，投稿 2026-11-16 AoE，补充材料 2026-11-23 AoE（见参考文献 R1）。

## 0. 审核结论与本次修正

原计划存在以下问题，已在本版处理：

1. 原文件把两版完整计划直接拼接，验收标准、数据检查数量和日期状态互相冲突。本版只保留一套阶段、主结果和补充结果。
2. “E00 U-Net/DeepLab”把两个不同模型写成一个实验，无法归因。本版固定 E00 为 U-Net-R50；DeepLabv3+-R50 改为可选的 E00b 结构敏感性实验。
3. “BGNet/FSPNet”把两个不同模型写成一个实验。本版 E04 固定为 BGNet；FSPNet只作为补充的 Transformer 参考，不混入 E04。
4. 五周内把 E00–E38 全部做三次种子训练不可执行。本版采用 screening（单种子）和 confirmatory（关键实验三种子）两级预算，E35–E38、噪声和超参敏感性移到补充材料。
5. VDT2048 是 RGB-D-T 显著目标检测数据集，不是 COD 数据集。本版把它定义为多模态特权知识源；COD 结论只在 COD10K、CAMO、NC4K、CHAMELEON 上验证。
6. 原计划没有定义 VDT/COD 的验证集、关系先验统计、边界标签和不确定性校准。本版给出固定的 train/val/test 规则；任何实现不符合规则时，必须作为新版本实验记录。
7. “如果结果不好就修改关系定义”会产生事后选择。本版先注册主指标、种子和损失，失败结果照实报告；修改定义只能新增实验版本，不能替换原结果。
8. “指标量级与公开结果一致”不能作为 E00 的唯一通过条件，因为输入尺寸、预训练和增强会改变数值。本版把公开结果复算、训练 smoke test 和同协议相对比较分开记录。

## 1. 研究问题和边界

研究问题是：训练阶段同时看到真实 VDT2048 的 RGB、Thermal、Depth，能否把跨模态结构知识蒸馏到只使用 RGB 的 COD 学生？

训练阶段：

I_rgb, I_T, I_D, y  →  teacher / student

推理阶段：

I_rgb  →  predicted COD mask

固定边界：

- 多模态知识源只使用 VDT2048 官方 Train；官方 Test 永不用于训练、调参或生成教师信号。
- COD 训练只使用 COD10K train 和 CAMO train。
- COD 测试只使用 COD10K test、CAMO test、NC4K、CHAMELEON。
- 不使用 PST900、合成 COD、伪造 T/D，也不使用 COD 测试图像生成教师信号。
- VDT2048 与 COD 数据先做文件名和感知哈希去重检查；若发现重叠，剔除并记录。
- VDT2048 的标签语义是显著目标，不能把 VDT 指标称作 COD 指标。VDT 只用于学习跨模态关系及验证教师是否利用 T/D。

## 2. 数据划分和评价协议

### 2.1 VDT2048

使用官方 Train split；若数据提供场景元数据，则按场景分组后固定拆分为 80% train、20% validation，同一场景不能跨 train/validation。若没有可用场景元数据，则使用固定样本分组并在结果中说明，不能把随机拆分写成场景隔离。官方 Test 只在最终附录报告一次。

每次检查并保存：

- RGB、T、D、mask 是否一一对应；
- 分辨率和有效值范围；
- 空 mask、全黑 Thermal、无效 Depth；
- 场景重复和 train/test 混淆；
- 固定的样本清单和随机种子。

### 2.2 COD

使用 COD10K train 和 CAMO train；从训练部分按图像固定拆出 10% validation，不改变官方 test。NC4K 和 CHAMELEON 只做测试，不参与模型选择。

主结果统一报告：

- S_alpha；
- F_beta_w；
- E_phi；
- MAE。

Boundary-F_w 只作为补充指标，定义为 CamoFormer evaluator 的边界裁剪流程（br-rate=15）后计算的 weighted F，不把它称为独立的标准 Boundary F1。预测图 resize 到 GT 尺寸，GT 永远不修改；CHAMELEON 的尺寸差异按官方 evaluator 处理。

指标依据见 R22–R26；评价脚本版本、commit、预测 resize 插值和阈值策略必须写入实验记录。

### 2.3 统计规则

- screening：每个候选模型 1 个预注册 seed，只用于发现实现错误和估算耗时。
- confirmatory：E00、E10、E13、E20、E21、E24、E26 使用 3 个预注册 seed：20261012、20261013、20261014。
- 主结果报告三种子均值、标准差和 95% bootstrap CI；主指标预先指定为 S_alpha 和 F_beta_w。
- 不要求每个指标都严格变好；结论依据预注册主指标的效应量和置信区间。
- 官方 checkpoint 与同协议重训结果分开列出，必须记录权重版本、输入分辨率、数据增强、后处理和是否使用额外数据。

## 3. 模型固定规则

E00、E00b、E10–E13 和 E20–E26 使用 ResNet-50 encoder、352×352 输入、相同增强和相同训练步数。E01–E05 遵循各自官方实现的 backbone 和预处理，另行记录；若改成 ResNet-50，必须标为 reimplementation variant，不能冒充原论文复现。这里的 U-Net-R50 和 DeepLabv3+-R50 都是为统一容量与输入协议而采用 ResNet-50 encoder 的实现变体；R4/R5 只作为架构原始依据，不能把它们写成原论文默认 backbone 的逐项复现。教师融合模型的 decoder 宽度和训练预算要对齐，并报告 raw 与 parameter-matched 两种比较；不能把增加分支带来的容量优势误报为模态优势。

代码中 `models/triprd_full.py` 和 `run_triprd_ab.py --model-variant resnet50 --pretrained` 是论文比较路径；`models/triprd.py` 的 tiny 版本只用于快速链路检查，不能与本节的 ResNet-50 结果混排。

教师：

RGB encoder / Thermal encoder / Depth encoder
→ fusion or relation module
→ multi-scale decoder
→ mask、relation、boundary、uncertainty heads

学生：

RGB encoder
→ relation predictor、boundary head、uncertainty head
→ COD decoder

COD 微调阶段删除 Thermal encoder、Depth encoder 和多模态融合层；推理只输入 RGB。

## 4. E00 为什么选择 U-Net / DeepLab

U-Net 和 DeepLab 都是通用 dense semantic segmentation 架构，不是专门为 COD 设计的 SOTA。它们在这里的作用是验证数据、损失、训练和评价管线是否正常，并提供一个透明的通用分割参照。

- U-Net：收缩路径提取上下文，扩张路径恢复空间分辨率，skip connection 将浅层定位信息传给解码器。它结构简单、边界行为直观，适合作为最小可复现 sanity baseline。原始依据是 Ronneberger 等人的 U-Net（MICCAI 2015，R4）。
- DeepLabv3+：用 atrous spatial pyramid pooling 捕获多尺度上下文，再用 decoder 细化边界；它比最小 U-Net 更强，适合作为通用语义分割架构敏感性参照。原始依据是 Chen 等人的 DeepLabv3+（ECCV 2018，R5）。

“U-Net/DeepLab”不能继续作为一个实验名。执行规则固定为：

- E00：U-Net-R50，主通用 baseline；
- E00b：DeepLabv3+-R50，可选结构敏感性，不进入主方法门槛；
- E01–E05：COD 专用模型，用来判断方法是否超越任务相关基线。

因此，E00 的选择理由是“简单、透明、可复现的通用分割基线”，不是声称 U-Net 或 DeepLab 在 COD 上最优。

## 5. 实验矩阵与参考依据

### 5.1 RGB COD 基线

| 编号 | 固定实现 | 输入 | 实验性质 | 依据 |
|---|---|---|---|---|
| E00 | U-Net-R50 | RGB | 主通用 baseline | U-Net，R4 |
| E00b | DeepLabv3+-R50 | RGB | 可选结构敏感性 | DeepLabv3+，R5 |
| E01 | SINet-V2 | RGB | COD 专用强 baseline | Concealed Object Detection，R6 |
| E02 | ZoomNet | RGB | 混合尺度 COD baseline | ZoomNet，R7 |
| E03 | C2F-Net | RGB | 上下文跨层融合 baseline | C2F-Net，R8 |
| E04 | BGNet | RGB | 边界引导 baseline | Boundary-Guided COD，R9 |
| E05 | CamoFormer | RGB | Transformer COD baseline | CamoFormer，R10 |

FSPNet（R11）是另一个独立的 Transformer COD 模型，不能并入 BGNet。只有在有足够算力时，才作为补充 baseline E05b；不使用官方 checkpoint 和重训结果混排。

主比较优先顺序：

- screening：E00、E01、E02、E05，单种子；
- confirmatory：E00 进入主确认实验；E01/E02/E05 默认只保留单种子 screening，最多按 10/22 的 go/no-go 将其中一个升级为三种子补充结果；
- E03、E04、E00b、E05b 放入补充或敏感性表。

### 5.2 多模态教师

| 编号 | 固定实现 | 输入 | 实验性质 | 依据 |
|---|---|---|---|---|
| E10 | RGB teacher | RGB | 单模态教师对照 | VDT 数据与单模态对照，R2 |
| E11 | Early fusion | RGB+T+D 五通道 | 输入级融合对照 | RGB-D/T 融合先例，R12 |
| E12 | Late fusion | 三个独立 encoder，decoder 前融合 | 特征级融合对照 | RGB-D/T 融合先例，R12、R13 |
| E13 | E12 + cross-modal relation | RGB-T-D | 本研究关系教师 | Relation Networks、R13；三模态 target/background 组合为本研究新增 |

E13 不能宣称复现某篇已有方法。Relation module 的 token 采样、关系矩阵归一化和复杂度必须固定：从 mask 内部、目标邻域和边界两侧各采样 K 个 token，计算归一化 cosine relation，避免对所有像素做 O(N²) 矩阵。

E10–E13 使用相同 backbone、decoder 宽度、训练步数和增强。E11 的第一层卷积明确如何从 ImageNet RGB 权重扩展到 5 通道。

### 5.3 RGB 学生和蒸馏

| 编号 | 固定实现 | 蒸馏内容 | 实验性质 | 依据 |
|---|---|---|---|---|
| E20 | mask-only student | 仅 mask supervision | control | 无单独方法论文 |
| E21 | Logit KD | 温度化 dense logits | 标准 KD 对照 | Hinton 等，R14 |
| E22 | Feature KD | projection 后的中间特征 | 特征蒸馏对照 | FitNets，R15；Attention Transfer 可作补充 |
| E23 | RKD-style | batch 内距离和角度关系 | 标准关系蒸馏对照 | RKD，R16 |
| E24 | Task-relation KD | 目标内部、目标-背景、边界两侧空间关系 | 本研究任务变体 | Dense prediction relation KD，R17、R18 |
| E25 | Relation + Boundary | E24 加 boundary distillation | 边界消融 | BPKD/BRD，R19 |
| E26 | Full | E25 加 uncertainty gate | 本研究组合 | UGTR、Aleatoric uncertainty，R20、R21 |

E20 的 mask supervision 分阶段定义：VDT 阶段使用 VDT mask，COD 阶段使用 COD mask；不得把两个域的 mask 混称。E23 是标准 RKD 的距离/角度关系，E24 是像素/区域结构关系，二者必须使用独立编号和同等损失预算。

边界标签固定为 GT mask 的 3×3 morphological gradient；boundary head 的输出、BCE 权重和梯度项提前写入 YAML。E26 的 uncertainty 定义为三模态 teacher 概率在 validation 校准后的 disagreement，teacher 输出 detach，门控权重 clamp 到 [0.1, 1.0]，防止蒸馏权重塌缩。计划中的 uncertainty gate 是本研究组合，不是对 UGTR 的直接复现。

### 5.4 真实性和消融控制

| 编号 | 控制 | 固定规则 | 依据 |
|---|---|---|---|
| E30 | Correct pairing | 同一 VDT 样本的 RGB/T/D | 配对多模态学习，R2、R13 |
| E31 | Random pairing | 在同一 split 内固定 seed 打乱 T/D，mask 不变 | 控制实验，无单独方法论文 |
| E32 | RGB-only same-data | 与 Full 使用相同样本数、更新步数和增强 | 数据量控制 |
| E33 | Fixed distillation weight | 用固定权重替代 uncertainty gate | uncertainty 控制，R20、R21 |
| E34 | No relation prior in COD | 关闭 VDT relation prior | 关系先验消融 |
| E35 | T-only teacher | 只输入 Thermal | 模态消融 |
| E36 | D-only teacher | 只输入 Depth | 模态消融 |
| E37 | RGB-T teacher | 双模态 | 模态消融 |
| E38 | RGB-D teacher | 双模态 | 模态消融 |

E30–E38 是 control/ablation，不应伪装成已有算法。它们使用与 E13 相同的 backbone、参数预算和 seeds。E35–E38、T/D 噪声/错位和超参敏感性默认进入补充材料。

## 6. 损失和跨域先验

教师基础损失：

L_teacher = BCE(mask) + lambda_iou IoU(mask)
             + lambda_rel L_relation
             + lambda_bd L_boundary
             + lambda_u L_uncertainty

学生损失：

L_student = BCE(mask) + lambda_iou IoU(mask)
            + lambda_d L_distill

COD 微调主实验先固定 BCE+IoU。SSIM 不作为默认项；如果加入，必须单列为新消融，因为原计划没有足够依据说明 SSIM 对该设置必要。

VDT relation prior 只从 VDT train 提取：对固定 K×K relation matrices 计算 train 均值和标准差，并保存统计量和 projection 版本。COD 阶段使用同一个 projection 将学生关系映射到该空间；如果维度或分布无法对齐，则关闭 E34 的 prior 版本并报告“不可迁移”，不能临时改映射。lambda_prior 在实验开始前固定为 0.05，并线性衰减到 0。

蒸馏的跨域边界必须单独记录：E21、E24、E26 的 logit、relation 和 uncertainty target 只在 VDT train 的 RGB 上由冻结 E13 teacher 生成，学生先完成 VDT RGB 预训练；COD 阶段只加载学生权重并使用 COD train/validation 的 mask 微调，不能把 VDT target 按索引套到 COD 图像。若要在 COD 图像上直接蒸馏，必须另定义可迁移的 prior/projection 并登记为新实验版本。

## 7. 按日执行计划（2026/10/12–2026/11/23）

以下每天单列任务、产出和验收条件，沿用前文实验编号与参考依据。日期按北京时间安排；官方截止日单独标注 AoE。表格描述计划，不代表任务已经完成；完成状态以日志、checkpoint 和 results 中的记录为准。已提前完成的工作只需核对并链接已有产出。

排程规则：

- 长训练可以跨日运行；“启动”不等于当天必须收敛。首次完整训练测得耗时后，按可用 GPU 小时安排队列，不假定有多张 GPU。
- screening 使用 seed 20261012；关键确认实验使用 20261012、20261013、20261014。只有数据、配置、训练预算完全一致时，筛选运行才能计入确认实验。
- E00、E10、E13、E20、E21、E24、E26 的三种子优先保障。E01、E02、E05 各完成一个单种子 screening；在 10/22 根据耗时和预算决定是否把其中一个升级为三种子补充结果，升级与否不改变主方法的确认实验门槛。
- COD10K train 与 CAMO train 按第 2 节组成同一训练池；同一个 checkpoint 在四个测试集评测，不为每个测试集各训一个模型。模型选择只看训练池内的 validation。
- 实验超时先延期可选模型与补充消融；保留关键对照和种子要求。11/08 仍未完成的结果标为缺失，并收缩论文结论，不用单种子或未收敛结果冒充三种子结果。
- 每天启动训练或评价前核对 split manifest、数据 hash、checkpoint 来源和预测目录；官方 VDT/COD 测试集在 11/09 数字锁定前保持封存。

### 第 1 周：数据、评价和基线（10/12–10/18）

| 日期 | 当天任务 | 必须产出与验收条件 |
|---|---|---|
| 10/12 | 创建并核对实验目录、基础 YAML、结果 CSV 与数据协议；归档已有初始化工作。 | 目录和配置可定位；实验编号、seed、checkpoint 路径有记录。 |
| 10/13 | 验收 VDT2048 Train 的 RGB/T/D/mask 配对、位深和有效值；核对场景元数据并建立内部 train/validation 清单。 | 对齐报告、固定样本清单、至少 20 组四联图；人工核对至少 200 组。未完成的真实数据验收单独记录，不以脚本存在代替通过。 |
| 10/14 | 核对 COD10K、CAMO、NC4K、CHAMELEON 的官方 split 和全量配对；建立训练池内的 validation 清单；完成 E00 小批量前后向和过拟合 smoke test。 | 数据报告、train/validation 文件列表、E00 smoke 日志；测试集未混入训练，CHAMELEON 原 GT 保持不变。 |
| 10/15 | 完成并核对统一 evaluator；明确 E_phi 的具体变体、预测 resize、Boundary-F_w 定义和 CSV 字段；启动 E00 seed=20261012 完整训练。 | 评价脚本和说明可用；E00 seed1 日志、首个 checkpoint 和耗时/显存记录。真实预测验收待预测文件到位后补做。 |
| 10/16 | 继续 E00 seed=20261012，并启动 E00 seed=20261013；用已生成预测串联 evaluator；接入 E01、E02、E05 的官方代码或权重来源核对。 | E00 seed1 validation 结果、seed2 启动日志；E01/E02/E05 的版本、配置和权重来源。没有同协议公开预测时，不宣称完成公开数值复现。 |
| 10/17 | 继续 E00 seed=20261013，并启动 E00 seed=20261014；分别启动 E01、E02、E05 的单种子 screening（若官方代码/权重已就绪）。 | E00 seed2/seed3 运行状态、三个 COD baseline 的 screening 日志；任何未启动项标为阻塞，不用未收敛结果填表。 |
| 10/18 | 验收并封存 E00 三种子队列；冻结第一版基线、数据和评价协议；汇总 E01/E02/E05 screening 进度；建立论文提纲。 | E00 三种子 checkpoint/日志清单、baseline 表、算力队列、数据清单版本、论文提纲；不再无记录地修改 split。 |

### 第 2 周：多模态教师（10/19–10/25）

| 日期 | 当天任务 | 必须产出与验收条件 |
|---|---|---|
| 10/19 | 实现并启动 E10 RGB teacher；统一教师 decoder、mask 监督和内部 validation；继续处理 E00 的剩余种子。 | E10 配置、初始日志和耗时；RGB 教师与多模态教师的比较预算明确。 |
| 10/20 | 实现并启动 E11 五通道 early fusion；固定 T/D 归一化和首层权重初始化。 | E11 输入检查、配置和训练日志；RGB/T/D/mask 的空间增强保持一致。 |
| 10/21 | 实现并启动 E12 三分支特征融合；统计各教师参数量和计算量，建立 E13 的同结构对照。 | E12 日志、教师容量表；容量差异有记录，不能直接把 E13 对 E10 的全部收益归因于关系模块。 |
| 10/22 | 实现 E13 relation module；冻结 token 数量、采样位置、归一化和 loss；检查边界头及各模态概率输出；对 E01/E02/E05 作一次 go/no-go，决定是否升级其中一个为三种子补充结果。 | 关系模块配置、关系图和短运行日志、baseline 预算决策记录；前向计算不依赖 GT，GT 仅用于训练监督/采样损失；无 NaN，显存可承受。 |
| 10/23 | 启动 E13 筛选训练；固定 E30/E31 的配对映射、种子和比较方式；整理教师部分方法草稿。 | E13 训练日志、正确配对与随机配对协议、方法草稿；区分训练时打乱与测试时扰动，不混作同一种实验。 |
| 10/24 | 按固定协议运行 E30/E31；在 VDT validation 检查模态贡献、边界预测和 disagreement。E30 与已有正确配对 E13 设置相同时复用结果。 | 配对对照表、样本映射、定性图；任何重训与扰动结果分别标注，保留无提升结果。 |
| 10/25 | 汇总 E10–E13 筛选结果；冻结教师配置并启动/继续 E10、E13 的其余 seeds；只有 E13 screening 已通过 validation 和 cache 检查后，才允许下游学生使用其 target，其他 E13 seeds 可并行。 | 教师 validation 表、checkpoint 清单、蒸馏缓存样例与版本；确认缓存仅来自 VDT train，教师运行不足三种子时明确列为待完成。 |

### 第 3 周：学生蒸馏与配置冻结（10/26–11/01）

| 日期 | 当天任务 | 必须产出与验收条件 |
|---|---|---|
| 10/26 | 固定 RGB 学生结构，冻结已验收的 E13 teacher；启动 E20 mask-only 的 VDT 预训练；生成并校验 VDT relation prior/teacher-target cache 及其到 COD 学生空间的 projection，完成一次泄漏审计。 | E20 配置和日志、E32 控制定义、E13 checkpoint 与 cache manifest、projection 可执行性报告；若 COD 阶段无法从 VDT cache 得到可执行 target，则对应 E21/E24/E26 标记为不可迁移并停跑。 |
| 10/27 | 在 VDT RGB 预训练阶段实现并启动 E21 Logit KD；明确 logits、概率、温度和二分类蒸馏计算；蒸馏 target 只来自冻结 E13 的 VDT 输出/cache，不在 COD 图像上运行教师；禁止教师梯度更新。 | E21 日志和损失检查、target 来源与版本记录；与 E20 使用相同学生及训练预算。 |
| 10/28 | 完成 E22 Feature KD 与 E23 RKD 的实现/小批量验证；有预算则排入筛选队列；补充相关工作草稿。 | 特征投影层、距离/角度定义和验证记录；未做完整训练的模型明确标为未完成，不填写正式性能。 |
| 10/29 | 在 VDT RGB 预训练阶段实现并启动 E24 Task-relation KD；使用已冻结的 E13 relation target/cache，核对教师与学生的空间对应、关系采样和损失规模。 | E24 关系图、配置、cache 版本和训练日志；与 E21 的比较仅改变预定蒸馏内容。 |
| 10/30 | 实现 E25 边界蒸馏；检查 GT 边界生成和教师软边界；开展 E20/E21 的 COD validation 小规模试跑。 | 边界标签与损失样例、COD 迁移入口；试跑只使用 COD train/validation，不查看四个测试集成绩调参。 |
| 10/31 | 在 VDT RGB 预训练阶段实现并筛选 E26 uncertainty gate；用冻结 E13 的校准 disagreement 生成 gate；核对 E33 固定权重对照，验证关系先验的保存/加载与 E34 关闭开关。 | E26/E33/E34 配置、门控分布及 cache 版本；先验仅由 VDT train 统计，不能把尚未实现的先验收益写入结论。 |
| 11/01 | 汇总 validation 结果，冻结学生与 COD 微调配置；启动/继续 E20/E21/E24/E26 的三种子 VDT 预训练；排定 E32/E33/E34 与边界对照队列。 | 冻结配置、种子清单、教师—学生 checkpoint 对应表、剩余 GPU 小时估算；方法草稿与实际实现一致。 |

### 第 4 周：COD 确认实验与消融（11/02–11/08）

| 日期 | 当天任务 | 必须产出与验收条件 |
|---|---|---|
| 11/02 | 对已完成 VDT 预训练的 E20/E21/E24/E26 启动三种子 COD 微调；按依赖继续剩余预训练和教师运行。 | COD 训练日志及初始化来源；每个 seed 的 VDT→COD 链路完整，学生推理输入只有 RGB。 |
| 11/03 | 继续主实验，按已冻结的 validation 规则选择 checkpoint；补齐 E00 与专用基线；锁定进入最终测试的模型清单并签名 test manifest。 | 模型清单、训练曲线、运行状态、manifest 签名；四测试集结果不用于挑 epoch、seed 或损失权重。 |
| 11/04 | 对完成训练并冻结的模型在 COD10K test、CAMO test、NC4K、CHAMELEON 统一生成预测；将预测、checkpoint hash 和 manifest 加密/只读封存，不解封指标。 | 四数据集预测包与封存记录；同一模型只换测试集，不按测试集重新微调或选择模型。 |
| 11/05 | 完成 E32 same-data、E33 固定权重、E34 无先验的主消融；有现成且严格等价的运行时复用并说明。 | 数据量、门控和先验消融表；不足种子数的比较仅作为探索结果，不作稳定提升结论。 |
| 11/06 | 整理 E25 与 E24/E26 的边界对照及 E30/E31 配对结果；只有主实验队列按时完成且有余量时运行 E35–E38。 | 边界与配对表、模态消融状态；没有对应消融时删除该模块的独立贡献宣称。 |
| 11/07 | 计算完成实验的种子均值、标准差与 CI；统计参数量、FLOPs、FPS；生成固定规则下的困难子集和失败案例图。 | 统计表、效率表、定性图及抽样规则；子集规则在查看模型输赢前固定，不挑选有利样本。 |
| 11/08 | 作为训练/评价缓冲日补齐关键缺口；审计数据、种子、权重、预测封存包和表格；只依据 validation 和训练完成状态决定投稿范围。 | 最终实验清单、未完成项、主结论与局限草稿；测试指标仍保持封存，不因结果不理想重定义测试集或新增模块。 |

### 第 5 周：锁实验、完善论文与提交（11/09–11/16）

| 日期 | 当天任务 | 必须产出与验收条件 |
|---|---|---|
| 11/09 | 解封已签名预测包并统一计算 COD10K test、CAMO test、NC4K、CHAMELEON 指标；锁定全部主文实验数字、CSV 和模型版本，从记录自动生成主表与消融表。 | 解封审计、可追溯的数字快照；未完成实验不进入主结论，方法修改停止。 |
| 11/10 | 完成论文注册，核对作者、标题、摘要和投稿系统信息；确认官方注册截止 11/10 AoE。 | 注册记录和信息核对结果；按北京时间当天完成，预留时区余量。 |
| 11/11 | 完成方法、训练阶段、损失及实现细节；把表格和已有实验分析写入正文。 | 方法与实验章节完整草稿；公式、代码和实验编号一一对应。 |
| 11/12 | 完成引言、相关工作和局限；逐条核对引用与贡献表述。 | 全文初稿、参考文献；清楚区分已有方法、本文改动、未支持的假设。 |
| 11/13 | 完成框架图、关系/边界示意和定性结果；核对图中数据来源。 | 最终图稿和 caption；COD 图只展示真实存在的 RGB/GT/预测，VDT 三模态图单独标注。 |
| 11/14 | 内部审阅全文，复核每个数字、结论、种子和图表引用；核对官方格式要求。 | 审阅问题清单及修正稿；没有表文不一致或把筛选结果写成确认结果。 |
| 11/15 | 完成匿名化、版式、页数、PDF 渲染和附件检查；准备上传版本。 | 可提交 PDF、文件校验与备份；图表可读、匿名信息处理完整。 |
| 11/16 | 提交主论文并检查平台生成的最终文件；确认官方论文截止 11/16 AoE。 | 提交回执、最终 PDF 和版本备份；以平台确认收到为完成标准。 |

### 补充材料与归档（11/17–11/23）

| 日期 | 当天任务 | 必须产出与验收条件 |
|---|---|---|
| 11/17 | 整理全部训练细节、超参数、数据划分和软件环境。 | 补充材料的实现细节与数据协议章节。 |
| 11/18 | 汇总已完成的完整种子结果与逐数据集结果。 | 完整结果表；与主文同一运行一致，不挑 seed。 |
| 11/19 | 整理已完成的 E35–E38、噪声/错位和敏感性实验。 | 补充消融表；未运行项说明未完成，不临时增添支持主结论的新实验。 |
| 11/20 | 整理更多失败案例、困难子集图和模态诊断图。 | 有固定选择规则的附加可视化与解释。 |
| 11/21 | 打包代码、配置、评价脚本和运行说明；核查匿名信息及第三方许可证。 | 可复现实验说明与代码包；不包含不应分发的数据或权重。 |
| 11/22 | 核对主文与补充材料的数字、命名、引用和附件；渲染最终 PDF。 | 补充材料候选版和一致性核查记录。 |
| 11/23 | 提交补充材料并归档主文、补充、代码及实验记录；确认官方截止 11/23 AoE。 | 补充材料提交回执及归档清单。 |

AoE 截止换算为北京时间通常是次日约 20:00；正式记录同时保留 AoE 和 UTC 时间。上表把注册、主文和补充材料分别安排在北京时间 11/10、11/16、11/23 当天完成，预留时区与上传缓冲，不把这些内部完成日期误作官方北京时间截止点。

## 8. 预注册的 go/no-go 规则

- 教师是否有效：E13 相对 E10 在 VDT validation 的主指标效应量及 CI；
- 关系是否有效：E24 相对 E21 在 VDT validation/COD validation 的主指标效应量及 CI；
- 跨模态配对是否有效：E30 相对 E31；
- 完整方法是否有效：E26 相对 E20；
- COD 泛化：四个测试集分别报告，不用“至少三个提升”替代完整结果。

如果某个假设不成立，保留负结果并修改结论。只有在预注册的工程错误（数据错配、NaN、评价脚本错误）下才修复；修复后重新运行全部相关 seeds。修改关系、权重或数据划分必须创建新版本，不能覆盖旧结果。

## 9. 论文中应明确的限制

- 学生不能恢复 RGB 中不存在的 T/D 信息；
- VDT2048 是显著目标数据，和真实 COD 存在任务及域差异；
- 多模态教师可能产生负迁移；
- boundary 和 uncertainty head 是设计选择，需要通过消融而不是先验保证有效；
- 公布官方 checkpoint 和重训结果时必须区分数据、预处理和权重来源。

## 10. 实验—参考依据矩阵

R1. CVPR 2027 Dates and Deadlines：
https://cvpr.thecvf.com/Conferences/2027/Dates

R2. Song et al., A Novel Visible-Depth-Thermal Image Dataset of Salient Object Detection for Robotic Visual Perception, IEEE T-Mech 2023；VDT-2048 官方数据仓库：
https://doi.org/10.1109/TMECH.2022.3215909
https://github.com/VDT-2048/VDT-Dataset

R3. Fan et al., Camouflaged Object Detection, CVPR 2020（COD10K、COD 任务和评测起点）：
https://openaccess.thecvf.com/content_CVPR_2020/html/Fan_Camouflaged_Object_Detection_CVPR_2020_paper.html

R4. Ronneberger et al., U-Net: Convolutional Networks for Biomedical Image Segmentation, MICCAI 2015：
https://arxiv.org/abs/1505.04597

R5. Chen et al., Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation, ECCV 2018（DeepLabv3+）：
https://arxiv.org/abs/1802.02611

R6. Fan et al., Concealed Object Detection, IEEE TPAMI 2022（SINet-V2）：
https://arxiv.org/abs/2102.10274
https://github.com/GewelsJI/SINet-V2

R7. Pang et al., Zoom In and Out: A Mixed-Scale Triplet Network for Camouflaged Object Detection, CVPR 2022：
https://openaccess.thecvf.com/content/CVPR2022/html/Pang_Zoom_in_and_Out_A_Mixed-Scale_Triplet_Network_for_Camouflaged_CVPR_2022_paper.html

R8. Sun et al., Context-aware Cross-level Fusion Network for COD（IJCAI 2021；后续有 TCSVT 扩展），C2F-Net：
https://www.ijcai.org/proceedings/2021/0142.pdf
https://arxiv.org/abs/2207.13362
https://github.com/Ben57882/C2FNet-TSCVT

R9. Sun et al., Boundary-Guided Camouflaged Object Detection, IJCAI 2022：
https://www.ijcai.org/proceedings/2022/186
https://arxiv.org/abs/2207.00794

R10. Yin et al., CamoFormer: Masked Separable Attention for Camouflaged Object Detection：
https://arxiv.org/abs/2212.06570
https://github.com/HVision-NKU/CamoFormer

R11. Huang et al., Feature Shrinkage Pyramid for Camouflaged Object Detection with Transformers, CVPR 2023（补充 baseline）：
https://openaccess.thecvf.com/content/CVPR2023/html/Huang_Feature_Shrinkage_Pyramid_for_Camouflaged_Object_Detection_With_Transformers_CVPR_2023_paper.html

R12. Park et al., RDFNet: RGB-D Multi-level Residual Feature Fusion for Indoor Semantic Segmentation, ICCV 2017（多层 RGB-D 融合）：
https://openaccess.thecvf.com/content_ICCV_2017/papers/Park_RDFNet_RGB-D_Multi-Level_ICCV_2017_paper.pdf

R13. Zhou et al., Channel and Spatial Relation-Propagation Network for RGB-Thermal Semantic Segmentation（CSRPNet）：
https://arxiv.org/abs/2308.12534

R14. Hinton, Vinyals, Dean, Distilling the Knowledge in a Neural Network：
https://arxiv.org/abs/1503.02531

R15. Romero et al., FitNets: Hints for Thin Deep Nets：
https://arxiv.org/abs/1412.6550

R16. Park et al., Relational Knowledge Distillation, CVPR 2019：
https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.html

R17. Liu et al., Structured Knowledge Distillation for Dense Prediction：
https://arxiv.org/abs/1903.04197

R18. Yang et al., Cross-Image Relational Knowledge Distillation for Semantic Segmentation, CVPR 2022：
https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_Cross-Image_Relational_Knowledge_Distillation_for_Semantic_Segmentation_CVPR_2022_paper.pdf

R19. Boundary/feature distillation 依据：Boundary-aware/Pixel-level Knowledge Distillation（BPKD）与 Boundary and Relation Distillation：
https://arxiv.org/abs/2306.08075
https://arxiv.org/abs/2401.13174

R20. Yang et al., Uncertainty-Guided Transformer Reasoning for Camouflaged Object Detection, ICCV 2021：
https://openaccess.thecvf.com/content/ICCV2021/html/Yang_Uncertainty-Guided_Transformer_Reasoning_for_Camouflaged_Object_Detection_ICCV_2021_paper.html

R21. Liu et al., Modeling Aleatoric Uncertainty for Camouflaged Object Detection, WACV 2022：
https://openaccess.thecvf.com/content/WACV2022/html/Liu_Modeling_Aleatoric_Uncertainty_for_Camouflaged_Object_Detection_WACV_2022_paper.html

R22. Fan et al., Structure-Measure: A New Way to Evaluate Foreground Maps, ICCV 2017：
https://openaccess.thecvf.com/content_iccv_2017/html/Fan_Structure-Measure_A_New_ICCV_2017_paper.html

R23. Margolin et al., How to Evaluate Foreground Maps?, CVPR 2014（weighted F-measure）：
https://openaccess.thecvf.com/content_cvpr_2014/papers/Margolin_How_to_Evaluate_2014_CVPR_paper.pdf

R24. Fan et al., Enhanced-alignment Measure for Binary Foreground Map Evaluation, IJCAI 2018：
https://www.ijcai.org/Proceedings/2018/97

R25. CamoFormer 官方评测实现（Boundary-F_w 的工程依据）：
https://github.com/HVision-NKU/CamoFormer

R26. CEKD：Cross-Modal Edge-Privileged Knowledge Distillation for Semantic Scene Understanding Using Only Thermal Images，作为“训练时特权模态、推理时单模态”的范式参考：
https://ieeexplore.ieee.org/document/10049523/

## 11. 每个实验必须记录

experiment_id、dataset、split、backbone、input_size、seed、teacher_type、student_type、distillation_type、loss_weights、best_epoch、S_alpha、F_beta_w、E_phi、MAE、Boundary-F_w、params、FLOPs、FPS、checkpoint、代码 commit、预处理版本、评测脚本版本。

所有主表从 CSV 自动生成。任何结果不能手工抄写或只保留最好 seed。
