# CVPR 2027：TriPRD 实验与投稿执行计划

> 研究路线修订：2026-10-16。以下新增决策优先于旧的逐波队列安排。目标明确为 **TriPRD 自身机制带来可验证的 COD 增益并冲击 SOTA**；采用标准预训练主干可以，完整移植 ZoomNeXt 再添加通用 KD 不作为主方法。历史实验配置继续保留，尚未实施的方案不写成已有贡献。

## 2026-10-16：当前最高优先级

第二波 transfer 已同步四份 comparison：A_schedule 在 seed13 相对 A_control 的宏 Sm 提升 0.002693，seed12 提升 0.009373；seed13 CAMO Sm 下降 0.002951、MAE 增加 0.001992，因此不能宣称两来源全面改善。B_encoder/C_encoder 的 Sm 为 0.780768/0.776117，相对同 seed12 的 A_schedule 分别低 0.026868/0.031518，且未改善各自 full-transfer 的 Sm。四组均完成 100 epoch 并跳过测试；这仍是内部验证结果。本地没有对应 split manifest，未独立核验成员。

当前代码是均值融合教师、融合特征的自相似关系 KD 和监督 COD 微调，原设计中的显式跨模态关系教师、RGB 关系预测解码、边界可靠性和 COD prior 尚未全部落地。下一版只先验证一条机制：**学习真实跨模态空间关系，让 RGB 学生预测关系并参与自身解码，仅迁移关系分支而保留通用主干的初始化。** 详细边界及否证条件见 [Paper Scope](paper_scope.md)。

- 第一个实现包：显式跨模态关系教师；一个尺度的 RGB 局部关系预测/残差聚合；VDT 阶段冻结学生主干及 BN 状态；仅保存/迁移关系分支，源分割头与 COD 分割头分开。新边界可靠性、uncertainty 和 COD prior 暂不一起加入。
- 第一轮四卡筛选：同结构学生无 VDT；VDT RGB 监督；容量匹配 RGB 教师关系 KD；真实 RGB/T/D 教师关系 KD。源数据、源训练步数、目标训练配方及种子明确匹配。此队列尚未实现，不是当前 --phase distill 的含义。
- 净增益成立后补 GT-only 关系、原融合 Gram KD、打乱关系目标、错误 T/D 配对重训、关系是否进入解码及匹配容量对照；最后确认第二主干与配对多种子。
- 标准 ResNet-50/PVTv2 可作为共同特征提取器；TriPRD 的关系机制、学生解码和跨任务训练流程需独立验证。ZoomNeXt、CODiff 等作为强对照；引用其结果或组件时明确来源。
- 保留既有训练配方/LR 排查作诊断，但不继续以旧实现的大规模调参替代核心机制验证。强学生上的无净增益、RGB 教师与多模态教师无差异、打乱关系后收益不变，均触发研究假设复查。

内部检查点建议：10/19 前完成最小实现与 GPU 验收；10/24 前检查匹配基线上的迁移收益；10 月底冻结有效配置，留时间做确认实验与论文。均为计划而非已完成状态或性能承诺。最终全 4040 张训练、固定预算/选权重规则和完整测试集覆盖校验仍需实现；内部约 0.81 的宏 Sm 不作为论文 SOTA 差距。

> 更新：2026-10-15，Asia/Singapore（UTC+8）。可用设备：4 张 RTX 4090；每日可用时长尚未确认。本版取代此前的逐日长矩阵，按 E001 证据及当前代码能力安排。研究设想见 [paper_scope.md](paper_scope.md) 和 [research_plan.md](research_plan.md)，其中尚未实现的模块不视为已有贡献；实验事实以 [E001](experiments/E001_triprd_full.md) 及后续记录为准。

## 1. 截止时间与当前判断

2026-10-15 已核验 [CVPR 2027 官方 Dates](https://cvpr.thecvf.com/Conferences/2027/Dates) 与 [Call for Papers](https://cvpr.thecvf.com/Conferences/2027/CallForPapers)：

| 事项 | 官方日期 | 本项目内部完成日期（UTC+8） |
|---|---|---|
| Paper registration / enrollment | 2026-11-10 AoE | 11/09 完成作者信息和注册 |
| 主论文 | 2026-11-16 AoE | 11/15 完成上传和回执检查 |
| 补充材料 | 2026-11-23 AoE | 11/22 完成上传和回执检查 |
| Reviews / final decisions | 2027-01-25 / 2027-02-25 AoE | 根据官方后续通知安排 |

AoE 为 UTC−12，晚于 UTC+8 20 小时；11/10 23:59 AoE 对应本地 11/11 19:59。主文和补充以官方平台显示的具体时刻为准，内部日期预留上传缓冲。距主文官方日期约 32 天，实际用于方法筛选和确认的窗口约三周，最后一周留给论文核验。作者 OpenReview 资料现在检查；官方提示无机构邮箱的新资料审核可能需要两周。

**现在可以冲刺实验闭环，尚不能判断已有 CVPR 主会所需的创新和实验证据。** E001 中 D 相对 A 的 Fw、mean E 在四集全部下降，完整蒸馏的稳定收益尚不成立。学习率调度、选模修正、损失数值稳定性属于必要工程改进；只有可复现的跨模态、跨任务结构迁移机制和公平因果对照，才可能支撑研究贡献。投稿门槛在第 8 节，不以“实现了三个模块”代替证据。

## 2. 固定研究边界与新近工作

研究问题收敛为：**真实 RGB/T/D 的显著目标知识，哪些结构可以迁移到仅 RGB 输入的伪装目标检测，何时会造成负迁移，以及怎样避免它。** 不能把“三模态 + 关系 + 边界 + 门控”的组合本身当作已证实的新颖性。

VDT2048 是显著目标检测数据，不是三模态 COD 数据；使用其官方 Train 的配对 RGB/T/D/mask，官方 Test 不参加训练和选模。[VDT 官方数据仓库](https://github.com/VDT-2048/VDT-Dataset)给出任务及数据来源。COD 使用 COD10K train 与 CAMO train；COD10K test、CAMO test、NC4K、CHAMELEON 仅做报告。学生在 COD 微调及推理均只输入 RGB；不生成伪 T/D，不把 VDT target 按索引用于 COD。当前实现没有 COD 关系先验损失，因此不能在方法图中画成已运行路径。

截至 2026-10-15 的相关性核对如下；这不是穷尽检索，也不是声称本方法首次提出这些问题。

| 已发表工作 | 与本项目直接重合处 | 本项目必须回答的区别 |
|---|---|---|
| [C²KD，CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Huo_C2KD_Bridging_the_Modality_Gap_for_Cross-Modal_Knowledge_Distillation_CVPR_2024_paper.html) | 分析跨模态 KD 的模态不平衡和软标签错位，采用选择性及双向蒸馏 | 为什么本项目的结构迁移优于普通/选择性蒸馏？dense binary COD 改编需单列实现，不能把一个门控称为完整 C²KD 复现 |
| [SAM-DSA，ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Liu_Improving_SAM_for_Camouflaged_Object_Detection_via_Dual_Stream_Adapters_ICCV_2025_paper.html) | RGB-D COD、双流 adapter、模型和模态蒸馏 | 原文 RGB-D 输入与深度分支要区分于本项目推理只运行 RGB 学生；预训练、深度生成及模型规模必须列出，不能直接把两个表当同预算对照 |
| [DepthSAM，CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Han_Beyond_Appearance_Camouflaged_Object_Detection_via_Geometric_Structure_CVPR_2026_paper.html) | 明确研究单目深度与 COD 的任务错位，结合几何及语义适配 | “深度提供结构”或“域差导致负迁移”已不足以单独构成创新；需证明真实多传感器知识在不保留深度推理路径时仍产生收益，并报告整个推理图的成本 |
| [MLLMs-Guided Graded KD，CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Beyond_Weak_Supervision_MLLMs-Guided_Graded_Knowledge_Distillation_for_Unsupervised_Camouflaged_CVPR_2026_paper.html) | 图像/像素级可靠性筛选及 KD 已用于 COD | 该工作是无监督 COD，与本项目全监督设定不同；列相关工作及监督资源差异，不混入同监督主排名，也不能把不确定性加权本身称为首次 |

旧的 SINet-V2、ZoomNet、BGNet、CamoFormer 仍是有用参照，但不足以覆盖投稿时的近作。主表至少增加上述 2025/2026 COD 方法的设定说明；优先复算公开 checkpoint，若无可用实现，引用原文数值并明确“论文报告值”，不写成同协议重训。强 CNN（ZoomNet 或 SINet-V2）与一个 Transformer COD 实现至少各有统一 evaluator 的结果；其中一个用于验证我们的训练方法能否迁移到第二种学生结构。

## 3. 当前状态：已有事实、工程实现与研究待办

| 项目 | 状态与依据 |
|---|---|
| A/B/C/D、ResNet-50、RGB-only COD 路径 | E001 已运行；16 条正式结果和 800 条 epoch 历史已核对；单 seed，不能给训练方差结论 |
| 教师结构 | 当前是 RGB/T/D 三编码器特征均值融合；关系约束发生在蒸馏损失中。没有独立的 learned relation teacher、learned uncertainty predictor 或 COD prior，不沿用旧设计稿中的已实现措辞 |
| E001 诊断 | 边界 KD 标量占总损失首轮约 78.95%、最佳 VDT epoch 46 约 63.44%；gate_mean 约 0.99942。提示优先消融，不等价于梯度贡献或因果证明 |
| 本轮工程入口 | 原尺寸分来源验证、Sm 选模、split/train seed 分离、教师/VDT 权重复用、encoder-only 迁移、COD 分组学习率和调度、skip-test/eval-only、可独立关闭 KD 分项、概率边界 MSE、关系区域/采样修正、门控统计。最终接口及测试以当前 runner 和优化队列为准 |
| 工程收益 | 真实 4090 smoke、四卡 100 epoch COD 筛选及第二波 transfer 已完成。A 的两 seed 宏 Sm 配对增益为 +0.009373/+0.002693，但 seed13 CAMO Sm/MAE 退步；encoder-only 未恢复 B/C 的 Sm。最终测试及完整方法确认仍待运行，详见 [优化实验汇总](experiments/optimization_summary.md) |
| 因果控制 | RGB 教师、容量匹配 RGB 教师、固定随机配对重训、feature KD/结构 KD 强对照、第二骨干迁移仍待实现/运行；不把四个 D 变体替代这些控制 |
| 原始证据 | E001 完整 run.log、split manifest、权重和预测仍待服务器核对；comparison.json 和原始日志只保留本地，不纳入 Git |

本轮关系修正改变了目标区域划分及跨类别关系，不只是提速；必须与 legacy 关系单独对照。`prob_mse` 把边界 logits 变为概率后比较，是数值和目标空间的变更，不预设一定更优。门控保留现有 disagreement 定义用于可归因消融，不因为均值接近 1 就立即增加另一套复杂门控。

## 4. 先冻结验证和复用协议

1. 新一轮固定 `split_seed=20261012`；训练 seeds 为 `20261012, 20261013, 20261014`。同一比较共享完全相同的清单，而模型初始化、loader shuffle、增强随机性随 train seed 变化。禁止三 seed 同时改变数据划分。
2. 本轮采用 runner 支持的 `val_fraction=0.2`，对 VDT 与 COD 的各来源分别拆分；这修订了旧纸面计划中的 COD 10%。写入配置及清单，E001 实际比例和清单未核验前不能宣称与 E001 严格同划分。若存在场景元数据，应先完成场景隔离后再冻结；没有元数据则如实称固定图像划分。
3. 验证用原 GT 尺寸和与正式评价一致的预测处理，按 COD10K-val/CAMO-val 各自均值、再来源等权汇总。主选模指标固定 Sm，Fw、mean E、MAE 共同报告；不在看到结果后换成有利指标或按不同测试集选不同 checkpoint。默认每 5 epoch 正式验证，各组使用相同评估时点；快速 SoftDice 只作训练诊断。
4. 从未见过测试结果不再是本项目的事实：E001 四集已查看。本轮新的筛选只看固定 validation，`--skip-test`；冻结候选、种子和选模规则后统一测试。报告中区分探索性 E001 和后续确认实验，不把公开测试称为完全盲测。
5. 复用旧教师/VDT 权重前核对来源 checkpoint 的架构、epoch、预训练配置、文件 hash 及 split manifest。若旧权重见过新的 VDT-val，不能用这份 validation 宣称教师泛化；清单无法恢复时，旧权重只用于工程筛选，确认阶段从新冻结清单重训。
6. 同一教师的三个学生 seed 只估计**条件于该教师的学生训练方差**。论文若报告整个方法的随机性，三个 seed 都要训练各自教师及学生；同 seed 的 C/D 共用同一教师以减少无关差异。不能把复用一次教师写成三次独立全流程。
7. 记录所有实验的完整命令、代码版本、实际 batch、学习率曲线、监督/KD 分项、门控分位数、源数据清单、GPU 型号、峰值显存和 epoch 时间。预算对齐包含学生在 VDT/COD 的更新步数、数据和增强；教师额外计算单独列出。

## 5. 上一版四卡队列：已完成筛选与未执行消融

本节保留上一版可复现的队列和对照设计。COD/transfer 已完成，旧 KD 消融尚未执行；下一版实施顺序以本文开头的 2026-10-16 决策为准。

使用 [tools/run_triprd_optimization.py](tools/run_triprd_optimization.py) 先生成 dry-run 队列，检查源 checkpoint 和配置后再加 `--execute`。每张 4090 独立运行一个任务，独立输出目录、日志和 seed；当前不假设 runner 已实现 DDP。数据读取并发和 CPU evaluator 可能成为瓶颈；现有 archive-backed reader 强制 workers=0，不能直接增加 workers。运行命令及完整参数见 [四卡优化实验说明](tools/TRIPRD_OPTIMIZATION.md)。

### 队列一：COD 配方，固定 seed 筛选

| GPU | 任务 | 目的与约束 |
|---|---|---|
| 0 | A_control | 旧优化器配方 + 新统一 Sm 选模，隔离选模协议与训练配方变化；不是 E001 数字的严格重跑 |
| 1 | A_schedule | encoder lr scale=0.1、cosine、5 epoch warmup |
| 2 | B_schedule | 复用 B/VDT 权重，和 A_schedule 相同 COD 配方；检查同数据 RGB 预训练 |
| 3 | C_schedule | 复用 C/VDT 权重，同 COD 配方；检查 logits KD |

四个任务均不输出测试预测。A_schedule 相对 A_control 是训练配方整体筛选，不可将收益单独归因于 cosine 或分组 lr；若论文需要归因，再分别关闭。

2026-10-15 本波 100 epoch 已完成：Sm 选模结果分别为 A_control 0.798263、A_schedule 0.807636、B_schedule 0.781063、C_schedule 0.776925。随后 `--phase transfer` 也已完成：B_encoder/C_encoder（seed12）为 0.780768/0.776117，A_control_repeat/A_schedule_repeat（seed13）为 0.807635/0.810328。第二波配置复用第一波 COD 清单；实际成员尚未独立核验。这些均为内部验证分数，完整历史命令见 [优化说明第 4 节](tools/TRIPRD_OPTIMIZATION.md#4-第二波迁移范围与第二种子复验已完成)。

### 队列二：旧 KD 组件（已实现、未执行）

| GPU | 任务 | 固定内容 |
|---|---|---|
| 0 | C_logit | 相同教师、相同 VDT 训练预算，普通 logits KD |
| 1 | D_relation | logits + uniform 关系，边界 KD=0、gate=none |
| 2 | D_boundary | 在 D_relation 上加概率边界 MSE，边界 KD 权重 0.1、gate=none |
| 3 | D_gate | 与 D_boundary 完全一致，仅启用 disagreement gate |

若 COD 队列最终采用 encoder-only 或其他非默认配方，先把它固化到四个 KD 任务的命令中，再执行；不要把不同 COD 配方的结果用于组件归因。这里的 0.1 是下一轮预先固定的候选，不是已验证最优权重。必要时增加一个 legacy relation/原 logits 边界对照；保留失败配置结果。只有 relation、boundary、gate 的增量分别成立，才保留相应主张；无效组件从主方法删去。

### 队列三：必须补齐的因果控制

以下尚不是自动队列已经支持的实验，需先实现并登记配置。先单 seed，再对决定主结论的对照追加确认 seeds：

| 优先级 | 对照 | 排除的解释 |
|---|---|---|
| 必须 | 同新协议重训 B：相同 VDT RGB、监督、初始化、更新步数 | 额外数据、额外训练、旧权重复用造成差异 |
| 必须 | RGB teacher → RGB student，使用与多模态 KD 相同预算/目标 | 有效的是普通 teacher KD，而非额外模态 |
| 必须 | 同三分支结构的 RGB/RGB/RGB 教师对照；记录参数和 FLOPs | 原教师约 75M、学生约 28M，容量/集成收益误当模态收益 |
| 必须 | 固定 seed 在 VDT train 内打乱 T/D 配对并重训教师、再蒸馏学生 | 多模态配对结构是否必要。测试时扰动已训教师只能诊断依赖，不能替代此重训控制 |
| 必须 | 一个常规 feature KD 或 dense relation KD 强对照 | 简单 feature matching 已足够，而所谓任务关系没有独立价值 |
| 主张决定 | RGB-T、RGB-D 教师及学生，分别去掉 T 或 D | 只有要主张三模态均不可或缺时才必须；删减该主张不能省掉多模态对 RGB 的核心对照 |
| 迁移证据 | 在第二种 COD 学生结构上复用冻结的方法和损失设置 | 收益是否只依赖自建 ResUNet；不重新用测试集调权重 |

四卡可分配为 RGB teacher、RGB×3 teacher、随机配对 teacher、B/强 KD 对照。教师训练完各自排学生依赖队列。early-fusion、T-only、D-only、大范围噪声和 20 多个旧矩阵项降为补充优先级；不能挤占上述因果控制与三 seed。

## 6. 预算与三种子确认

E001 记录的 800 个 epoch 总时间约 **20.75 小时**，其中每次 100 epoch COD 约 3.30–3.57 小时、VDT 学生约 1.60–1.81 小时、教师约 1.92 小时。原设备及完整运行配置未确认，且这些时间不包含全部评估与导出，**不能直接当作 4090 的 ETA**。

上一版真实 4090 smoke 及两波完整筛选已完成，第二波四任务并行约 4 小时 24 分钟。下一版结构变化后重新做 smoke 和短程测速，记录原尺寸评价增加的 CPU/IO 耗时，不直接沿用旧模型 ETA。若每卡每天可用 H 小时，四卡可用预算约 `4 × H × 天数` GPU 小时，并预留约 30% 给失败重跑、评估和资源争用。未确认每天 24 小时可用，不用 96 GPU 小时/天作为已知条件。

最小确认集是 A、B、C、最终胜出方法，以及决定跨模态/关系主张的控制。先完成 A/B/C/胜出方法的 3 seeds，不能只给 ours 三种子而基线挑最好 seed。独立完整 A/B/C/D 三 seed 共约 24 个阶段：3 个教师、9 个 VDT 学生、12 个 COD 学生；同 seed 的 C/D 共享教师。该计数不包含 RGB/容量/随机配对和强 KD 对照，须另留预算。

确认排队建议：GPU 0–2 分别负责一个 seed 的完整依赖链，GPU 3 跑待补控制、强模型复算和计时。保存每次运行的真实可用时间，不为了填满四张卡同时重复训练相同教师。screening 只有数据、代码版本和全部配置与冻结版本一致时才可计入确认集。

主表报告每个数据集、每个方法的三 seed 均值与样本标准差，同时给出逐 seed 数字及配对差值。逐图 paired bootstrap CI 可表征固定模型在测试样本上的差异，不能替代训练 seed 方差；3 seeds 的置信区间不稳定，不能仅以窄的逐图 CI 宣称跨训练随机性的显著性。若做分层 bootstrap，明确同时重采样 seed 和图像，并说明 seed 数少的限制。

## 7. 倒排到主文的执行节点

| 时间（UTC+8） | 必须交付 | 四卡安排与决定 |
|---|---|---|
| 10/15–10/17 | smoke、数据/划分核对、10 epoch 耗时和显存、两套入口 dry-run；作者资料检查 | 四卡前先单卡验收；不把单元测试当实测成功 |
| 10/18–10/21 | COD 配方四任务；必要时 full/encoder 迁移对照；新近工作及强基线协议表 | 不重训旧教师；只在 validation 做选择；10/21 冻结 COD 配方 |
| 10/22–10/25 | KD 四任务；补 legacy 对照；RGB teacher、容量及随机配对入口至少各有可运行 smoke | 10/25 方法门槛：有没有超越 B/C 的可解释增量；决定删去无效组件 |
| 10/26–10/30 | 必要因果控制；完整三 seed 确认开始；固定第二学生实验 | 10/30 决定主会主线继续或停止扩张，不新开一套无关机制 |
| 10/31–11/04 | A/B/C/胜出方法及决定性控制的确认结果；强 CNN/Transformer 统一评估；论文方法和结果初稿 | 11/04 数值/配置冻结；没有证据的主张删除 |
| 11/05–11/08 | 统一最终测试、逐图差异/失败案例、速度/显存测量、全表校验 | 测试失败只修工程错误并重算受影响项，不根据测试表现改 loss |
| 11/09–11/10 | 论文注册、完整初稿和引用审阅 | 11/09 内部注册；官方11/10 AoE |
| 11/11–11/15 | 主文审阅、图表核对、匿名化、格式核验、上传回执 | 停止新增筛选实验，11/15 完成内部上传 |
| 11/16 | 仅处理提交校验和可明确定位的错误 | 官方主文截止 11/16 AoE |
| 11/17–11/22 | 补充训练细节、全部 seed、失败案例、可复现代码包 | 主结论必须已有主文证据；11/22 内部上传 |
| 11/23 | 补充回执与最终归档 | 官方补充截止 11/23 AoE |

这些是内部项目期限，不是保证完成所有项目。资源不足时先删可选模型和广泛超参搜索，保留公平基线、因果控制和复验；不把未运行或部分收敛实验写入主结果。

## 8. 继续或止损标准

以下是团队预先采用的研究决策规则，不是 CVPR 官方录用标准。阈值在新队列运行前登记；每次改动都保存旧版和理由。

- **10/21，工程门槛**：统一验证链路完整、预算可追溯，无泄漏/错配/NaN。A 新配方没有可信提升时保留控制配方；不得为了“优化成功”只挑较好数据集。
- **10/25，方法门槛**：候选相对同预算 B 和 C 在来源等权 validation Sm 的增量为正，Fw 不出现相反的明显退步；至少两个训练 seeds 方向一致后才投入完整确认。为避免追逐第4位小数，可预设 Sm 最小实用增量 0.002、任一来源下降容忍 0.002；这是本项目资源筛选阈值，不是统计显著性或论文录用标准。若必须放宽，先登记新版本，不能看测试决定。
- **10/30，机制门槛**：正确多模态配对相比 RGB 教师、同容量 RGB 教师、随机配对控制仍有可复现收益；任务关系相比 logits/常规 KD 有独立价值。若仅靠调度或额外预训练提升，就收缩为工程/迁移分析，停止“三模态关系新方法已经成立”的主张。
- **11/04，投稿门槛**：关键三 seed、主张所需因果控制、近作对照、至少一个迁移/机制证据齐备。只有单 seed 小幅涨点、只有 CHAMELEON 76 张提升，或仍不能排除容量/数据量因素时，不继续扩张实验和包装主会贡献；保留全部负结果，明确现有证据不足并重新评估投稿目标。

最终四个测试集全部报告，不能用“至少三个上涨”替代效应量、代价和失败分析。模块无增益时可删除模块而保留成立的研究结论；若整体不优于公平 A/B/C，保留这次证伪结果，不用挑种子、测试调参或补充材料掩盖。

## 9. 投稿与归档核验

论文贡献必须逐条对应实验证据：真实多模态贡献、关系结构贡献、跨任务迁移贡献各自有哪些对照；对尚无证据的“可靠性”“冲突感知”只写假设。当前架构图应画三编码器均值融合教师和 RGB 学生，把 GT 仅用于训练采样/监督，不能画推理时读取 GT。

主结果同时列出训练数据、是否额外 VDT、监督级别、backbone、输入尺寸、预训练、推理输入/额外网络、参数、FPS/显存、训练预算和 evaluator 版本。RGB-D/深度模型、foundation model、无监督方法分设定比较，引用值与本地复算值明确标注。Boundary_F 是边界区域 weighted F，不写为标准 contour F1；E_phi 固定报告 mean 版本，必要时另列 adaptive/max。

每个新 run 按 [experiments 规范](experiments/README.md) 记录 hypothesis、唯一变量、配置、失败及保留/淘汰决定。Git 保存代码、配置、记录、索引和小型正式指标；原始 comparison/log、大型权重和预测留服务器/本地，以 hash 和路径追溯，继续遵守不提交大型原始文件的要求。

官方 CFP 的 contemporaneous-work 界线为 2026-09-15，但影响了本文的工作仍应引用讨论，不能用该规则省略相关性说明。匿名格式、页数、作者要求和政策以提交时的 [官方 CFP](https://cvpr.thecvf.com/Conferences/2027/CallForPapers) / Author Guidelines 为准；不根据往年模板自行推定。
