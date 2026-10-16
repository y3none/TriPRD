# TriPRD 相关 CCF-A 论文调研

整理日期：2026-10-16（Asia/Singapore）。共 22 篇；覆盖 2016–2026 年，以 2023–2026 年工作为主。

本文依据当前 `paper_scope.md`、`models/triprd_relation.py` 和 `tools/TRIPRD_RELATION.md` 整理。目的不是堆叠模块，而是寻找能够检验或改善 **真实 RGB/T/D 源域关系 → RGB-only COD** 的方法。论文机制与针对 TriPRD 的研究建议分开叙述；所有建议均未在本项目验证。

**实验状态更新（同日归档）：**首轮单种子、100 轮 A/B/C/D 内部验证已完成，D 的 Sm 未超过 A/B/C，见 [E011–E014 汇总](../experiments/relation_summary.md)。以下文献与提案作为独立研究笔记保留，不构成当前方案有效性的证据；下一步优先诊断教师是否依赖其关系分支。正式测试、全训练池复训及多种子确认仍未完成。

针对“怎样构建 SOD→COD 迁移方法”，另有[方法构建补充笔记](sod_to_cod_method_design_2026-10-16.md)：新增 16 篇论文，重点讨论任务条件读出、多类型结构传播、知识分解和跨任务关系迁移。

## 1. 核心判断与阅读顺序

最值得借鉴的是三点：**筛选 RGB 能够学习的教师知识；区分有价值的模态差异与错误关系；验证结构知识在 SOD→COD 后是否仍有效。** 当前最小 A/B/C/D 已完成首轮内部验证；应先根据负结果做机制诊断，再逐项测试这些思路。

| 优先阅读 | 论文 | 解决我们什么问题 | 用途 |
|---|---|---|---|
| 1 | MFH，ICLR 2023，§3.2 | 教师强，为什么学生和 COD 不一定提高？ | 检查关系的 RGB 可预测性；负迁移解释框架 |
| 2 | A2dele，CVPR 2020，§3.1 | 深度训练、RGB 推理已有怎样的先例？ | 必须对比的特权信息蒸馏基线 |
| 3 | QSF-Net，TIP 2024，§3.4 | 同一 VDT2048 数据中 T/D 何时有用、何时误导？ | 按模态、按区域评估教师质量 |
| 4 | C2KD，CVPR 2024，§3.3 | 不适合学生的跨模态监督如何处理？ | 选择性蒸馏；避免全图机械拟合 |
| 5 | BSD，ICCV 2023，§5.4 | 教师是否在所有位置都比学生可靠？ | 有标注的区域质量筛选 |
| 6 | VRM，ICCV 2025，§5.5 | 错误关系是否会传播或造成过拟合？ | 关系损失加权或剪枝对照 |
| 7 | DepthSAM，CVPR 2026，§4.1 | 几何结构辅助 COD 的最新接近工作是什么？ | 新颖性比较与任务适配思路 |
| 8 | PopNet，ICCV 2023，§4.2 | 什么样的中间结构比直接迁移特征更有意义？ | 目标内、跨边界关系的诊断 |

**对当前项目的判断：**“多模态教师、RGB 学生”“深度辅助 COD”“关系 KL”“冻结主干训练小模块”分别已有先例。较值得检验的组合是：从真实且配对的 T/D 中提取局部关系，让 RGB 学生在前向解码中使用这些关系，并在源阶段隔离主干、只迁移关系模块以跨越 SOD→COD。这是一项待验证研究主张，本文不能据此判定首创或达到 SOTA。

## 2. CCF 分级、核实口径与项目背景

### 2.1 分级采用 2026 年第七版

CCF 在 2026-03-31 发布第七版目录，公告注明 4 月 9 日修正一处勘误。本次按该版统一筛选，不把“计算领域高质量科技期刊”的 T1 等级当作 CCF-A。会议条目仅收正式主会长文，不以 workshop 或未录用预印本充数。[CCF 官方发布说明](https://www.ccf.org.cn/Academic_Evaluation/By_category/)

本清单涉及的 CVPR、ICCV、ACM MM、NeurIPS、ICLR、TPAMI、TIP、TMM 均在该版 A 类中。需注意 **ICLR 和 TMM 在新版为 A；IJCAI、ECCV 为 B**。目录正文从 CCF 原始文件的[上海大学课题组镜像](https://scdm-shu.github.io/ccf/2026-CCF-Ranking.pdf)读取：PDF 第 44 页为 TIP/TMM、第 47 页为 ACM MM、第 51 页为 TPAMI、第 57–58 页为人工智能会议。官网分类子页可能仍保留旧表，不据其旧条目覆盖新版 PDF。此处使用的是当前版本筛选口径，不代表各单位对往年成果的追溯认定规则。

发表信息优先使用 CVF、ACM、NeurIPS、OpenReview 正式论文页；期刊结合 DOI、作者正式代码库及作者稿核对。每项都附来源。核实深度包括正式题名、发表渠道、年份与主要机制，对直接相关论文进一步检查正文、补充材料或官方代码；没有声称全部复现或逐页精读。该清单是有针对性的检索结果，并非截至当日全部相关工作的穷尽综述。

### 2.2 当前方法，而非旧版所有设计稿

- 源数据为 VDT2048 的真实、配对 RGB/Thermal/Depth；目标为 RGB-only 伪装目标检测。
- 教师在 stride-16 网格上，以 RGB query 对 T/D key 构造两路 3×3 局部关系；关系参与特征聚合与解码。RGB 学生预测同形关系并参与自身解码。
- VDT 学生阶段冻结 ImageNet 主干参数与 BN 统计，只训练关系模块和源解码头。进入 COD 时仅加载完整关系模块，COD 主干与解码头按匹配初始化开始。
- 新 A/B/C/D 分别为无 VDT、VDT RGB 监督、容量匹配的三 RGB 教师关系蒸馏、真实 RGB/T/D 教师关系蒸馏。它们与历史 A/B/C/D 定义不同。
- 新版没有已实现的关系可靠性模块、学生 uncertainty、COD relation-prior loss 或新边界 KD；首轮完整 100 轮 GPU 内部验证未显示 D 的 Sm 收益，正式测试和跨种子结论仍待验证。旧 full/encoder-only 源迁移的负结果仅构成研究动机，不能证明新机制有效或解释失败原因。

依据：[研究范围](../paper_scope.md)、[当前关系实现](../models/triprd_relation.py)、[实验指南](../tools/TRIPRD_RELATION.md)、[旧实验结论](../experiments/optimization_summary.md)。

## 3. 特权信息、多模态到单模态与模态质量（7 篇）

### 3.1 A2dele — CVPR 2020｜必读

**A2dele: Adaptive and Attentive Depth Distiller for Efficient RGB-D Salient Object Detection**，9060–9069。[正式论文页及 PDF](https://openaccess.thecvf.com/content_CVPR_2020/html/Piao_A2dele_Adaptive_and_Attentive_Depth_Distiller_for_Efficient_RGB-D_Salient_CVPR_2020_paper.html)

- **论文做法：**利用预测和注意力两条途径把深度信息传给 RGB；自适应控制预测蒸馏，并用深度定位信息指导 RGB 注意力。测试仅用 RGB。
- **我们的借鉴：**建立简单的深度预测/注意力蒸馏对照，并分别评估 T、D 两路监督质量，判断显式关系是否优于普通空间监督。
- **区别与限制：**原文是同任务 SOD；我们包含 SOD→COD 和关系模块隔离迁移。仅称“深度训练、RGB 推理”或“空间信息蒸馏”不构成创新。

### 3.2 MFH — ICLR 2023｜必读

**The Modality Focusing Hypothesis: Towards Understanding Crossmodal Knowledge Distillation**。[正式论文 PDF](https://openreview.net/pdf?id=w0QXrZ3N-s)、[作者项目](https://zihuixue.github.io/MFH/index.html)

- **论文发现：**跨模态蒸馏不总有效；高教师精度不足以预测学生收益。关键是教师是否保留两种模态共有且对任务有判别力的特征。
- **我们的借鉴：**同时记录教师分割表现、源验证集上的 RGB 学生关系拟合程度、最终 COD 收益。检验“关系更可迁移”，而非预设它成立。
- **区别与限制：**该文没有证明局部关系天然更通用，也未证明 VDT→COD。热图特有但 RGB 不可见的信息不能仅靠增加 KL 权重恢复。对旧负结果只能提出解释假设。

### 3.3 C2KD — CVPR 2024｜必读

**C2KD: Bridging the Modality Gap for Cross-Modal Knowledge Distillation**，16006–16015。[正式论文页](https://openaccess.thecvf.com/content/CVPR2024/html/Huo_C2KD_Bridging_the_Modality_Gap_for_Cross-Modal_Knowledge_Distillation_CVPR_2024_paper.html)

- **论文做法：**针对模态不均衡与软标签错位，结合在线筛选、非目标类别蒸馏、代理教师/学生及双向蒸馏，定制学生可接受的知识。
- **我们的借鉴：**先测试简化的关系监督筛选，检查教师可靠性与学生可学习性，而不是直接搬入代理网络。
- **区别与限制：**分类软标签中的非目标类分布不能原样套到二值 COD 或 9 邻域空间分布。模态差异与 SOD→COD 任务差异需分开分析。

### 3.4 QSF-Net — TIP 2024｜必读，直接使用 VDT2048

**Quality-Aware Selective Fusion Network for V-D-T Salient Object Detection**，33:3212–3226。[正式 DOI](https://doi.org/10.1109/TIP.2024.3393365)、[作者全文](https://arxiv.org/html/2405.07655v1)、[官方代码](https://github.com/Lx-Bao/QSFNet)

- **论文做法：**从各模态初步预测与 GT 构造质量区域监督，学习 T/D 质量图，再选择性融合。其高价值区域可来自“该模态正确而其他模态遗漏”的互补信息。
- **我们的借鉴：**按 T、D 分别评估局部监督是否正确、是否补足 RGB。它特别提醒我们：**模态不一致不等于模态不可靠**；把所有分歧都压低可能抹去真正的特权信息。
- **区别与限制：**它的推理仍用三模态，并非 RGB 学生跨任务蒸馏。借鉴质量评估，不把完整融合网络直接换成我们的学生；分割质量也不能直接当作关系正确性的真值。

### 3.5 TNet — TMM 2023｜热模态可靠性

**Does Thermal Really Always Matter for RGB-T Salient Object Detection?**，25:6971–6982；online 2022，正式卷年 2023。[正式 DOI](https://doi.org/10.1109/TMM.2022.3216476)、[作者稿](https://arxiv.org/abs/2210.04266)、[官方代码](https://github.com/rmcong/TNet_TMM2022)

- **论文做法：**结合光照条件调节模态作用，补足热图语义，并逐步传递定位与目标完整性线索。
- **我们的借鉴：**将 RGB→T、RGB→D 的关系误差和下游收益分别报告，再考虑不同损失权重；不能假设热信息始终有益。
- **区别与限制：**它是 RGB-T 融合检测，非训练后移除 T 的迁移。光照是质量线索之一，不足以覆盖失配、反射、深度空洞等问题。

### 3.6 Modality Hallucination — CVPR 2016｜经典定位与必要对照

**Learning With Side Information Through Modality Hallucination**，826–834。[正式论文页](https://openaccess.thecvf.com/content_cvpr_2016/html/Hoffman_Learning_With_Side_CVPR_2016_paper.html)

- **论文做法：**训练时用深度指导 RGB hallucination 分支拟合中间特征；测试利用 RGB 分支与预测的辅助特征改善检测。
- **我们的借鉴：**构造参数量匹配的“从 RGB 预测教师辅助特征”基线，与“从 RGB 预测局部关系”比较，区分关系形式与附加容量的价值。
- **区别与限制：**原任务是目标检测；我们是局部关系参与解码及小模块跨任务迁移。该对照预测的是潜在特征，不需要生成伪 T/D 图，也不应改变现有真实模态数据协议。

### 3.7 CMAD — ICCV 2025｜新颖性边界

**CMAD: Correlation-Aware and Modalities-Aware Distillation for Multimodal Sentiment Analysis with Missing Modalities**，4626–4636。[正式论文页](https://openaccess.thecvf.com/content/ICCV2025/html/Zhuang_CMAD_Correlation-Aware_and_Modalities-Aware_Distillation_for_Multimodal_Sentiment_Analysis_with_ICCV_2025_paper.html)

- **论文做法：**完整模态教师指导缺模态学生，保留相似性与高阶相关结构，并根据模态学习难度调整蒸馏权重。
- **我们的借鉴：**T/D 两路关系可研究分别加权或课程式学习；先看分路误差与收益再改权重。
- **区别与限制：**情感表征相关性不同于图像内局部空间对应。“完整模态→缺模态＋关系蒸馏＋动态权重”已有先例；训练误差小也不等于关系有价值。

## 4. COD 中的几何结构、蒸馏与边界（7 篇）

### 4.1 DepthSAM — CVPR 2026｜最重要的近期比较对象

**Beyond Appearance: Camouflaged Object Detection via Geometric Structure**，25830–25840。[正式论文页](https://openaccess.thecvf.com/content/CVPR2026/html/Han_Beyond_Appearance_Camouflaged_Object_Detection_via_Geometric_Structure_CVPR_2026_paper.html)、[作者项目](https://depthsam.github.io/)

- **论文做法：**针对单目深度与 COD 的任务错配，以 Sparse Mixture-of-Experts Adapter 适配深度先验，再融合几何和语义。
- **我们的借鉴：**关注结构知识是否适合目标任务。只迁移模块不能自动保证避免负迁移，还需验证 COD 微调前后关系的变化。
- **关键区别：**不能写成它“必须额外输入实测深度”。官方 [test.py](https://github.com/hjy0518/DepthSAM/blob/main/test.py) 的模型调用实际传入 image，[构建代码](https://github.com/hjy0518/DepthSAM/blob/main/segment_anything_training/build_DepthSAM.py)保留 DepthAnythingV2 作为图像编码器。我们的差异在真实 T/D 源域关系蒸馏和仅迁移轻量关系分支，而非单凭 RGB 输入。效率优势也需实测。

### 4.2 PopNet — ICCV 2023｜中间结构迁移

**Source-free Depth for Object Pop-out**，1032–1042。[正式论文页](https://openaccess.thecvf.com/content/ICCV2023/html/WU_Source-free_Depth_for_Object_Pop-out_ICCV_2023_paper.html)、[官方代码](https://github.com/Zongwei97/PopNet)

- **论文做法：**从单目深度模型提取几何知识，利用目标突出于接触背景面的先验，学习接触面和伪语义；不需要访问深度模型的源训练数据。
- **我们的借鉴：**不仅看关系 KL 是否下降，还要检查目标内联系、跨轮廓传播和前背景分离能力，明确“关系”携带了什么结构。
- **区别与限制：**它使用具体几何先验，我们用真实跨模态局部关系；并非所有 COD 对象都符合 pop-out 假设，不宜直接作为统一监督。

### 4.3 DSAM — ACM MM 2024｜深度指导 RGB 的 COD 蒸馏先例

**Exploring Deeper! Segment Anything Model with Depth Perception for Camouflaged Object Detection**。[ACM 正式 DOI](https://doi.org/10.1145/3664647.3681119)、[官方代码](https://github.com/guobaoxiao/DSAM)

- **论文做法：**Prompt-Deeper Module 用蒸馏和 Bias Correction 建立 RGB/深度交互；Finer Module 继续挖掘遗漏区域。
- **我们的借鉴：**对噪声模态知识采用修正和选择机制；分析蒸馏在哪些区域补足 RGB，而不是只比较总损失。
- **区别与限制：**“深度指导 RGB＋蒸馏＋COD”已有先例。我们的研究侧重 VDT 源训练、实测 T/D 关系以及跨任务模块迁移，不宜搬入完整 SAM 系统后仍归因于原机制。

### 4.4 SAM-DSA — ICCV 2025｜冻结主干与小模块

**Improving SAM for Camouflaged Object Detection via Dual Stream Adapters**，21906–21916。[正式论文页](https://openaccess.thecvf.com/content/ICCV2025/html/Liu_Improving_SAM_for_Camouflaged_Object_Detection_via_Dual_Stream_Adapters_ICCV_2025_paper.html)

- **论文做法：**冻结 SAM，训练 RGB/深度双流 adapter，结合高频小波特征、model/modal distiller 与双解码输出。
- **我们的借鉴：**为相同冻结主干策略增加直接 feature-KD 或普通 adapter 对照，判断关系内容是否有额外贡献。
- **区别与限制：**冻结主干、adapter 和模态蒸馏本身不新；其双流/prompt/预训练资源与我们不同，直接引用绝对分数不能证明我们的机制优劣。

### 4.5 CamoFormer — TPAMI 2024｜关系参与解码

**CamoFormer: Masked Separable Attention for Camouflaged Object Detection**，46(12):10362–10374。[正式 DOI](https://doi.org/10.1109/TPAMI.2024.3438565)、[官方代码与论文信息](https://github.com/HVision-NKU/CamoFormer)

- **论文做法：**通过不同 mask 策略分离注意力建模，并逐级恢复细节；提供边界区域评价。
- **我们的借鉴：**先把前景内部、背景内部和跨边界关系分开诊断，必要时研究软权重，检验是否减少背景向目标的错误传播。
- **区别与限制：**这是 RGB 注意力建模，不是特权关系迁移；预测 mask 的错误可能反馈到关系传播。项目现用的 Boundary_F 是边界区域加权 F 度量，不应另称通用轮廓匹配 F1。

### 4.6 FEDER — CVPR 2023｜结构与边缘

**Camouflaged Object Detection With Feature Decomposition and Edge Reconstruction**，22046–22055。[正式论文页](https://openaccess.thecvf.com/content/CVPR2023/html/He_Camouflaged_Object_Detection_With_Feature_Decomposition_and_Edge_Reconstruction_CVPR_2023_paper.html)

- **论文做法：**可学习小波分解、频率注意力与辅助边缘重建共同捕捉微弱线索和模糊轮廓。
- **我们的借鉴：**分析关系收益发生在目标内部结构还是轮廓；若确有边界缺陷，再单独增加轻量边界监督实验。
- **区别与限制：**高频还包含背景纹理和传感噪声，不等于有效轮廓。首轮不同时增加频域模块、边界损失与关系门控，否则无法归因。

### 4.7 GKD — CVPR 2026｜分级蒸馏与边界保留

**Beyond Weak Supervision: MLLMs-Guided Graded Knowledge Distillation for Unsupervised Camouflaged Object Detection**，27547–27557。[正式论文页](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Beyond_Weak_Supervision_MLLMs-Guided_Graded_Knowledge_Distillation_for_Unsupervised_Camouflaged_CVPR_2026_paper.html)

- **论文做法：**MLLM+SAM 生成伪标签，以 mask 质量评价和图像/像素级分级蒸馏处理不可靠监督。
- **我们的借鉴：**分别考虑图像质量与局部关系质量，避免单个全图权重掩盖局部失败。
- **区别与限制：**原文为无监督 COD；不建议给我们的全监督设置引入整套 MLLM。其[补充材料](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Chen_Beyond_Weak_Supervision_CVPR_2026_supplemental.pdf)指出 certainty gating 会牺牲边界细节；低置信边界不能简单删除。

## 5. 关系、密集预测蒸馏与迁移诊断（8 篇）

### 5.1 SKD — CVPR 2019｜关系蒸馏基础

**Structured Knowledge Distillation for Semantic Segmentation**，2604–2613。[正式论文页](https://openaccess.thecvf.com/content_CVPR_2019/html/Liu_Structured_Knowledge_Distillation_for_Semantic_Segmentation_CVPR_2019_paper.html)

- **论文做法：**组合逐像素、像素对和整体结构蒸馏，把分割知识从单点输出扩展至结构。
- **我们的借鉴：**设置传统 feature-affinity KD 对照，比较普通结构约束与显式跨模态关系预测。
- **区别与限制：**我们还让关系进入学生前向解码，因此需单独比较“仅作训练约束”和“参与解码”。不能将使用关系蒸馏本身作为贡献。

### 5.2 CIRKD — CVPR 2022｜关系覆盖与类别均衡

**Cross-Image Relational Knowledge Distillation for Semantic Segmentation**，12319–12328。[正式论文页](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_Cross-Image_Relational_Knowledge_Distillation_for_Semantic_Segmentation_CVPR_2022_paper.html)

- **论文做法：**利用类别组织的像素/区域队列，蒸馏跨图像 pixel-to-pixel 与 pixel-to-region 关系；采用规范化表征、softmax/KL 及类别均衡采样。
- **我们的借鉴：**先借鉴前景/背景均衡原则，避免全图关系 KL 主要由大量背景贡献；再考虑局部与区域关系是否互补。
- **区别与限制：**COD 中不同图像的“前景”未必语义相似，不能直接照搬类别队列。它的跨图像关系也不同于我们的空间邻域偏移分布。

### 5.3 HDMNet — CVPR 2023｜多尺度相关性

**Hierarchical Dense Correlation Distillation for Few-Shot Segmentation**，23641–23651。[正式论文页](https://openaccess.thecvf.com/content/CVPR2023/html/Peng_Hierarchical_Dense_Correlation_Distillation_for_Few-Shot_Segmentation_CVPR_2023_paper.html)

- **论文做法：**层级 query/support 匹配，使高层语义相关性指导低层细节。蒸馏的是经 support mask 聚合并空间 softmax 的相关图，不是直接对全部 QK 矩阵做 KL。
- **我们的借鉴：**若 stride-16 关系确实有效但细节不足，再研究 stride-8 分支和尺度间指导。
- **区别与限制：**跨尺度的 3×3 邻域覆盖不同实际范围，不能只插值张量后对九个偏移直接做 KL；需先定义一致空间对应并匹配模型容量。

### 5.4 BSD — ICCV 2023｜区域可靠性，优先借鉴

**A Good Student is Cooperative and Reliable: CNN-Transformer Collaborative Learning for Semantic Segmentation**，11720–11730。[正式论文页](https://openaccess.thecvf.com/content/ICCV2023/html/Zhu_A_Good_Student_is_Cooperative_and_Reliable_CNN-Transformer_Collaborative_Learning_ICCV_2023_paper.html)、[补充材料](https://openaccess.thecvf.com/content/ICCV2023/supplemental/Zhu_A_Good_Student_ICCV_2023_supplemental.pdf)

- **论文做法：**异构特征交换配合区域/像素级选择性知识传递；利用与 GT 的误差比较决定区域中更可靠的模型。
- **我们的借鉴：**利用源域已有 GT，检查教师是否在对应区域更可靠，再决定关系蒸馏强度。先改损失权重，保留原推理结构。
- **区别与限制：**原文是 CNN/Transformer 双向协作，我们是单向教师蒸馏。分割正确不能证明关系分布正确，需结合关系诊断；不能只按学生易学程度筛选而永远忽略难目标。

### 5.5 VRM — ICCV 2025｜关系错误传播

**VRM: Knowledge Distillation via Virtual Relation Matching**，2707–2717。[正式论文页](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_VRM_Knowledge_Distillation_via_Virtual_Relation_Matching_ICCV_2025_paper.html)

- **论文做法：**构造真实/虚拟视图的丰富关系，删除冗余边，并依据学生预测的联合熵及批内分位数动态移除不可靠边。
- **我们的借鉴：**检验局部不可靠关系是否经聚合扩散；先对关系损失加权，再单独考虑推理图剪枝。
- **区别与限制：**其图主要描述分类/检测预测关系，不能直接套用于跨模态九邻域。关系熵低也不等于正确，硬删边可能伤害 COD 轮廓。

### 5.6 Af-DCD — NeurIPS 2023｜局部特征对照

**Augmentation-Free Dense Contrastive Knowledge Distillation for Efficient Semantic Segmentation**。[主会正式论文页](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a12779b5e802668df1cbc73fa00da62f-Abstract-Conference.html)

- **论文做法：**结合 masked feature mimicking 与空间、通道分块的密集对比蒸馏，减少额外增强和 memory bank 的依赖。
- **我们的借鉴：**在相同冻结主干与源预算下，比较局部特征对比监督和 relation-KL，检验究竟是关系形式还是一般特征学习有益。
- **区别与限制：**原文总目标包含多项损失，不能把其全部收益归因于一个局部对比项；它也不直接生成参与解码的跨模态关系。

### 5.7 IAD — CVPR 2026｜蒸馏后的可迁移性

**From Infusion to Assimilation Distillation for Medical Image Segmentation**，20985–20995。[正式论文页](https://openaccess.thecvf.com/content/CVPR2026/html/Hong_From_Infusion_to_Assimilation_Distillation_for_Medical_Image_Segmentation_CVPR_2026_paper.html)

- **论文做法：**先通过 soft label/类别加权原型注入知识，再以样本关系等约束吸收知识；另考察蒸馏后冻结 encoder、重新学习 decoder 的迁移表现。
- **我们的借鉴：**区分源域拟合成功和目标域迁移成功，比较 COD 阶段冻结关系分支与正常微调，并监测迁移过程。
- **区别与限制：**这是医学分割/SAM 蒸馏，不是 VDT→COD。当前 COD 已会微调关系分支，建议新增的是对照和诊断，不是声称尚未做任何目标适配。

### 5.8 Omnivorous Vision Encoder — CVPR 2026｜跨模态关系的表征空间

**A Mixed Diet Makes DINO An Omnivorous Vision Encoder**，36850–36860。[正式论文页](https://openaccess.thecvf.com/content/CVPR2026/html/Kabra_A_Mixed_Diet_Makes_DINO_An_Omnivorous_Vision_Encoder_CVPR_2026_paper.html)

- **论文做法：**发现预训练 DINOv2 对同场景 RGB/深度的表征未必对齐；结合跨模态对齐与冻结教师锚定，构造统一表示空间并保留原有判别能力。
- **我们的借鉴：**验证关系投影后正确配对是否比错配更有区分度，检查关系是否塌缩为近似均匀分布。当前已有共同投影和任务监督，仍需要实证其关系含义。
- **区别与限制：**其观察不能直接断言我们的 ResNet-50 也完全失配。强行令不同模态特征相同可能损失互补信息；先诊断，再考虑归一化或对齐约束。

## 6. 最值得落地的实验：按顺序，而非同时叠加

### 6.1 第一步：完成当前最小机制和诊断

沿用新 runner 的 A/B/C/D，首先回答 D−C、D−B、D−A。C/D 保持教师容量、源数据、学生初始化、监督、学生更新预算及 COD 配方一致；教师额外成本单独报告。该步骤不添加新损失。

在既有 `relation_entropy` / `teacher_relation_entropy` 之外，建议记录：

1. 源验证集的两路关系 KL，按前景内部、背景内部、GT 边界附近分别统计，同时记录有效邻居数。
2. 教师关系熵、学生关系熵、中心偏移权重及关系分布的空间变化，排查“都接近均匀所以 KL 很小”。沿用当前按有效邻居数归一化的熵，避免图像边缘与内部不可比；单一熵指标不作为质量真值。
3. 关系聚合残差相对原特征的范数，检查学生是否实际利用分支；辅以关闭/扰动分支的敏感性诊断。临时关闭分支会引入分布变化，不能替代完整重训对照。
4. 源阶段、COD 初始化、COD 固定训练轮次的关系图变化；在固定的 VDT 留出样本上检查源知识保留，在 COD 内部验证样本上检查目标结构。不得构造不存在的 COD 真实 T/D 教师信号。

### 6.2 第二步：关系损失的区域均衡

**借鉴 CIRKD，结合 CamoFormer/FEDER 的结构分析。** 比较全图均值 KL 与前景内部/背景内部/边界带三个不重叠区域的分组均衡 KL；每组先归一化，再用预先固定权重组合。处理某张图没有该区域的情况，避免改变损失量级后误把权重变化当方法收益。

这是相对低成本的实验：主要修改 `RelationLoss` 的归约方式，保留学生推理结构。它测试“背景数量是否掩盖关键区域”，不预设教师关系应总在同类像素间更强——跨边界关系也可能有用。

### 6.3 第三步：可靠性加权，但保留互补知识

**借鉴 QSF-Net、BSD、C2KD、VRM。** 先用 VDT 训练标签检查教师区域误差与关系质量是否相关，再比较等权、可靠性加权以及只用相同 GT 的简单权重控制。

优先区分三类情形：教师与 RGB 都正确；教师正确而 RGB 错误；教师错误。第二类可能正是值得传递的特权知识，不能因分歧较大就一律压低。为 T/D 各自报告权重分布、边界保留率、前景覆盖率；使用 detached 权重并检查是否退化为全零或近常数。

目前教师只输出融合后的分割结果，不能直接假装已有 T/D 各自的可靠性预测。若新增模态预测头或质量头，应在 RGB 控制教师 C 中同步匹配结构与训练成本；若把质量头保留到学生推理，另报告参数和时间。

### 6.4 第四步：证明“学到的是特权关系”

| 对照 | 主要排除的解释 | 执行注意 |
|---|---|---|
| 容量匹配的三 RGB 教师 C | 教师更大、一般 KD 即可解释 | 已完成首轮单种子验证，见 E013；继续诊断与配对复验 |
| GT-only 局部关系 | 只是重新编码源 mask | 匹配监督/预算与关系分支容量 |
| 普通 feature-affinity KD / Af-DCD 式特征对照 | 任何特征约束都同样有效 | 保持相同源数据、训练范围与目标配方 |
| 参数匹配的辅助特征预测 | 额外参数或模态 hallucination 即可解释 | 不引入伪 T/D 数据；报告潜在特征分支成本 |
| 空间打乱关系目标 | 只需关系分布正则，不需要真实对应 | 保留边界有效邻域，避免非法位置造成混淆 |
| 固定错误 T/D 配对后重训教师/学生 | 真实配对并不重要 | 与训练后随机打乱目标分开；固定错配规则 |
| 关系仅作训练约束 vs 进入解码 | 效果只来自辅助 loss | 容量和训练路径尽量匹配；完整重训 |

这些对照已有部分写入项目计划，文献调研是在补充依据；不能将尚未实施的消融写成实验结果。

### 6.5 第五步：有证据后再扩展尺度或目标适配

只有最小机制产生可复验净收益后，再考虑 HDMNet 式多尺度关系、COD 阶段冻结/微调关系分支对照，或关系保持正则。增加源数据 replay 会改变训练资源与更新预算，应另设匹配控制；不能把额外预算收益归于关系保持本身。

所有配置先在固定内部验证集筛选，再做配对种子复验；保留 COD10K/CAMO 两来源结果，报告 Sm/Fw/mean E/MAE 以及边界指标。最终全量训练和测试沿用已冻结规则，不按测试集挑阈值、epoch 或关系权重。本文不承诺任何提升幅度。

## 7. 对应代码入口与论文写作边界

| 候选研究项 | 当前代码入口 | 工作性质 |
|---|---|---|
| 分组 KL、可靠性加权 | `models/triprd_relation.py` 的 `RelationLoss` | 待实现；优先保持推理图不变 |
| Q/K 归一化、关系诊断、多尺度 | 同文件的 `LocalRelationBranch` | 先诊断，尺度扩展需要容量匹配 |
| 冻结或微调关系分支 | `tools/run_triprd_relation.py` 的源/COD 训练流程 | 当前已经 COD 全模型微调；补冻结对照 |
| 只迁移关系参数 | `tools/triprd_relation_training.py` | 已实现，继续保留主干/BN/解码头边界检查 |
| 配对对照、种子复验 | `tools/run_triprd_relation_queue.py` | 已有四组队列；新消融需独立登记 |

论文相关工作建议分成三条线：训练期特权信息与 RGB 部署（Hallucination、A2dele、MFH、C2KD）；结构与关系蒸馏（SKD、CIRKD、HDMNet、VRM、CMAD）；几何/深度辅助 COD 与参数高效适配（PopNet、DSAM、SAM-DSA、DepthSAM）。QSF-Net/TNet 说明实测辅助模态的可靠性问题，但不是跨任务蒸馏有效的证据。

可以把研究问题写成：**源任务的多模态空间结构，能否通过与主干隔离的关系预测模块，改善另一个任务中的 RGB 目标/背景推理？** 避免在完成对照前写“首次”“消除负迁移”“显著提升”或“达到 SOTA”。

## 8. 未计入 22 篇的边界说明

- VDT2048 数据原始论文 *A Novel Visible-Depth-Thermal Image Dataset of Salient Object Detection for Robotic Visual Perception* 发表于 IEEE/ASME Transactions on Mechatronics（正式卷年 2023，online 2022），不是本清单采用的 CCF-A 来源，故未计入 22 篇。[数据作者官方仓库](https://github.com/VDT-2048/VDT-Dataset)。正式论文仍应引用数据来源，不能因本次只检索 A 类而省略数据归属。
- ECCV、IJCAI 本次按 2026 新版 B 类排除；未核实正式录用的预印本不纳入主表。已发表论文的作者 arXiv 副本可作为全文阅读入口，是否纳入由正式发表记录决定。
- 比较各论文性能前需单独核对 backbone、预训练资源、输入模态、prompt/监督方式、训练集和指标定义。本次不拼接不同协议的分数制作虚假的统一排名。

本文只新增文献整理，不修改模型、不启动训练；当前项目中的有效性结论仍以实际实验记录为准。
