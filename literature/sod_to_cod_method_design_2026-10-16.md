# 三模态 SOD 到 RGB COD 的方法构建文献与研究方案

整理日期：2026-10-16。补充 16 篇 CCF-A 论文，重点是知识表示、跨任务迁移机制和模型构建，衔接[上一轮 22 篇文献](ccf_a_related_work_2026-10-16.md)。

最值得推进的问题是：**三模态 SOD 能教会 RGB 模型哪些局部结构，以及 COD 应当怎样使用这些结构，而不继承源任务的显著性选择偏好？** 对应的方法需要同时明确结构的含义、RGB 学习结构的方式、结构参与 COD 推理的路径。仅换一个蒸馏损失或冻结更多参数，不能完整回答这个问题。

下文先列论文依据，再给出三个研究方案。方案是针对 TriPRD 的推导，尚未在本项目实现或验证；不把论文在原任务上的结论直接当作 VDT→COD 的证据。

**实验状态更新（2026-10-16 归档）：**当前关系管线的首轮单种子、100 轮 A/B/C/D 内部验证已完成，D 未获得 Sm 净收益，见 [E011–E014 汇总](../experiments/relation_summary.md)。本文三个方案仍是独立提案；先开展现有教师的关系干预诊断，再据证据选择后续实现，不同时叠加全部设计。

## 分级和数据范围

沿用 CCF 2026 年第七版目录：本次涉及 CVPR、ICCV、ICML、NeurIPS、ICLR、TIP，均按该版 A 类筛选。会议限正式主会论文；ICLR 在该版为 A。依据为 [CCF 官方发布页](https://www.ccf.org.cn/Academic_Evaluation/By_category/)和[目录正文镜像](https://scdm-shu.github.io/ccf/2026-CCF-Ranking.pdf)。

方法建议保持当前数据范围：源域为 VDT2048 官方训练集的真实配对 RGB/T/D 与 SOD 标注；目标训练为 COD10K train 和 CAMO train 的 RGB 与 COD 标注；推理仅使用 RGB。这里不引入合成 COD 图像、生成的辅助模态、其他训练数据或测试集标注。涉及联合训练、源数据回放或额外参考编码器的方案，需要单独记账并匹配对照。

## 优先读哪些论文

| 顺序 | 论文 | 阅读时要回答的问题 | 对方法的影响 |
|---|---|---|---|
| 1 | VSCode，CVPR 2024 | 模态改变与 SOD/COD 任务改变，是否应由不同机制处理？ | 分开设计模态知识获取和目标选择 |
| 2 | SAT，TIP 2023 | 对象语义与显著性属性是否必须一起迁移？ | 给源任务偏好单独的表示路径 |
| 3 | Spider，ICML 2024 | 同一视觉表示如何根据上下文概念选择不同对象？ | 共享结构加任务条件读出 |
| 4 | PAP，CVPR 2019 | 不同关系如何共同参与预测，而不被强行对齐成一张图？ | 保留多种关系，由接收任务选择和传播 |
| 5 | FactorCL，NeurIPS 2023 | 跨模态共有信息之外的知识应怎样保留？ | 关系蒸馏分支与 RGB 判别分支分工 |
| 6 | ReFilled，CVPR 2020 | 源目标语义不一致时，还能传什么知识？ | 迁移局部比较和排序，而非固定 logits |
| 7 | AffinityNet，CVPR 2018 | 怎样让“关系”有明确监督含义，并实际参与推理？ | 区分兼容性预测与归一化传播权重 |
| 8 | DMIL，CVPR 2026 | 冗余、独有、协同信息是否适合用同一种方式交给 RGB？ | 更深入的教师知识分解路线 |

## 直接连接 SOD 与 COD 的论文

### 1 VSCode

**VSCode: General Visual Salient and Camouflaged Object Detection with 2D Prompt Learning**，CVPR 2024，17169–17180。[正式论文](https://openaccess.thecvf.com/content/CVPR2024/html/Luo_VSCode_General_Visual_Salient_and_Camouflaged_Object_Detection_with_2D_CVPR_2024_paper.html) · [作者全文](https://arxiv.org/html/2311.15011v2) · [代码](https://github.com/Sssssuperior/VSCode)

**原文机制。** 在共享编码解码模型中区分模态提示和任务提示，分别处理 RGB/depth/thermal/flow 差异与 SOD/COD 差异。任务提示同时进入编码和解码阶段。训练覆盖四种 SOD、两种 COD 任务，再组合提示测试未训练的 RGB-D COD 组合。

**对我们的启发。** 不必照搬 prompt 微调；更重要的是把方法划分为“如何获取跨模态结构”和“当前任务如何使用结构”两个机制。RGB 学生可以保留结构预测器，由 COD 条件模块选择结构的作用位置和传播方式。其零样本实验是未见过的模态与任务组合，并不是从未使用 COD 标签，也不证明测试时可以删除辅助模态。

### 2 SAT

**Nowhere to Disguise: Spot Camouflaged Objects via Saliency Attribute Transfer**，TIP 2023，32:3108–3120。[正式 DOI](https://doi.org/10.1109/TIP.2023.3277793) · [IEEE 论文页](https://ieeexplore.ieee.org/abstract/document/10132418) · [代码](https://github.com/wdzhao123/SAT)

**原文机制。** 分离对象语义表示与上下文属性，并通过显著性属性迁移生成较弱伪装的图像，连接 SOD 和 COD，使 SOD 模型能够用于伪装目标定位。

**对我们的启发。** “对象是什么/结构怎样”与“对象相对背景有多显著”可以作为不同表示目标。源教师可有结构分支和 SOD 选择分支，学生不必接收后者。原文包含图像生成；我们借鉴的是属性分解问题与迁移接口，不采用其合成图像流程。只把两个分支命名为 structure/saliency 还不能证明分离成功。

### 3 Spider

**Spider: A Unified Framework for Context-dependent Concept Segmentation**，ICML 2024，PMLR 235:60906–60926。[正式论文及 PDF](https://proceedings.mlr.press/v235/zhao24j.html) · [作者全文](https://arxiv.org/html/2405.01002v2) · [代码](https://github.com/Xiaoqi-Zhao-DLUT/Spider-UniCDSeg)

**原文机制。** 用图像与掩码组提示提取前景、背景概念，通过概念过滤器调节共享分割过程，处理显著、伪装、透明、阴影等上下文相关概念。其提示来自参考样本，不是测试图像的真实掩码。

**对我们的启发。** 相同的候选结构可以交给不同目标选择器。先学习局部区域和边界线索，再用 COD 的上下文条件读取这些线索，比直接继承 SOD foreground 响应更有针对性。我们可以设计训练集原型或可学习的任务条件，但应明确这是简化适配，不是复现 Spider；推理不能依赖测试 GT。

### 4 Uncertainty-Aware Joint SOD and COD

**Uncertainty-Aware Joint Salient Object and Camouflaged Object Detection**，CVPR 2021，10071–10081。[正式论文及 PDF](https://openaccess.thecvf.com/content/CVPR2021/html/Li_Uncertainty-Aware_Joint_Salient_Object_and_Camouflaged_Object_Detection_CVPR_2021_paper.html)

**原文机制。** 采用任务专用编码器和共享解码器，通过连接图像约束任务表示的差异，并结合数据交互和不确定性建模进行联合学习。

**对我们的启发。** 该文的价值不只是 uncertainty loss，而是直接讨论 SOD 与 COD 的共享和差异应放在哪里。可用其对照“全部共享”“结构共享、任务选择独立”两种设计。不能把所有 SOD/COD 特征都强迫为相反方向；原文的连接图像资源也不能悄悄加入当前固定数据范围。

### 5 USCNet

**Rethinking Detecting Salient and Camouflaged Objects in Unconstrained Scenes**，ICCV 2025，22372–22382。[正式论文及 PDF](https://openaccess.thecvf.com/content/ICCV2025/html/Zhou_Rethinking_Detecting_Salient_and_Camouflaged_Objects_in_Unconstrained_Scenes_ICCV_2025_paper.html) · [代码](https://github.com/ssecv/USCNet)

**原文机制。** 面向显著目标和伪装目标可能同时存在或都不存在的场景，以显著、伪装、背景查询及跨样本、图像内条件提示进行目标区分。

**对我们的启发。** COD 不是把显著性反过来。目标选择需要结合图像上下文，不能仅将低显著区域都视为伪装目标。当前 SOD/COD 数据没有统一的三类完整标注，因此不能直接训练“显著/伪装/背景”互斥三分类；可保留独立二值任务头，只计算有对应标注的任务损失。

### 6 MGL

**Mutual Graph Learning for Camouflaged Object Detection**，CVPR 2021，12997–13007。[正式论文及 PDF](https://openaccess.thecvf.com/content/CVPR2021/html/Zhai_Mutual_Graph_Learning_for_Camouflaged_Object_Detection_CVPR_2021_paper.html)

**原文机制。** 区域图和边缘图相互推理，区域语义帮助边缘表征，边缘结构反过来约束区域预测。

**对我们的启发。** 可以把“关系知识”定义成有不同功能的区域关系和轮廓关系，让关系决定前向传播，而非仅加边界损失。若用深度或热图辅助这些关系，应保留测量关系与语义边界的区别：深度跳变和热响应跳变都不自动等于对象轮廓。

## 决定哪些多模态知识值得迁移的论文

### 7 FactorCL

**Factorized Contrastive Learning: Going Beyond Multi-view Redundancy**，NeurIPS 2023。[正式论文及 PDF](https://proceedings.neurips.cc/paper_files/paper/2023/hash/6818dcc65fdf3cbd4b05770fb957803e-Abstract-Conference.html)

**原文机制。** 分别学习任务相关的跨模态共享和模态独有表示，通过互信息界及增强构造，避免普通跨模态对齐只保留共有信息。

**对我们的启发。** 教师可分别保留 RGB-D、RGB-T 的候选共享结构，以及模态专属表示；学生保留一个不受关系蒸馏强制约束的 RGB 判别分支。不要为了三模态一致而删除对 COD 有用的 RGB 细纹理，也不要强迫三种模态只有一个公共交集。源 SOD 相关的共享信息仍不等于 COD 所需信息。

### 8 CoMM

**What to align in multimodal contrastive learning?**，ICLR 2025。[正式论文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/108030643e640ac050e0ed5e6aace48f-Abstract-Conference.html) · [PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/108030643e640ac050e0ed5e6aace48f-Paper-Conference.pdf) · [代码](https://github.com/Duplums/CoMM)

**原文机制。** 对齐增强后的融合多模态视图，避免直接跨模态对齐丢失独有和协同信息。

**对我们的启发。** 用于构建更完整的三模态教师，而不是要求教师所有分支都接近 RGB。随后再决定哪些知识交给学生。强教师可以利用 RGB 无法独立恢复的知识，因此教师训练目标与学生蒸馏目标不必相同；空间增强需保持 RGB/T/D 对应关系。

### 9 DMIL

**Information-Theoretic Decomposition for Multimodal Interaction Learning**，CVPR 2026。[正式论文 PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Yang_Information-Theoretic_Decomposition_for_Multimodal_Interaction_Learning_CVPR_2026_paper.pdf) · [作者全文](https://arxiv.org/html/2606.11614v1) · [代码](https://github.com/GeWu-Lab/DMIL)

**原文机制。** 用变分分解区分单模态可解释成分及残余，再建模冗余、独有和协同信息，并以分阶段训练和预测组合利用这些成分。

**对我们的启发。** 把三模态教师的信息分解放在模型结构中：哪些结构可由 RGB 与 D/T 共同解释，哪些只有教师能用。学生优先接收可预测的结构，教师仍可使用其他成分完成源任务。这里的分解是建模和近似估计，不能将其表述为精确、可识别的三模态信息分解；VDT 数据规模下应先考虑两组模态对。

### 10 Multimodal Information Bottleneck

**Aligning Multimodal Representations through an Information Bottleneck**，ICML 2025，PMLR 267:1250–1270。[正式论文及 PDF](https://proceedings.mlr.press/v267/almudevar25a.html) · [作者全文](https://arxiv.org/html/2506.04870v1)

**原文机制。** 从信息瓶颈角度解释跨模态对齐，除保留共有信息外，还约束表示保留的模态特有信息；在特定分布假设下得到可实现的配对对齐目标。

**对我们的启发。** 仅对低维结构投影施加共同信息约束，同时保留未被压缩的 RGB 路径。不能对整个 COD 表征强行去除 RGB 独有信息；低维、正交或特征接近都不足以证明语义独立。

### 11 LSMI

**Efficient Quantification of Multimodal Interaction at Sample Level**，ICML 2025，PMLR 267:71302–71317。[正式论文及 PDF](https://proceedings.mlr.press/v267/yang25aj.html) · [作者全文](https://arxiv.org/html/2506.17248v1) · [代码](https://github.com/GeWu-Lab/LSMI_Estimator)

**原文机制。** 在样本层面估计冗余、独有和协同交互，并研究按交互类型选择知识蒸馏方式，例如特征层或输出层蒸馏。

**对我们的启发。** 教师知识不是一个统一的 KL 目标：可共享结构、模态独有证据和融合协同信息应分别处理。该文可指导诊断和知识路由，但其蒸馏设置不同于我们的辅助模态缺失场景，不能推出“协同输出也一定能被 RGB 学会”。从样本估计推广到像素还涉及样本量与估计稳定性。

## 改变跨任务迁移对象的论文

### 12 ReFilled

**Distilling Cross-Task Knowledge via Relationship Matching**，CVPR 2020，12396–12405。[正式论文及 PDF](https://openaccess.thecvf.com/content_CVPR_2020/html/Ye_Distilling_Cross-Task_Knowledge_via_Relationship_Matching_CVPR_2020_paper.html)

**原文机制。** 面向师生类别空间不一致的任务，以教师度量关系指导学生的相对比较，并通过教师表示构造目标任务局部分类器，避免机械匹配源类别输出。

**对我们的启发。** 从“复制 9 个邻居的完整概率分布”改成“保留哪些邻居更相似、哪些跨越结构变化”的局部比较。可设计三模态教师→RGB 比较器→COD 的迁移链，使 COD 阶段有一个实际可运行的 RGB 知识提供者。原文是类别任务，像素或区域三元组是我们的扩展；RGB 比较器也可能在 COD 上犯错，必须由目标标注检验。

### 13 AffinityNet

**Learning Pixel-Level Semantic Affinity With Image-Level Supervision for Weakly Supervised Semantic Segmentation**，CVPR 2018，4981–4990。[正式论文及 PDF](https://openaccess.thecvf.com/content_cvpr_2018/html/Ahn_Learning_Pixel-Level_Semantic_CVPR_2018_paper.html)

**原文机制。** 学习邻近像素的语义兼容关系，并通过传播改进分割；其弱监督流程中还处理前景同类、背景同类和异类关系。

**对我们的启发。** 将“关系预测”定义为明确的中间任务，而不是把任意注意力称为结构。我们已有分割 GT，可在源和目标各自构造同标注类别关系。它首先适合作为重要基线：如果 GT-only 关系已足够，三模态教师就必须说明提供了什么额外信息。当前 9 邻域 softmax 表示相互竞争的传播比例，不能直接当作多个独立的同类概率来做二值监督。

### 14 PAP

**Pattern-Affinitive Propagation Across Depth, Surface Normal and Semantic Segmentation**，CVPR 2019，4106–4115。[正式论文 PDF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_Pattern-Affinitive_Propagation_Across_Depth_Surface_Normal_and_Semantic_Segmentation_CVPR_2019_paper.pdf)

**原文机制。** 分别构建不同任务的亲和关系，按接收任务组合这些关系并传播特征，利用不同预测任务之间的结构联系。

**对我们的启发。** 对 TriPRD 很直接：深度连续性、热响应相似性、标注区域兼容性不必成为同一张图；COD 可以学习怎样使用它们。原文连接的是深度、法线、分割任务，并非 RGB/T/D 传感器蒸馏，因此这是一种架构迁移思路。先做局部图即可检验机制，无须立即构造全分辨率二次复杂度关系矩阵。

### 15 Cross-Task Consistency

**Robust Learning Through Cross-Task Consistency**，CVPR 2020，11197–11206。[正式论文及 PDF](https://openaccess.thecvf.com/content_CVPR_2020/html/Zamir_Robust_Learning_Through_Cross-Task_Consistency_CVPR_2020_paper.html) · [项目](https://consistency.epfl.ch/)

**原文机制。** 将多个预测任务构成图，要求不同预测路径之间保持一致，以约束单一任务容易产生的错误。

**对我们的启发。** 可比较 RGB→标注兼容图与 RGB→掩码→兼容图两条路径，使关系不只是源训练阶段的附加损失。例如对 i≠j，由预测概率构造 a_mask(i,j)=p_i p_j+(1−p_i)(1−p_j)，再与独立兼容性预测比较；自环直接设为 1，不计算一致性损失。该表达式是我们的二值扩展，并非原文公式。两条路径都要有 GT 锚定，以免共同退化；此一致性只适用于语义兼容图，不能强加给深度或热关系。

### 16 DSN

**Domain Separation Networks**，NeurIPS 2016。[正式论文及 PDF](https://proceedings.neurips.cc/paper_files/paper/2016/hash/45fbc6d3e05ebd93369ce542e8f2322d-Abstract.html)

**原文机制。** 通过共享与域专属表示、差异约束和重建等目标，学习适用于域适应的表示分工。

**对我们的启发。** 为源任务保留专属路径，只把候选结构部分送入学生。该文主要处理同标签空间的域差异，我们同时有任务语义和数据分布变化，不能直接把两套 foreground 对齐后称为通用对象表示。SOD 来自 VDT、COD 来自另一数据集，也使“任务特有”和“数据集特有”因素难以单独识别。

## 三种方法构建路线

### 路线一 多类型结构预测与 COD 条件传播

**优先推荐。** 主要依据 PAP、VSCode、Spider 和 AffinityNet，SAT 提供分离显著性选择的动机。关键改变是为被迁移的关系赋予不同的监督含义，再让目标任务决定如何使用这些关系。

#### 将关系类型分开定义

| 关系 | 源阶段监督从哪里来 | 表示什么 | 不能解释成什么 |
|---|---|---|---|
| 深度关系 | 有效的真实深度观测 | 局部相对差、排序或连续性，先选一种定义 | 未经标定的真实距离、同一物体 |
| 热响应关系 | 真实热图观测 | 局部响应相似性或变化 | 未经辐射标定的温度、目标归属 |
| 标注兼容关系 | 当前任务的二值掩码 | 两点是否属于同一标注类别 | 同一实例、通用 objectness |

“关系类型”与“共享/私有信息”是两个维度。深度关系不是天然共享，热关系也不是天然私有。RGB 是否能预测它们需要在源训练集内部留出的数据上测量；辅助模态能观察到、RGB 看不到的变化，不会因蒸馏而自动变得可恢复。

源掩码不是场景中所有对象的完整标注，背景中的两个点可能属于不同对象。因此，GT 同类关系应命名为**标注类别兼容性**。只学习这种关系可能只是重编码 SOD mask，不能作为已经去掉显著性偏好的证据。

#### 让预测关系实际控制计算

源阶段用三模态信息学习或提炼这些关系，再监督 RGB 关系预测器。学生输出低分辨率边属性，关系沿邻域控制特征传播。另保留未受关系蒸馏强制约束的 RGB 判别路径。

对适合用 [0,1] 表示的兼容性边，可令每条边独立输出 g(i,j)=sigmoid(h(i,j))，仅在传播时归一化。一个可检验的设计是设固定自环 g(i,i)=1，再计算：

    α(i,j) = g(i,j) / [1 + Σ(k≠i) g(i,k)]，j≠i
    α(i,i) = 1 / [1 + Σ(k≠i) g(i,k)]

这样所有邻边都弱时，传播主要保持原位置，避免“所有边都不可信，但 softmax 仍迫使模型挑一个邻居”。不同类型的关系可分别生成传播权重；有向深度差需要先通过独立映射转为传播权重，不能直接套概率解释。

COD 的选择模块根据 RGB 上下文、预测关系和任务条件，学习在什么位置使用哪种关系，并允许关闭关系传播。它不能读取 COD 的真实 T/D 或依赖测试 GT。SOD 和 COD 可以使用相同的结构候选，但不必采用相同的融合权重或前景选择方式。

#### 说明教师比直接传感器监督多做什么

如果深度边和热响应边都能由原始观测直接计算，学生可以直接学习这些目标，教师不是必需的。保留教师应有具体职责，例如利用三模态联合证据提炼多尺度关系、处理测量缺陷，或输出适合传播的关系表示。

因此该路线必须比较**直接真实观测边监督**和**教师提炼边监督**。如果后者没有额外价值，方法应定位为特权结构监督与跨任务关系迁移，不必为了维持 teacher–student 名称增加教师。学生不重建 T/D 图像，也不把生成图作为输入；但它仍属于特权信息预测，在广义上与关系层面的模态补全思想有关。

#### COD 阶段保留什么并改变什么

COD GT 可以继续训练语义兼容性和任务读出，但不能把深度/热关系都监督成 foreground/background。分割梯度也可能改变这些关系的含义，即使没有给它们直接施加语义边损失。

若要检验“固定结构知识能否供 COD 使用”，必须固定完整的结构预测函数，包括其输入特征路径；仅冻结最后一个关系头，而让前面的主干继续变化，不能保证关系不变。可以先冻结公共结构特征路径，或使用单独的固定关系编码路径；后一种方案的计算和参数成本必须纳入匹配基线。

另一种选择是允许关系适配，但应称为“由深度/热监督初始化的关系”，不能继续默认其保留原测量含义。若以源数据回放保持含义，则作为另一个训练协议单独比较。固定与适配两种版本分别回答“结构本身能迁移”和“结构初始化是否有帮助”，不应混为一个结论。

**比普通微调多出的机制：** 独立定义的结构监督对象、结构控制前向传播、COD 根据上下文选择关系。其研究主张是“真实 T/D 改善了对 COD 有用的 RGB 结构预测”，而不是“注意力天然跨任务通用”。

### 路线二 分解教师知识后选择性迁移

主要依据 FactorCL、DMIL、信息瓶颈和 SAT。教师分别设置候选共有结构与模态专属表示；RGB-D 和 RGB-T 可以有不同的共有子空间，避免把全部信息压入三模态唯一交集。教师用完整信息完成 SOD，学生只接收可由 RGB 学习的结构表示，同时保留自身 RGB 判别路径。

这条路线改变的是教师和学生的表示分工，而不是给所有教师特征统一乘一个置信度。CoMM 可用于教师学习完整信息；LSMI 则提供按交互类型分析知识的思路。它们不是都必须加入的模块。

源共有表示还需接受明确的结构监督或跨任务检验，否则它可能只是“三个模态共同支持的显著性”。正交损失、低相关性和两个分支的名称都不能证明因果解耦或严格任务不变性。VDT 规模下，先做两组模态对和少量低维投影，比直接训练复杂的多专家信息分解系统更容易判断收益来源。

**适用情形：** 源教师很强，但 RGB 学生难以拟合；或模态一致化明显损伤 RGB 细节。它可作为路线一的后续研究，也可以独立成方法，但不宜一开始与全部图模块同时引入。

### 路线三 通过 RGB 比较器迁移局部排序知识

主要依据 ReFilled，可结合标注兼容关系和 Cross-Task Consistency。

1. 在 VDT 上，三模态教师提供局部比较，例如相对锚点 i，邻居 j 比邻居 k 在指定结构上更接近。
2. RGB 比较器学习这些排序。它必须作为可运行的中间模型存在，三模态教师不能在缺少 T/D 的 COD 图像上直接提供监督。
3. 在 COD 训练图像上运行 RGB 比较器；根据 COD GT 检验与当前语义兼容的比较，指导 COD 图推理或特征度量。涉及深度等其他关系的排序则保持独立语义，不能仅因跨越 COD 前景边界就判为错误。

这条路线从“复制一整行注意力分布”变为“保留相对结构判断”。它不要求师生输出相同的源类别，也不要求源目标各邻边权重逐项相同。但 RGB 比较器可能沿用源偏差，增加中间模型和训练阶段也需要匹配计算预算。若 COD 的比较目标完全由其 GT 重新生成，则应检查源比较器是否仍提供额外知识。

**适用情形：** 绝对关系分布随任务变化较大，但局部相对次序更稳定。先验证这个现象，再采用排序迁移，不能仅因排序损失不同就宣称更可迁移。

## 怎样证明它不是普通预训练加微调

应优先围绕路线一构建可区分的证据，而不是一次叠加三条路线。

| 对照 | 回答的问题 |
|---|---|
| 同架构、同源数据和训练预算的普通 SOD 预训练与 COD 微调 | 结构目标是否优于一般源预训练？ |
| GT-only 标注兼容图 | 三模态是否提供了掩码关系以外的价值？ |
| 容量匹配三 RGB 教师与真实 RGB/T/D 教师，并区分是否使用观测边标签 | 辅助结构监督和教师多模态输入分别有什么价值？ |
| 直接观测边监督与教师提炼边监督 | 教师是否必要，是否只是包装可直接计算的标签？ |
| 只预测边但不用于传播 | 收益来自辅助监督还是结构推理路径？ |
| 固定关系组合与 COD 条件选择 | 任务条件使用是否必要？ |
| 固定完整结构预测函数与允许目标适配 | 迁移的是可用知识，还是仅有利的初始化？ |
| 合法配对与错误 T/D 配对，正确空间关系与打乱关系 | 是否真正依赖传感器对应和空间结构？ |

这里的教师对照需要三个组：纯 RGB 教师只接受 SOD 与标注兼容关系监督；第二组仍只输入 RGB，但获得真实 T/D 导出的结构标签；第三组输入真实 RGB/T/D 并获得同样的结构标签。第二组已经使用特权监督，第三组相对第二组的提升只隔离多模态教师输入的作用；第三组相对第一组则是输入与结构监督的整体增益。不能把两种比较混为“只改变教师输入”。这些是新路线的设计对照，不覆盖当前运行脚本的 A/B/C/D 定义。

三项条件应分别测量：**源观测与结构目标是否可信、RGB 能否预测、COD 是否受益。** 教师分割分数高不自动满足后两项。目标没有真实 T/D，不能声称验证了目标上的物理关系精度；可以评估 COD 分割、语义兼容性和关系干预对预测的影响，并据此限定结论。

源观测质量与 RGB 结构误差使用 VDT 官方训练集内部划分；目标选择在固定的 COD 训练内部验证划分完成。正式测试集不用于选择图类型、损失、训练时长或读取规则。新增模型与回放的费用单独报告，并区分收益相对同架构控制的提升与绝对排行榜结果。

## 与当前 TriPRD 的关系和推荐次序

当前实现是 RGB→T/D 两路局部 attention、源任务分割监督、关系 KL、关系参与解码及关系分支迁移。它已经有显式前向路径，但两路关系还主要由 SOD 目标塑造；不因只迁移小模块就自动消除了源任务选择偏好。依据：[研究范围](../paper_scope.md)、[关系实现](../models/triprd_relation.py)、[运行指南](../tools/TRIPRD_RELATION.md)。

建议先精读 **VSCode → SAT → Spider → PAP → AffinityNet**，据此画清“结构获取—RGB 预测—COD 条件使用”的计算图。首个新方案以路线一为核心：一个明确的观测结构目标、一个独立的标注兼容关系，以及允许关闭传播的 COD 选择模块，就足以检验主要假设；不必同时加入复杂信息分解、多个尺度和排序教师。

如果主要瓶颈是 RGB 学不到教师关系，再用 **FactorCL、信息瓶颈、DMIL** 重构迁移对象；如果主要瓶颈是目标任务改变了绝对关系权重，而相对比较仍稳定，再采用 **ReFilled** 路线。

可作为研究主张的表述是：**利用训练期真实辅助观测学习可由 RGB 预测的局部结构，让 COD 在目标上下文中选择其用途。** “通用 objectness”“严格去显著性”“因果解耦”和“已证明任务不变”都不是当前设计本身能够推出的结论。

检索中另有 **Unsupervised Camouflaged Object Segmentation as Domain Adaptation**，但正式发表于 [ICCV 2023 OODCV Workshop](https://openaccess.thecvf.com/content/ICCV2023W/OODCV/html/Zhang_Unsupervised_Camouflaged_Object_Segmentation_as_Domain_Adaptation_ICCVW_2023_paper.html)，不计入上述 CCF-A 主会清单。
