可以。现在按 **CVPR 2027 主会**倒排，目标不是继续扩展想法，而是在投稿前证明一个清晰、可复现的核心结论：

> 真实 VDT2048 的 RGB-T-D 关系，可以通过关系、边界和不确定性蒸馏，稳定提升只使用 RGB 的真实 COD 模型。

CVPR 2027 官方节点是：注册截止 **2026-11-10 AoE**，论文截止 **2026-11-16 AoE**，补充材料截止 **2026-11-23 AoE**。[官方时间表](https://cvpr.thecvf.com/Conferences/2027/Dates)

---

# 一、立刻冻结的研究版本

## 方法名称

**Privileged Tri-Modal Relation Distillation for RGB-Only Camouflaged Object Detection**

## 固定数据

### 多模态知识源

只使用：

- VDT2048 train split；
- RGB、T、D 和 mask。

不使用：

- PST900；
- VDT2048 test split；
- 合成 COD；
- 伪 T/D；
- 真实 COD 测试集中的额外信息。

### COD 数据

训练：

- COD10K train；
- CAMO train。

测试：

- COD10K test；
- CAMO test；
- NC4K；
- CHAMELEON。

## 固定模型

教师：

```text
RGB encoder + T encoder + D encoder
        ↓
cross-modal relation module
        ↓
mask / boundary / uncertainty heads
```

学生：

```text
RGB encoder
        ↓
relation / boundary / uncertainty heads
        ↓
COD decoder
```

最终测试只输入 RGB。

## 固定核心损失

\[
\mathcal L
=
\mathcal L_{COD}
+
\lambda_l\mathcal L_{logit}
+
\lambda_r\mathcal L_{relation}
+
\lambda_b\mathcal L_{boundary}
+
\lambda_u\mathcal L_{uncertainty}
\]

暂时不加入：

- 生成模型；
- 扩散模型；
- 额外数据集；
- 复杂理论定理；
- 多教师；
- 多 backbone；
- 新的伪装图像生成模块。

---

# 二、五周倒排计划

## 第 1 周：数据和基线必须跑通

**时间：10 月 12 日—10 月 18 日**

### 工作

1. 整理 VDT2048：
   - RGB/T/D/mask 对齐；
   - 官方训练划分；
   - 缺失值和尺寸检查；
   - 生成可视化网格。

2. 整理 COD：
   - COD10K；
   - CAMO；
   - NC4K；
   - CHAMELEON。

3. 完成统一评价脚本：
   - \(S_\alpha\)；
   - \(F_\beta^w\)；
   - \(E_\phi\)；
   - MAE；
   - Boundary F。

4. 跑通一个 RGB-only COD baseline。

### 本周必须产出

- `vd2048_dataset.py`
- `cod_dataset.py`
- `evaluate_cod.py`
- VDT2048 可视化报告；
- RGB baseline checkpoint；
- 第一版结果表。

### 通过标准

- 至少 100 组 VDT2048 样本人工检查无错配；
- COD10K 能完成一次训练和测试；
- 指标结果与公开结果量级一致。

如果本周末还没有 RGB baseline，不应继续增加方法模块，必须先修数据管线。

---

## 第 2 周：训练教师并证明 T/D 有效

**时间：10 月 19 日—10 月 25 日**

### 必须训练的教师

| 教师 | 输入 | 用途 |
|---|---|---|
| RGB teacher | RGB | 单模态对照 |
| Early-fusion teacher | RGB-T-D 拼接 | 简单融合对照 |
| Relation teacher | RGB-T-D 三分支 + relation module | 最终教师 |

### 本周实验

\[
\text{RGB teacher}
\quad \text{vs} \quad
\text{RGB-T-D teacher}
\]

同时做：

- 正确 RGB/T/D 配对；
- 随机 T/D 配对。

### 本周必须产出

- 三个 teacher checkpoint；
- VDT2048 验证集结果；
- 正确配对/随机配对结果；
- mask、relation、boundary、uncertainty 可视化。

### 通过标准

必须观察到：

\[
\text{RGB-T-D teacher}
>
\text{RGB teacher}
\]

并且：

\[
\text{Correct pairing}
>
\text{Random pairing}
\]

如果随机配对几乎不降，说明模型没有真正利用跨模态信息，需要立即修改 relation module。

---

## 第 3 周：完成蒸馏对比

**时间：10 月 26 日—11 月 1 日**

### 学生版本

必须训练：

1. No KD；
2. Logit KD；
3. Feature KD；
4. Relation KD；
5. Full model。

### 统一学生输入

所有学生只输入：

\[
I_{rgb}
\]

### 必须比较的蒸馏方式

| 版本 | 监督 |
|---|---|
| No KD | COD/VDT mask |
| Logit KD | 教师 soft mask |
| Feature KD | 中间特征 |
| Relation KD | 目标—背景、目标内部关系 |
| Full | relation + boundary + uncertainty |

### 本周必须产出

- 学生 checkpoint；
- 蒸馏损失曲线；
- VDT2048 验证结果；
- 关系图和边界图；
- 至少一次 COD10K 微调结果。

### 通过标准

核心顺序最好是：

\[
\text{No KD}
<
\text{Logit KD}
<
\text{Relation KD}
\leq
\text{Full}
\]

如果 Relation KD 不如 Logit KD，优先修改关系定义，不要继续加新模块。

---

## 第 4 周：真实 COD 主实验和消融

**时间：11 月 2 日—11 月 8 日**

### COD 训练

在 COD10K train 和 CAMO train 上训练学生。

### 必须完成的主结果

测试：

- COD10K test；
- CAMO；
- NC4K；
- CHAMELEON。

### 必须完成的消融

| 实验 | 去掉内容 |
|---|---|
| Full w/o Logit | logit |
| Full w/o Relation | relation |
| Full w/o Boundary | boundary |
| Full w/o Uncertainty | uncertainty |
| RGB teacher | T/D |
| Random pairing | 正确 T/D 配对 |
| RGB-only pretraining | 多模态教师 |

### 必须完成的困难子集

至少划分：

- 小目标；
- 低对比度；
- 边界复杂；
- 背景纹理密集。

### 本周必须产出

- 主结果表；
- 模块消融表；
- 模态消融表；
- 正确配对/随机配对表；
- 困难子集表；
- 定性可视化图。

### 投稿决策门槛

至少满足：

1. 四个测试集中的三个有提升；
2. Relation KD 优于 Logit KD；
3. 正确配对优于随机配对；
4. 去掉 uncertainty 后困难边界样本下降；
5. same-data RGB-only 对照低于完整方法。

如果这些条件有两项以上不成立，不能再继续扩展方法，应改成短论文或延后投稿。

---

## 第 5 周：锁实验、写论文、提交

**时间：11 月 9 日—11 月 16 日**

| 日期 | 工作 |
|---|---|
| 11 月 9 日 | 固定全部实验数字；不再新增模型 |
| 11 月 10 日 | 完成 CVPR registration 和所有作者 OpenReview 信息 |
| 11 月 11 日 | 完成方法、实验、主表 |
| 11 月 12 日 | 完成引言、相关工作、局限性 |
| 11 月 13 日 | 完成主图、框架图、可视化 |
| 11 月 14 日 | 全文内部审稿，检查所有数字和引用 |
| 11 月 15 日 | 匿名化、格式检查、生成最终 PDF |
| 11 月 16 日 | 提交 CVPR 2027 |
| 11 月 17 日—23 日 | 整理补充材料、代码和额外实验 |

---

# 三、论文必须形成的证据链

论文不能只展示一个最终数字，而要完整回答五个问题。

## 1. 三模态教师真的有用吗？

比较：

\[
T_{RGB}
\quad \text{vs} \quad
T_{RGB-T-D}
\]

## 2. 提升来自关系，还是普通 KD？

比较：

\[
\text{Logit KD}
\quad \text{vs} \quad
\text{Relation KD}
\]

## 3. 关系是跨模态关系吗？

比较：

\[
\text{Correct pairing}
\quad \text{vs} \quad
\text{Random pairing}
\]

## 4. 边界和不确定性有独立作用吗？

逐项去掉：

\[
\mathcal L_{boundary},\quad
\mathcal L_{uncertainty}
\]

## 5. 方法真的迁移到了 COD 吗？

在以下数据集统一测试：

- COD10K；
- CAMO；
- NC4K；
- CHAMELEON。

---

# 四、论文结构

## 1. Introduction

只突出三个问题：

1. 真实 COD 缺少多模态观测；
2. 普通 logits 蒸馏无法处理模态差异；
3. COD 的关键困难是目标—背景关系和边界不确定性。

## 2. Related Work

分三部分：

- Camouflaged object detection；
- Privileged multimodal learning；
- Cross-modal knowledge distillation。

## 3. Method

### 3.1 Problem definition

训练：

\[
(I_{rgb},I_T,I_D,y)
\]

推理：

\[
I_{rgb}\rightarrow \hat y
\]

### 3.2 Tri-modal relation teacher

### 3.3 Relation distillation

### 3.4 Boundary reliability distillation

### 3.5 Conflict-aware uncertainty gating

### 3.6 RGB-only COD adaptation

## 4. Experiments

### 4.1 Datasets and implementation

### 4.2 Comparison with state of the art

### 4.3 Ablation

### 4.4 Cross-modal pairing verification

### 4.5 Difficulty-based analysis

### 4.6 Robustness and efficiency

## 5. Limitation

必须主动承认：

- 学生不能恢复 RGB 中不存在的 T/D 信息；
- VDT2048 与真实 COD 存在域差异；
- 多模态先验在极端域偏移下可能产生负迁移。

主动写出局限性比回避更可信。

---

# 五、工程管理规则

## 每天必须保存

- 配置文件；
- git commit；
- checkpoint；
- 日志；
- seed；
- 结果 CSV；
- 显存和训练时间。

## 每个实验必须记录

```text
experiment_id
dataset
split
backbone
input_size
seed
teacher_type
distillation_type
loss_weights
best_epoch
S_alpha
F_beta_w
E_phi
MAE
Boundary_F
```

## 11 月 9 日之后禁止

- 新增数据集；
- 更换 backbone；
- 修改主损失；
- 临时加入新模块；
- 重新定义测试集；
- 只挑最好 seed 报告。

---

# 六、建议的最终投稿标准

### 最低可投稿线

- 3 个 COD 测试集有提升；
- Relation KD 优于 Logit KD；
- 正确配对优于随机配对；
- 有完整消融。

### 较强 CVPR 竞争线

- 4 个 COD 测试集均有提升；
- 至少一个指标接近或超过当前强基线；
- 困难子集提升明显；
- T/D 噪声和错位实验稳定；
- 学生参数量和推理成本接近 RGB baseline；
- 代码和训练协议可复现。

### 不建议投稿的情况

- 只有 VDT2048 内部提升；
- COD10K/CAMO 没有稳定提升；
- 关系蒸馏不如普通 logits；
- 随机 T/D 配对影响很小；
- 只跑了一个 COD 数据集；
- 结果依赖单个随机种子。

---

从现在起，最重要的不是继续完善概念，而是按以下顺序执行：

\[
\boxed{
\text{数据管线}
\rightarrow
\text{RGB baseline}
\rightarrow
\text{三模态教师}
\rightarrow
\text{关系蒸馏}
\rightarrow
\text{真实 COD}
\rightarrow
\text{消融}
\rightarrow
\text{写作提交}
}
\]

如果现有代码基础为零，CVPR 2027 是高风险冲刺；如果 VDT2048 和 COD 基线已有可用实现，则这个五周计划是可执行的。



# CVPR 2027 完整计划与实验执行方案

CVPR 2027 的官方节点是：论文注册 **2026-11-10 AoE**，论文提交 **2026-11-16 AoE**，补充材料 **2026-11-23 AoE**。[官方时间表](https://cvpr.thecvf.com/Conferences/2027/Dates)

按北京时间约为：

- 注册截止：11 月 11 日晚；
- 论文截止：11 月 17 日晚；
- 补充材料截止：11 月 24 日晚。

下面按“不使用 PST900、不使用合成 COD、只使用 VDT2048 作为多模态知识源”的版本规划。

---

# 一、最终研究方案冻结

## 1. 研究问题

真实 COD 数据通常只有 RGB。我们研究：

\[
\text{真实 VDT2048 RGB-T-D}
\rightarrow
\text{多模态关系教师}
\rightarrow
\text{RGB 学生}
\rightarrow
\text{真实 COD}
\]

目标是让学生在测试时只使用 RGB，但继承：

- 目标—背景关系；
- 目标内部结构；
- 跨模态边界可靠性；
- 模态冲突不确定性。

## 2. 数据范围

### 多模态知识源

只使用：

- VDT2048 train split；
- RGB；
- Thermal；
- Depth；
- mask。

### COD 训练

- COD10K train；
- CAMO train。

### COD 测试

- COD10K test；
- CAMO；
- NC4K；
- CHAMELEON。

### 明确禁止

- PST900；
- 合成 COD；
- 伪 T/D；
- VDT2048 test split；
- COD 测试集参与调参；
- 使用测试集图像生成任何教师信号。

---

# 二、最终模型

## 1. 多模态教师

```text
RGB encoder
Thermal encoder
Depth encoder
        ↓
Cross-modal relation module
        ↓
Multi-scale decoder
        ↓
Mask / Relation / Boundary / Uncertainty
```

教师输出：

\[
p_T,\quad R_T,\quad B_T,\quad U_T
\]

## 2. RGB 学生

```text
RGB encoder
        ↓
Relation predictor
        ↓
Boundary / Uncertainty heads
        ↓
COD decoder
```

测试时：

\[
I_{rgb}\rightarrow p_S
\]

不输入 T/D。

---

# 三、实验执行顺序

## 阶段 A：数据和评价管线

### 必须完成的脚本

```text
datasets/vdt2048.py
datasets/cod10k.py
datasets/camo.py
datasets/nc4k.py
datasets/chameleon.py
evaluation/cod_metrics.py
tools/visualize_rgb_t_d.py
tools/check_alignment.py
configs/*.yaml
```

### VDT2048 检查内容

每个样本检查：

- RGB/T/D 文件是否一一对应；
- 分辨率是否一致；
- mask 是否为空；
- 深度值范围是否正常；
- 热红外是否存在异常全黑/全白；
- 是否存在跨场景重复；
- 官方 train/test 是否混淆。

### 通过标准

- 随机检查至少 200 个样本；
- 至少 20 个样本进行 RGB/T/D/mask 四联图可视化；
- 训练和测试划分自动断言；
- 评价指标与公开实现结果量级一致。

---

# 四、完整实验编号

## A. COD RGB 基线

| 编号 | 实验 | 输入 | 目的 |
|---|---|---|---|
| E00 | U-Net/DeepLab baseline | RGB | 基础分割能力 |
| E01 | SINet-v2 | RGB | 经典 COD 基线 |
| E02 | ZoomNet | RGB | 多尺度 COD 基线 |
| E03 | C2FNet | RGB | 上下文融合基线 |
| E04 | BGNet/FSPNet | RGB | 背景或前景关系基线 |
| E05 | CamoFormer | RGB | Transformer COD 基线 |

至少重新训练 E00、E01 或 E02、E05；其余可以使用官方 checkpoint 并统一评价。

## B. 多模态教师

| 编号 | 实验 | 输入 | 融合方式 |
|---|---|---|---|
| E10 | RGB teacher | RGB | 单模态 |
| E11 | Early-fusion teacher | RGB-T-D | 五通道拼接 |
| E12 | Late-fusion teacher | RGB-T-D | 三分支特征融合 |
| E13 | Relation teacher | RGB-T-D | 三分支 + 关系模块 |

## C. 学生蒸馏

| 编号 | 实验 | 蒸馏内容 |
|---|---|---|
| E20 | No KD | 只有 mask 监督 |
| E21 | Logit KD | soft mask 蒸馏 |
| E22 | Feature KD | 中间特征蒸馏 |
| E23 | RKD | 距离/角度关系蒸馏 |
| E24 | Relation KD | 目标—背景和目标内部关系 |
| E25 | Relation + Boundary | 关系和边界 |
| E26 | Full | 关系、边界、不确定性 |

## D. 真实性与排除实验

| 编号 | 实验 | 目的 |
|---|---|---|
| E30 | Correct pairing | 正确 RGB/T/D 配对 |
| E31 | Random pairing | 排除伪跨模态关系 |
| E32 | RGB-only same-data pretraining | 排除数据量因素 |
| E33 | Fixed distillation weight | 对比不确定性门控 |
| E34 | No relation prior in COD | 验证 COD 阶段关系先验 |
| E35 | T-only teacher | 热红外贡献 |
| E36 | D-only teacher | 深度贡献 |
| E37 | RGB-T teacher | 两模态贡献 |
| E38 | RGB-D teacher | 两模态贡献 |

---

# 五、阶段一：训练多模态教师

## 1. 基础损失

\[
\mathcal L_{seg}^{T}
=
\mathcal L_{BCE}
+
\lambda_{iou}\mathcal L_{IoU}
\]

## 2. 关系表示

对三种模态特征建立：

\[
R_T^{ab}(i,j)
=
\operatorname{sim}(z^a(i),z^b(j))
\]

关系分为：

- 目标内部；
- 目标—局部背景；
- 边界两侧。

\[
\mathcal L_{rel}^{T}
=
\mathcal L_{intra}
+
\mathcal L_{obj-bg}
+
\mathcal L_{boundary-rel}
\]

## 3. 边界表示

\[
B_T=h(\nabla I_{rgb},\nabla I_T,\nabla I_D,p_T)
\]

\[
\mathcal L_{bd}^{T}
=
\operatorname{BCE}(B_T,b_y)
+
\lambda_g\|\nabla B_T-\nabla b_y\|_1
\]

## 4. 不确定性

\[
U_T(i)
=
\operatorname{Var}
[
p_{rgb}(i),p_T(i),p_D(i)
]
\]

教师总损失：

\[
\mathcal L_T
=
\mathcal L_{seg}^{T}
+
\lambda_r\mathcal L_{rel}^{T}
+
\lambda_b\mathcal L_{bd}^{T}
+
\lambda_u\mathcal L_U^{T}
\]

## 5. 阶段一必须保存

每个 VDT2048 样本保存：

```text
teacher_mask.pt
teacher_relation.pt
teacher_boundary.pt
teacher_uncertainty.pt
teacher_quality.pt
```

---

# 六、阶段二：训练 RGB 学生

教师输入：

\[
(I_{rgb},I_T,I_D)
\]

学生输入：

\[
I_{rgb}
\]

## 1. Logit 蒸馏

\[
\mathcal L_{logit}
=
\tau^2
KL\left[
\sigma(p_T/\tau)
\|
\sigma(p_S/\tau)
\right]
\]

## 2. 关系蒸馏

\[
\mathcal L_{rel}
=
\left\|
\Pi_S(R_S)-\Pi_T(R_T)
\right\|_1
\]

## 3. 边界蒸馏

\[
\mathcal L_{bd}
=
\operatorname{BCE}(B_S,B_T)
+
\lambda_g\|\nabla B_S-\nabla B_T\|_1
\]

## 4. 不确定性门控

\[
w_i=\exp(-\gamma U_T(i))
\]

\[
\mathcal L_{distill}
=
\sum_i w_i
[
\mathcal L_{logit}(i)
+
\lambda_r\mathcal L_{rel}(i)
+
\lambda_b\mathcal L_{bd}(i)
]
+
\lambda_u\|U_S-U_T\|_1
\]

## 5. 学生训练损失

\[
\mathcal L_{student}
=
\mathcal L_{BCE}(p_S,y)
+
\lambda_{iou}\mathcal L_{IoU}(p_S,y)
+
\lambda_d\mathcal L_{distill}
\]

阶段二必须完成：

- E20；
- E21；
- E22；
- E23；
- E24；
- E26。

---

# 七、阶段三：真实 COD 微调

## 1. 初始化

从阶段二加载：

- RGB encoder；
- relation predictor；
- boundary head；
- uncertainty head；
- COD decoder。

删除：

- Thermal encoder；
- Depth encoder；
- 多模态融合层。

## 2. COD 损失

\[
\mathcal L_{COD}
=
\mathcal L_{BCE}
+
\lambda_{iou}\mathcal L_{IoU}
+
\lambda_{ssim}\mathcal L_{SSIM}
\]

## 3. 关系先验

从 VDT2048 提取：

\[
\mathcal P=
\{
P_{obj-bg},P_{obj-obj},P_{boundary}
\}
\]

COD 学生输出关系：

\[
R_S^{COD}
\]

\[
\mathcal L_{prior}
=
d(R_S^{COD},\mathcal P)
\]

关系先验权重初始设置较小：

\[
\lambda_{prior}\in[0.01,0.1]
\]

并随训练衰减：

\[
\lambda_{prior}(t)
=
\lambda_{prior,0}
\left(1-\frac{t}{T}\right)
\]

最终损失：

\[
\mathcal L_{COD-final}
=
\mathcal L_{COD}
+
\lambda_b\mathcal L_{bd}^{COD}
+
\lambda_p\mathcal L_{prior}
\]

---

# 八、五周详细计划表

## 第 1 周：10 月 12 日—10 月 18 日

| 日期 | 工作 |
|---|---|
| 10/12 | ✅ 已完成：创建实验目录、配置文件和结果记录表 |
| 10/13 | 完成 VDT2048 数据读取和四联图 |
| 10/14 | 完成 COD10K、CAMO、NC4K、CHAMELEON 读取 |
| 10/15 | 完成指标计算和统一评价脚本 |
| 10/16 | 训练 E00 RGB baseline |
| 10/17 | 运行 E01/E02 中至少一个强 COD baseline |
| 10/18 | 固定第一版 baseline 表和数据协议 |

### 周目标

- 数据无错；
- RGB baseline 跑通；
- 评价脚本稳定；
- 不再修改数据划分。

---

## 第 2 周：10 月 19 日—10 月 25 日

| 日期 | 工作 |
|---|---|
| 10/19 | 训练 E10 RGB teacher |
| 10/20 | 训练 E11 early-fusion teacher |
| 10/21 | 训练 E12 late-fusion teacher |
| 10/22 | 实现 relation module |
| 10/23 | 训练 E13 relation teacher |
| 10/24 | 正确配对/随机配对实验 |
| 10/25 | 教师可视化和阶段性结论 |

### 周目标

必须满足：

\[
E13>E10
\]

且：

\[
\text{Correct pairing}>\text{Random pairing}
\]

---

## 第 3 周：10 月 26 日—11 月 1 日

| 日期 | 工作 |
|---|---|
| 10/26 | 实现 E20 No KD |
| 10/27 | 实现 E21 Logit KD |
| 10/28 | 实现 E22 Feature KD |
| 10/29 | 实现 E23 RKD |
| 10/30 | 实现 E24 Relation KD |
| 10/31 | 完成 boundary distillation |
| 11/01 | 完成 uncertainty gate 和 E26 Full |

### 周目标

至少得到：

\[
E24>E21
\]

否则暂停扩展，重新检查 relation loss。

---

## 第 4 周：11 月 2 日—11 月 8 日

| 日期 | 工作 |
|---|---|
| 11/02 | COD10K 微调 E20、E21、E24、E26 |
| 11/03 | CAMO 微调和测试 |
| 11/04 | NC4K、CHAMELEON 测试 |
| 11/05 | 去 relation、去 boundary、去 uncertainty |
| 11/06 | RGB-only same-data pretraining |
| 11/07 | T-only、D-only、RGB-T、RGB-D 消融 |
| 11/08 | 困难子集和可视化 |

### 周目标

至少满足：

- 四个测试集中的三个有提升；
- Relation KD 优于 Logit KD；
- Full 在困难子集上明显提升；
- same-data RGB-only 低于 Full。

---

## 第 5 周：11 月 9 日—11 月 16 日

| 日期 | 工作 |
|---|---|
| 11/09 | 锁定实验数字和主结论 |
| 11/10 | 完成 CVPR 注册和 OpenReview 作者信息 |
| 11/11 | 写方法和实验章节 |
| 11/12 | 写引言、相关工作、局限性 |
| 11/13 | 完成框架图、损失图、可视化 |
| 11/14 | 内部审稿和数字检查 |
| 11/15 | 匿名化、格式检查、生成最终 PDF |
| 11/16 | 完成提交 |
| 11/17–11/23 | 补充材料和代码整理 |

---

# 九、论文必须准备的图表

## 正文图

1. **整体框架图**  
   展示教师使用 RGB-T-D，学生只使用 RGB。

2. **关系蒸馏图**  
   展示目标内部、目标—背景和边界关系。

3. **不确定性门控图**  
   展示模态一致、模态冲突和对应蒸馏权重。

4. **定性结果图**  
   展示 RGB、T、D、baseline、Full、GT。

## 正文表

1. COD 四数据集主结果；
2. 蒸馏方式消融；
3. 模态消融；
4. 随机配对和 same-data 对照；
5. 参数量、FLOPs、FPS。

## 补充材料

- 完整训练细节；
- 额外随机种子；
- 超参数敏感性；
- T/D 噪声和错位；
- 更多困难案例；
- 完整基线结果；
- 代码和配置文件。

---

# 十、最终投稿门槛

## 必须达到

- RGB-T-D 教师优于 RGB teacher；
- 正确配对优于随机配对；
- Relation KD 优于 Logit KD；
- Full 优于 No KD；
- COD10K、CAMO、NC4K、CHAMELEON 至少三个数据集提升；
- 学生测试时只使用 RGB；
- 无合成 COD；
- 无 PST900；
- 所有关键结论至少运行 3 个随机种子。

## 如果结果不理想

### 情况一：教师没有优于 RGB

优先修复：

- RGB/T/D 配准；
- 深度归一化；
- Thermal 质量；
- 模态分支容量；
- relation module。

### 情况二：Logit KD 优于 Relation KD

简化关系定义，只保留：

- 目标—背景关系；
- 目标内部关系；
- 不做过细的像素级跨模态特征对齐。

### 情况三：VDT2048 有提升，COD 没提升

降低：

\[
\lambda_{prior}
\]

并让 COD 阶段主要使用：

- RGB encoder 初始化；
- boundary head；
- uncertainty head；

关系先验只作为弱正则。

### 情况四：只有单个数据集提升

不能把结论写成普适 COD 方法，应改为：

> 对低对比度和复杂边界 COD 的结构先验迁移。

---

# 十一、每天的实验记录格式

每次实验必须记录：

```text
experiment_id:
dataset:
split:
backbone:
input_size:
seed:
teacher_type:
student_type:
distillation_type:
lambda_logit:
lambda_relation:
lambda_boundary:
lambda_uncertainty:
lambda_prior:
best_epoch:
S_alpha:
F_beta_w:
E_phi:
MAE:
Boundary_F:
params:
FLOPs:
FPS:
checkpoint:
```

所有主表从 CSV 自动生成，不能手工抄数字。

---

# 十二、核心执行原则

从现在开始，研究路线固定为：

\[
\boxed{
\text{VDT2048 数据验证}
\rightarrow
\text{RGB/COD 基线}
\rightarrow
\text{RGB-T-D 教师}
\rightarrow
\text{关系蒸馏}
\rightarrow
\text{真实 COD 微调}
\rightarrow
\text{消融与鲁棒性}
\rightarrow
\text{CVPR 投稿}
}
\]

最重要的不是继续增加模块，而是尽快得到三组决定性结果：

\[
\boxed{
\text{RGB-T-D teacher}>\text{RGB teacher}
}
\]

\[
\boxed{
\text{Relation KD}>\text{Logit KD}
}
\]

\[
\boxed{
\text{Full}>\text{No KD on real COD}
}
\]

这三组结果成立，方案才具备冲击 CVPR 2027 的基础。
