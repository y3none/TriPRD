# 不使用 PST900 的最终方案

> 状态说明（2026-10-16）：以下保留为原始设计草案，不代表所列模块已实现，也不再按其 20 周时间表执行。当前研究边界、实现缺口和下一版最小验证机制以 [paper_scope.md](paper_scope.md) 为准，投稿排程以 [CVPR 2027 计划](cvpr2027_submission_plan.md) 为准。用户已明确要求以 TriPRD 自身方法冲击 SOTA；完整现成 COD 网络加普通 KD 不作为主方法替代方案。

## 一、最终研究定位

### 题目

**Privileged RGB-T-D Relation Distillation for RGB-Only Camouflaged Object Detection**

核心设定：

\[
\text{VDT2048 真实 RGB-T-D}
\rightarrow
\text{多模态教师}
\rightarrow
\text{RGB 学生}
\rightarrow
\text{真实 COD 微调}
\]

最终模型只输入 RGB，不使用：

- PST900；
- 合成 COD；
- 伪热红外；
- 伪深度；
- COD 测试集中的任何 T/D 信息。

VDT2048只作为**多模态结构知识源**，不作为 COD 测试集。

---

# 二、数据使用方式

## 2.1 VDT2048

使用 VDT2048 的：

- RGB；
- Thermal；
- Depth；
- 目标 mask；
- 官方训练划分。

用途：

1. 训练 RGB-T-D 教师；
2. 训练 RGB 学生；
3. 提取目标—背景关系原型；
4. 学习跨模态边界可靠性；
5. 学习模态冲突不确定性。

严格要求：

- 只使用官方 train split；
- 官方 test split 不能参与训练；
- 同一场景或序列不能跨 split；
- T/D 缺失和错位只作为鲁棒性实验，不伪造缺失模态。

## 2.2 COD 数据

训练：

- COD10K train；
- CAMO train。

测试：

- COD10K test；
- CAMO test；
- NC4K；
- CHAMELEON。

COD 数据只提供：

\[
(I_{rgb}, y)
\]

不提供 T/D。

---

# 三、网络结构

## 3.1 RGB-T-D 教师

```text
RGB encoder
Thermal encoder
Depth encoder
        ↓
Cross-modal relation module
        ↓
Multi-scale decoder
        ↓
Mask / Relation / Boundary / Uncertainty heads
```

教师输出：

\[
p_T,\ R_T,\ B_T,\ U_T
\]

分别表示：

- mask 预测；
- 跨模态关系；
- 边界可靠性；
- 模态冲突不确定性。

推荐 backbone：

- ResNet-50；
- 或 PVT-v2-B2。

建议初版使用 ResNet-50，先降低实验复杂度；最终投稿版本可增加 PVT-v2-B2 结果。

## 3.2 RGB 学生

```text
RGB encoder
        ↓
RGB relation predictor
        ↓
Boundary / Uncertainty heads
        ↓
COD decoder
        ↓
RGB-only mask
```

学生不包含：

- Thermal encoder；
- Depth encoder；
- T/D fusion module。

最终推理只使用 RGB。

---

# 四、阶段一：训练 VDT2048 多模态教师

## 4.1 分割损失

\[
\mathcal L_{seg}
=
\mathcal L_{BCE}
+
\lambda_{iou}\mathcal L_{IoU}
\]

## 4.2 跨模态关系

对 RGB、T、D 特征建立关系：

\[
R_T^{ab}(i,j)
=
\operatorname{sim}
\left(
z^a(i),z^b(j)
\right),
\quad a,b\in\{rgb,T,D\}
\]

只保留三种对 COD 有用的关系：

1. 目标内部关系；
2. 目标—局部背景关系；
3. 边界两侧关系。

教师关系损失：

\[
\mathcal L_{rel}^{T}
=
\mathcal L_{intra}
+
\mathcal L_{obj-bg}
+
\mathcal L_{boundary-rel}
\]

## 4.3 边界可靠性

由 mask 边界和三模态梯度共同生成：

\[
B_T(i)=
h\left(
\nabla I_{rgb}(i),
\nabla I_T(i),
\nabla I_D(i),
p_T(i)
\right)
\]

\[
\mathcal L_{bd}^{T}
=
\operatorname{BCE}(B_T,b_y)
+
\lambda_g\|\nabla B_T-\nabla b_y\|_1
\]

## 4.4 模态冲突不确定性

\[
U_T(i)
=
\operatorname{Var}
[
p_{rgb}(i),p_T(i),p_D(i)
]
\]

可加入特征冲突和梯度冲突：

\[
U_T
=
\alpha U_{pred}
+
\beta U_{feat}
+
\gamma U_{edge}
\]

教师总损失：

\[
\mathcal L_T
=
\mathcal L_{seg}
+
\lambda_r\mathcal L_{rel}^{T}
+
\lambda_b\mathcal L_{bd}^{T}
+
\lambda_u\mathcal L_U^{T}
\]

训练完成后冻结教师。

---

# 五、阶段二：VDT2048 上的 RGB 学生蒸馏

教师输入：

\[
(I_{rgb},I_T,I_D)
\]

学生输入：

\[
I_{rgb}
\]

## 5.1 Logit 蒸馏

\[
\mathcal L_{logit}
=
\tau^2
\operatorname{KL}
\left[
\sigma(p_T/\tau)
\|
\sigma(p_S/\tau)
\right]
\]

## 5.2 关系蒸馏

教师关系投影到学生空间：

\[
\tilde R_T=\Pi_T(R_T)
\]

学生关系：

\[
R_S=\Pi_S(z_S^{rgb})
\]

\[
\mathcal L_{rel}
=
\|R_S-\tilde R_T\|_1
\]

## 5.3 边界蒸馏

\[
\mathcal L_{bd}
=
\operatorname{BCE}(B_S,B_T)
+
\lambda_g
\|\nabla B_S-\nabla B_T\|_1
\]

## 5.4 不确定性门控

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

学生监督：

\[
\mathcal L_{sup}
=
\mathcal L_{BCE}(p_S,y)
+
\lambda_{iou}\mathcal L_{IoU}(p_S,y)
\]

总损失：

\[
\mathcal L_{VDT}
=
\mathcal L_{sup}
+
\lambda_d\mathcal L_{distill}
\]

---

# 六、阶段三：真实 COD 微调

使用 COD10K 和 CAMO 的真实 RGB 图像与 mask。

## 6.1 初始化

从阶段二加载：

- RGB encoder；
- relation predictor；
- boundary head；
- uncertainty head；
- COD decoder。

删除教师的：

- Thermal encoder；
- Depth encoder；
- 多模态融合分支。

## 6.2 COD 损失

\[
\mathcal L_{COD}
=
\mathcal L_{BCE}
+
\lambda_{iou}\mathcal L_{IoU}
+
\lambda_{ssim}\mathcal L_{SSIM}
\]

## 6.3 关系先验

从 VDT2048 中提取：

\[
\mathcal P=
\{
P_{obj-bg},
P_{obj-obj},
P_{boundary}
\}
\]

COD 学生关系：

\[
R_S^{COD}
\]

关系先验：

\[
\mathcal L_{prior}
=
d(R_S^{COD},\mathcal P)
\]

为了避免 VDT2048 和 COD 的域差异，采用较小权重：

\[
\lambda_{prior}\in[0.01,0.1]
\]

并在训练过程中衰减：

\[
\lambda_{prior}(t)
=
\lambda_{prior,0}
\left(
1-\frac{t}{T}
\right)
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

# 七、必须做的基线

## 7.1 RGB COD 基线

必须比较：

- RGB U-Net/DeepLabV3+；
- SINet-v2；
- ZoomNet；
- C2FNet；
- BGNet；
- FSPNet；
- CamoFormer。

至少需要重新训练：

- RGB baseline；
- SINet-v2 或 ZoomNet 中的一个强 CNN；
- CamoFormer 或其他 Transformer 基线。

其余方法可以使用官方结果或官方 checkpoint，但必须统一评价脚本。

## 7.2 教师基线

| 教师 | 输入 | 目的 |
|---|---|---|
| RGB teacher | RGB | 排除 T/D 作用 |
| Early-fusion teacher | RGB-T-D 五通道拼接 | 简单融合基线 |
| Late-fusion teacher | 三分支特征融合 | 普通多模态融合 |
| Ours teacher | 三分支 + relation module | 验证关系建模 |

## 7.3 蒸馏基线

| 方法 | 蒸馏内容 |
|---|---|
| No KD | 无蒸馏 |
| Logit KD | 教师 mask 概率 |
| Feature KD | 中间特征 |
| RKD | 距离和角度关系 |
| C²KD-style | 选择性跨模态蒸馏 |
| Relation KD | 目标—背景关系 |
| Relation + Boundary | 关系和边界 |
| Full | 关系、边界、不确定性 |

## 7.4 必须做的真实性对照

### 正确配对 vs 随机配对

正确：

\[
(I_{rgb},I_T,I_D)
\]

随机：

\[
(I_{rgb},I_T^{wrong},I_D^{wrong})
\]

预期：

\[
\text{correct pairing}>\text{random pairing}
\]

### 同数据量 RGB 预训练

```text
RGB-only VDT pretraining
```

与：

```text
RGB-T-D teacher → RGB student
```

使用相同图像数量和训练预算。

### 固定蒸馏权重 vs 冲突门控

比较：

\[
w_i=1
\]

和：

\[
w_i=\exp(-\gamma U_T(i))
\]

---

# 八、实验结果表设计

## 8.1 主结果表

| Method | Test Input | VDT Pretrain | CAMO | COD10K | NC4K | CHAMELEON |
|---|---|---:|---:|---:|---:|---:|
| SINet-v2 | RGB | No |  |  |  |  |
| ZoomNet | RGB | No |  |  |  |  |
| BGNet | RGB | No |  |  |  |  |
| CamoFormer | RGB | No |  |  |  |  |
| RGB baseline | RGB | No |  |  |  |  |
| RGB teacher → RGB student | RGB | Yes |  |  |  |  |
| RGB-T-D logit KD | RGB | Yes |  |  |  |  |
| C²KD-style | RGB | Yes |  |  |  |  |
| Ours | RGB | Yes |  |  |  |  |

指标分别报告：

- \(S_\alpha\)；
- \(F_\beta^w\)；
- \(E_\phi\)；
- MAE；
- Boundary F。

## 8.2 消融表

| Setting | Logit | Relation | Boundary | Uncertainty | \(S_\alpha\) | \(F_\beta^w\) | MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| No KD |  |  |  |  |  |  |  |
| Logit KD | ✓ |  |  |  |  |  |  |
| Feature KD |  | Feature |  |  |  |  |  |
| Relation KD |  | ✓ |  |  |  |  |  |
| Relation + Boundary |  | ✓ | ✓ |  |  |  |  |
| Full | ✓ | ✓ | ✓ | ✓ |  |  |  |

## 8.3 模态消融表

| Teacher | RGB | T | D | \(S_\alpha\) | \(F_\beta^w\) |
|---|---:|---:|---:|---:|---:|
| RGB | ✓ |  |  |  |  |
| RGB-T | ✓ | ✓ |  |  |  |
| RGB-D | ✓ |  | ✓ |  |  |
| RGB-T-D | ✓ | ✓ | ✓ |  |  |

---

# 九、计划表

以下按 20 周安排，适合单卡 4090 或同等级 GPU 的研究节奏。

| 周次 | 任务 | 产出 | 通过标准 |
|---|---|---|---|
| 第 1 周 | 整理 VDT2048 数据、检查 RGB/T/D 对齐和 mask | 数据读取脚本、可视化 | 随机抽样 200 组无读取错误 |
| 第 2 周 | 建立统一训练框架和 COD 评价脚本 | 训练配置、指标代码 | 能复现 MAE、\(S_\alpha\)、\(F_\beta^w\) |
| 第 3 周 | 训练 RGB baseline | RGB baseline 结果 | 结果与公开范围基本一致 |
| 第 4 周 | 重训 SINet-v2/ZoomNet | 强 CNN baseline | 主指标稳定 |
| 第 5 周 | 训练 RGB teacher | RGB teacher checkpoint | VDT2048 验证集收敛 |
| 第 6 周 | 训练 early-fusion RGB-T-D teacher | early fusion 结果 | 优于 RGB teacher |
| 第 7 周 | 训练 late-fusion teacher | late fusion 结果 | 与 early fusion 公平比较 |
| 第 8 周 | 实现 relation module | 关系图可视化 | 目标内部关系具有空间连续性 |
| 第 9 周 | 加入 boundary head | 边界预测结果 | Boundary F 明显高于无边界版本 |
| 第 10 周 | 加入 uncertainty head | 冲突图和置信度图 | 冲突区域不确定性更高 |
| 第 11 周 | 完成 RGB-T-D 教师训练 | 最终教师 checkpoint | 教师完整损失稳定下降 |
| 第 12 周 | 实现 logit KD、feature KD、RKD | 蒸馏基线结果 | 蒸馏训练稳定 |
| 第 13 周 | 实现 C²KD-style baseline | 跨模态蒸馏对照 | 可复现选择性蒸馏 |
| 第 14 周 | 训练 relation KD student | 关系蒸馏结果 | 优于 logit KD |
| 第 15 周 | 加入 boundary 和 uncertainty gate | 完整学生预训练 | 困难区域提升 |
| 第 16 周 | COD10K 微调 | COD10K 结果 | 超过 RGB-only baseline |
| 第 17 周 | CAMO 微调和跨数据集测试 | CAMO/NC4K/CHAMELEON | 至少 3 个数据集有提升 |
| 第 18 周 | 完成全部消融和随机配对实验 | 消融表、真实性表 | 关系、边界、门控均有独立贡献 |
| 第 19 周 | 复杂度、鲁棒性和困难子集实验 | 参数量、FPS、噪声实验 | 结论稳定 |
| 第 20 周 | 可视化、论文图表和代码清理 | 初稿、补充材料、代码 | 形成完整投稿包 |

---

# 十、每个阶段的决策门槛

为了避免做到最后才发现主线不成立，设置三个判断点。

## 第 8 周判断

如果：

\[
\text{RGB-T-D teacher}
\leq
\text{RGB teacher}
\]

说明多模态教师没有学到有效互补关系，需要先修复模态对齐和质量处理。

## 第 15 周判断

如果：

\[
\text{Relation KD}
\leq
\text{Logit KD}
\]

说明关系设计没有优于普通蒸馏，需要重新定义：

- 目标—背景关系；
- 边界两侧关系；
- 局部背景环；
- 关系投影方式。

## 第 17 周判断

如果完整方法只在 VDT2048 上有效、COD10K/CAMO 没有提升，说明存在严重域差，应降低关系先验权重或只保留可迁移的结构关系。

---

# 十一、最终投稿版本必须满足

### 方法层面

- RGB-T-D 教师；
- RGB 学生；
- 跨模态目标—背景关系；
- 边界可靠性蒸馏；
- 冲突感知不确定性门控；
- 不使用合成 COD。

### 实验层面

- COD10K；
- CAMO；
- NC4K；
- CHAMELEON；
- 强 COD 基线；
- 普通 KD；
- RKD；
- C²KD-style；
- 同数据量 RGB 预训练；
- 随机 T/D 配对；
- T/D 缺失、噪声和错位鲁棒性。

### 论文结论应保持克制

不要写：

> RGB 学生获得了完整三模态感知能力。

应写：

> RGB 学生继承了由真实 RGB-T-D 数据诱导的目标—背景结构先验、边界可靠性和冲突感知策略。

这个版本的核心优势是：**不依赖合成 COD，训练和测试任务边界清晰，方法创新集中在可验证的跨模态关系迁移上，实验能够排除普通蒸馏和数据量因素。**
