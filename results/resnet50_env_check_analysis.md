# ResNet-50 A/B/C/D 环境检查结果

运行命令：

```bash
CUDA_VISIBLE_DEVICES=1 python tools/run_triprd_ab.py \
  --model-variant resnet50 --pretrained \
  --groups A,B,C,D --epochs 1 --vdt-epochs 1 \
  --max-samples 16 --max-test-samples 4 --batch-size 1 \
  --image-size 352 --base-channels 256 \
  --test-datasets COD10K --device cuda \
  --out results/resnet50_env_check
```

这是一次流程和环境检查，不是正式性能实验：每个阶段只训练 1 个 epoch，训练样本上限为 16，COD10K 测试只取 4 张图。

## 结果摘要

| 组别 | COD 验证 Dice | COD10K 测试 Dice | COD10K 测试 MAE |
|---|---:|---:|---:|
| A | 0.2529 | 0.3845 | 0.4607 |
| B | **0.2732** | **0.4116** | **0.4434** |
| C | 0.2500 | 0.3798 | 0.5191 |
| D | 0.2714 | 0.3911 | 0.4451 |

本次排序为 B > D > A > C。相对于 A，B 的测试 Dice 增加 0.0271；C 比 B 低 0.0318；D 比 C 高 0.0113，但仍比 B 低。由于样本量和训练轮数很小，这些差异只能作为环境检查信号，不能作为最终模型排名或论文结论。

日志中的 `iou` 是 `1 - soft IoU` 损失，数值越小越好；`dice` 是 sigmoid 概率上的 soft Dice。官方 COD 指标在这次运行中显示 `unavailable`，原因是评估汇总阶段读取了不存在的 `adp` 键，已记录为后续评估修复项；模型预测文件已经生成。

## 四轮实验分别做什么

### A：COD RGB 监督基线

学生模型只使用 RGB 图像，在 COD 训练集上直接进行监督训练，再在 COD 验证集和测试集上评估。它提供最基本的 RGB 基线，用来衡量预训练和蒸馏是否带来增益。

### B：VDT RGB 监督预训练 → COD 微调

学生模型先只使用 VDT 的 RGB 图像和显著性标注进行监督预训练，然后使用 COD RGB 数据继续微调。它检验跨数据集的 RGB 监督预训练本身是否有迁移价值，也是蒸馏组最重要的对照。

### C：VDT 三模态教师的 logits 蒸馏 → COD 微调

先训练一个使用 RGB、热红外和深度的三模态教师；学生仍然只有 RGB 输入。VDT 阶段除了真实标注损失，还加入教师与学生输出 logits 的蒸馏损失，之后再进行 COD 微调。它单独检验输出层知识蒸馏的效果。

### D：VDT 完整关系、边界和不确定性门控蒸馏 → COD 微调

在 C 的 logits 蒸馏基础上，再加入特征关系蒸馏和边界蒸馏，并根据教师三模态预测之间的不一致程度计算门控权重，降低不可靠教师信号的影响。之后同样进行 COD 微调。它检验完整 TriPRD 蒸馏机制的综合效果。

## 后续正式实验建议

先修复官方指标读取问题，再增加训练轮数和样本量；正式比较应重点报告 B、C、D 相对 B 的变化，并使用完整测试集和多个随机种子。若要单独验证不确定性门控，还应增加关闭门控但保留其他蒸馏项的对照组。
