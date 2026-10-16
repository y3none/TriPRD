# TriPRD 显式关系首轮实验汇总

登记日期：2026-10-16（Asia/Singapore）。范围：`triprd_relation_v1`，`results/optimization/relation/seed-20261012` 下四份 comparison；新实现的首轮单种子验证筛选，尚无测试集结果。历史基线见 [optimization_summary.md](optimization_summary.md)。

## 结论

流程完成并且元数据内部一致，但多模态关系蒸馏尚未带来主指标收益。D 对 A 的宏 Sm 差值为 **-0.001036**，对 C 为 **-0.000249**；D 的 MAE 略好，不能据此更换已固定的选模主指标。多模态教师 VDT soft Dice 高出 RGB 教师约 4.96 个百分点，收益没有传到本轮源学生或 COD 主指标。仅一个种子，不能声称稳定胜负、显著性或达到 SOTA。

## 运行协议与控制变量

以下来自本轮 comparison.protocol 及对应源码。四组为同结构 RGB 学生，参数量 28,138,562，其中关系分支 104,448；C/D 三编码器教师均为 75,154,626。A 不做 VDT；B 做 RGB 监督；C 与 D 仅在教师输入模态上作匹配比较（C 为 RGB/RGB/RGB，D 为 RGB/T/D）。学生在 VDT 阶段冻结 encoder 参数及 BN 状态，COD 阶段恢复训练；仅关系分支迁移，decoder/head 与主干以 A 相同初始状态重新构建。C/D 源学生使用 BCE+IoU 和权重 1、温度 1 的局部分类 KL；B 与所有 COD 阶段只有 BCE+IoU，不启用 logit/boundary KD。关系在 stride-16 特征上预测两组 3×3 邻域分布并参与残差聚合，关系维度 32。

COD 训练/划分 seed 均为 20261012；ResNet-50 ImageNet，352 输入，base channels 256，batch 8；AdamW lr=1e-4、weight decay=1e-4，encoder LR×0.1，cosine、warmup 5、min ratio 0.05。max_samples=0，本轮 cod_train_mode=split：3232 张训练、808 张验证（COD10K 608、CAMO 200），不是全部 4040 张的最终训练。B/C/D 的 VDT 为 838 训练、210 验证，各源阶段 100 轮；teacher/source 以缩放尺寸 soft Dice 选模。COD 为 100 轮，在第 5、10…100 轮计算正式验证指标，共 20 个候选 checkpoint；四组均选 epoch 70 的来源等权宏 Sm。未评估的轮次不能声称其 Sm 更低。

验证先按来源逐图平均，再以 COD10K/CAMO 各 50% 合成宏值。预测经 uint8→PIL 双线性恢复原 GT 尺寸；指标内部逐图 min-max 归一化。Sm/Fw/mean E 越高越好，MAE 越低越好；E_phi 是 256 阈值曲线均值，不是 max/adaptive E。

完整逐运行 protocol、源文件哈希、checkpoint 路径和冻结检查见 E011–E014。四组均记录 NVIDIA GeForce RTX 4090、Python 3.11.17、PyTorch 2.7.1+cu128、CUDA 12.8；队列采用独立进程而非四卡 DDP。comparison 不含物理 GPU 卡号，不能从示例命令推断实际分配。

## COD 验证结果

各行的四项指标均来自该组 Sm 选中的 epoch 70，非每项分别取最优。

| 组别 / 记录 | Epoch | Sm ↑ | Fw ↑ | mean E ↑ | MAE ↓ |
| --- | --- | --- | --- | --- | --- |
| [A / E011_relation_A_scratch](E011_relation_A_scratch.md) | 70 | 0.808668 | 0.703867 | 0.868845 | 0.048991 |
| [B / E012_relation_B_rgb_source](E012_relation_B_rgb_source.md) | 70 | 0.807764 | 0.701407 | 0.869505 | 0.049096 |
| [C / E013_relation_C_rgb_relation](E013_relation_C_rgb_relation.md) | 70 | 0.807881 | 0.701479 | 0.870593 | 0.048955 |
| [D / E014_relation_D_multimodal_relation](E014_relation_D_multimodal_relation.md) | 70 | 0.807632 | 0.703131 | 0.869142 | 0.048766 |

分来源指标：

| 组别 | 来源 | 样本数 | Sm ↑ | Fw ↑ | mean E ↑ | MAE ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| A | COD10K | 608 | 0.798720 | 0.663441 | 0.868147 | 0.035816 |
| A | CAMO | 200 | 0.818615 | 0.744294 | 0.869542 | 0.062167 |
| B | COD10K | 608 | 0.797182 | 0.660621 | 0.871013 | 0.035981 |
| B | CAMO | 200 | 0.818346 | 0.742193 | 0.867997 | 0.062212 |
| C | COD10K | 608 | 0.798671 | 0.663328 | 0.871783 | 0.035980 |
| C | CAMO | 200 | 0.817091 | 0.739630 | 0.869404 | 0.061930 |
| D | COD10K | 608 | 0.798369 | 0.663506 | 0.870419 | 0.036005 |
| D | CAMO | 200 | 0.816895 | 0.742756 | 0.867866 | 0.061528 |

D−A：

| 来源 | ΔSm ↑ | ΔFw ↑ | Δmean E ↑ | ΔMAE ↓ |
| --- | --- | --- | --- | --- |
| macro | -0.001036 | -0.000736 | +0.000298 | -0.000225 |
| COD10K | -0.000351 | +0.000066 | +0.002272 | +0.000189 |
| CAMO | -0.001720 | -0.001539 | -0.001677 | -0.000639 |

D−C：

| 来源 | ΔSm ↑ | ΔFw ↑ | Δmean E ↑ | ΔMAE ↓ |
| --- | --- | --- | --- | --- |
| macro | -0.000249 | +0.001652 | -0.001451 | -0.000189 |
| COD10K | -0.000302 | +0.000178 | -0.001364 | +0.000025 |
| CAMO | -0.000196 | +0.003126 | -0.001538 | -0.000402 |

D−A 的两来源 Sm 都下降；宏 MAE 改善来自 CAMO，而 COD10K MAE 反而 +0.000189。C 的 mean E 最高、A 的 Sm/Fw 最高，避免将局部指标优胜包装为整体胜出。

## 教师变强与传递不足

| 组别 | Teacher epoch | Teacher val soft Dice | Source epoch | Source val soft Dice | 教师关系熵 | 学生关系熵 | Source val KL |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B | — | — | 75 | 0.830162 | — | 0.982075 | 0.000000 |
| C | 89 | 0.855075 | 89 | 0.829510 | 0.995166 | 0.998245 | 0.006672 |
| D | 69 | 0.904641 | 90 | 0.827633 | 0.990839 | 0.996127 | 0.011142 |

teacher/source 各以自身 VDT val soft Dice 选模。教师熵来自固定最佳 teacher 在 source 最佳轮的验证诊断，与 teacher 自己最佳轮的 val 熵一致；均使用前向温度 1、按有效邻居数归一化。KD 温度=1、权重=1，weighted KL 与 KL 相等；B 不做 KD。

多模态教师的本轮 VDT soft Dice 高于 RGB 教师（0.904641 对 0.855075），但对应源学生为 0.827633 对 0.829510，且无 KD 的 B 为 0.830162。教师变强没有转化为本轮学生或 COD 主指标优势。关系归一化熵接近 1 提示平均邻域分布接近均匀；这是平均诊断，不能证明每个位置坍缩，也不能推断强制变尖必然有利。

代码中的教师还可通过多尺度 RGB/T/D 均值融合及 T/D value 特征获得分割收益，而学生 KD 目标只有局部关系分布；所以“教师更强”并不保证“关系目标包含更多有用知识”。这是基于实现的信息路径假设，不是已经由干预实验确认的因果结论。

源学生 C 的训练 KL 从第 1 轮 0.011031 降至第 100 轮 0.007389，D 从 0.017057 降至 0.010576。KL 非零且首末轮下降，说明蒸馏目标参与了优化；这不能证明已传递有益知识或期间单调下降。

COD 选中轮的学生关系熵分别为 A 0.908685、B 0.823990、C 0.857770、D 0.858180。B/C/D 的 COD 熵低于各自 VDT 最佳轮；A 没有 VDT 阶段。跨数据集、跨训练阶段的熵差不能证明遗忘，本轮性能也未随熵降低而改善，不能用熵作为替代目标。

## 收敛与运行

| 组别 | 开始 UTC | 结束 UTC | COD epoch70 train loss | COD epoch100 train loss | epoch100 Sm |
| --- | --- | --- | --- | --- | --- |
| A | 2026-10-16T02:11:45.206417+00:00 | 2026-10-16T06:25:07.774770+00:00 | 0.182759 | 0.165956 | 0.803070 |
| B | 2026-10-16T02:11:46.054135+00:00 | 2026-10-16T07:53:13.352410+00:00 | 0.182457 | 0.166266 | 0.803116 |
| C | 2026-10-16T02:11:45.986317+00:00 | 2026-10-16T09:45:18.910393+00:00 | 0.182511 | 0.166024 | 0.805074 |
| D | 2026-10-16T02:11:46.076749+00:00 | 2026-10-16T09:44:55.769896+00:00 | 0.182089 | 0.165567 | 0.803630 |

四组 COD 历史均为100轮，正式验证20轮、选中记录均与其中最大 Sm 一致。后期 loss 继续下降，正式 Sm 未超过70轮；延长训练不是当前优先项。各 source/teacher 历史同样完整100轮，指标与损失为有限值。

各阶段 `history[].seconds` 求和（小时，仅本轮记录的 epoch 耗时）：

| 组别 | Teacher | Source student | COD |
| --- | ---: | ---: | ---: |
| A | — | — | 4.200 |
| B | — | 1.474 | 4.182 |
| C | 1.723 | 1.572 | 4.162 |
| D | 1.724 | 1.572 | 4.154 |

从最早开始到最晚结束为 7 小时 33 分 34 秒（约 7h34m），与四组并行运行的记录时段相符；不等于各进程耗时相加。epoch 求和不含全部进程启动、checkpoint 与阶段切换开销。该记录可用于当前配置的成本参考，不能当作下一版结构或干预实验的 ETA。

## 来源与核验边界

本地每个运行目录只有 comparison.json（内嵌训练 history）。没有同步 run.log、launch.json、exit_status.json、split_manifest.json、分离的 metrics.jsonl/history.json、模型权重或预测图。下述 checkpoint、artifact 与 manifest 哈希均为 JSON 记录值，未独立读取相应文件复核；未独立访问服务器，也未核验样本成员/重叠或逐图复算。status=complete 不能代替进程退出码。没有具体预测样本，不能编造失败图像或类别。

四份 protocol.source_sha256 相同，所列 13 个文件已在登记时逐个与本地源码 SHA256 比对，一致。该列表不覆盖 queue launcher 及所有依赖；服务器运行基点是 074cc55b81c45dd5b4105f789f19c662e63eedfc 且 code_dirty=true，不能把实验归因到一个未经验证的干净提交。

B/C/D 记录的 split_manifest 整文件哈希相同，A 哈希不同；A 的 VDT 样本数为 0，B/C/D 含 VDT 划分，因此整文件内容本就不完全相同。四组 split_seed、COD 样本数相同且源码路径一致，但在缺少 manifest 文件时不能把这些元数据当作 COD 成员已独立核验的证明。

已读取 comparison 的哈希：

| 原始路径（本地，不提交） | comparison SHA256 |
| --- | --- |
| `results/optimization/relation/seed-20261012/A_scratch/comparison.json` | `aecfec0c193f12edf5792f27990c61865a56831eecc320c112f246ccf7653059` |
| `results/optimization/relation/seed-20261012/B_rgb_source/comparison.json` | `98f3a845e3d946db7a7c2d1ec3574f7a2ff1a555bfdf159778f50d32dce8d081` |
| `results/optimization/relation/seed-20261012/C_rgb_relation/comparison.json` | `979e60517fe2649f8633e8b746942570192617209601c4cd8b4a5d9cdedc1860` |
| `results/optimization/relation/seed-20261012/D_multimodal_relation/comparison.json` | `03b162576be7019f5de2641c747e6a1d353bac701032a9bd231512ba62ad73cc` |

四份 comparison 共有的源码指纹（登记时已与当前本地文件逐个比对）：

| 文件 | SHA256 |
| --- | --- |
| `models/triprd_relation.py` | `ee67a807bfce6f17ad0b0a7007712843237d9ae722411c9ec0d830104dae0e5d` |
| `models/triprd.py` | `fb7fc3d091cb6d993104c566b2bd6ff8e85a7d1dcc0dce8c94611ada4123d7eb` |
| `models/triprd_full.py` | `823a10a2b9f040d03eca58b02d758d34eeef360534298f7a823534bceed30065` |
| `tools/run_triprd_relation.py` | `5ac638bfaa2777184c00e7c0704615fd149b0c84e5567cceed97ba55d6b5120b` |
| `tools/triprd_relation_training.py` | `f3601a16fcf744683ec325bb3ddcef97a593f034def87de118d497b04270bc62` |
| `tools/run_triprd_ab.py` | `f0f549dc4b59942f4b501c71375fc4a9befb640cf6dbfb47d87eca409cb3ce56` |
| `tools/triprd_training.py` | `2536dfd13be0f555b6d478511d74471b4d2783800de9099e29835499a6c8b55b` |
| `evaluation/validation.py` | `15ea8cee017dc5f4feb7958e88c6808409d943a86d5c45ffba875d0f59b4168a` |
| `evaluation/cod_metrics.py` | `275d8192dd92292ddb0a67e9a6b0af36c4c29cc093367715d0946891e3da3e5f` |
| `evaluation/camoformer_metrics.py` | `0b99d035113861e4720f238eb2156e8dde74ea021010df79014eeeb0d66c5a5a` |
| `tools/evaluate_cod.py` | `d0b17fe18189e5d3bc16b92fca71765c9944f53a975e0124b5b5748ce7b8d1d7` |
| `datasets/cod_dataset.py` | `acbb1a299bca85174319ab1a6badedb34986b2b1d7723fc0d8b66c8499e3eddd` |
| `datasets/vdt2048.py` | `5852b03e736f3950badffe0e76141ed9a561978d79ea4254a0a1e5aabce535b2` |

A 的 manifest hash 为 `5bb0b54160d265a4564e184ff6a6531a47e5571dfa60153ac2fe5fa6eb48f428`；B/C/D 均为 `a781db0cc97fec81ed87eff0a07925efc287039a684bf68e815a54dbbcb5e3bc`。JSON 记录均 `split_manifest=null`，由相同 split seed 独立生成；未同步清单，实际成员仍待独立对照。

## 下一步决定

优先利用已有 C/D 教师权重，在完全相同的 VDT 验证图像上比较 learned、uniform、shuffled 关系以及 zero relation residual，记录 Dice、预测变化及关系分布。若关系干预几乎无影响，再检验“value 和解码输入保持 RGB，T/D 仅影响关系权重”的教师设计。干预脚本及上述受限教师尚未实现，本次没有相应实验结果。出现 D 对 A/B/C 的净增益后再做配对多 seed、固定预算全量 COD 训练和冻结后的四测试集评估；不依据本轮结果直接延长训练或追加损失。

本次归档保留实现、测试、协议与负结果；没有执行新干预、受限教师训练、多种子复验或最终测试。实验命令见 [关系实验指南](../tools/TRIPRD_RELATION.md)。
