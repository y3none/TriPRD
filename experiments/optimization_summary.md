# 上一版 TriPRD 优化实验汇总

- 登记日期：2026-10-16（Asia/Singapore）。
- 范围：E002 GPU smoke、E003–E006 COD 配方队列、E007–E010 transfer/第二种子队列。
- 所有原始运行的 UTC 起止时间、comparison SHA256、代码 hash 和服务器路径均见各自记录。
- 本文是既有运行分析，不创建新运行 ID，不代表新增模型机制已实现或已验证。

## 结论

新 COD 优化配方在两个训练种子的来源等权宏 Sm 上均优于旧配方，但增益从 +0.009373 缩小到 +0.002693，第二个种子存在 CAMO Sm/MAE 退步。仅两个种子不能证明统计显著或稳定收益。旧 VDT 监督与 logits KD 权重在 full/encoder-only 两种迁移方式下均未超过同 seed 的 RGB 基线；encoder-only 的部分指标改善没有修复主指标 Sm 差距。所有优化运行均 skip-test，不能声称测试集提高或达到 SOTA。

## 共同协议与比较边界

E003–E010 均为 ResNet-50、ImageNet 归一化、352 输入、256 decoder channels、batch8、AdamW、100 COD epoch。A 直接使用 ImageNet 初始化；B/C 从已有 VDT 权重加载全部学生参数或仅 encoder。COD 总训练池 4040 张中 3232 张用于梯度训练、808 张内部验证（COD10K 608、CAMO 200）。各来源先逐图平均，再以 50%/50% 得到宏指标；每 5 轮原尺寸验证一次，按这 20 个候选轮次的宏 Sm 选模。

八个完整优化运行均只执行 COD 监督训练；B/C 的 VDT 阶段因权重复用而跳过。protocol 中保留的 vdt_epochs、KD 权重与 gate_mode 不表示本轮重新蒸馏或在 COD 阶段使用了这些损失。

新配方为 encoder LR×0.1、cosine、warmup5、min LR ratio 0.05；旧配方为 encoder/decoder 统一 1e-4、无 scheduler。三项优化设置一起改变，当前实验只能判断组合配方，不能归因到单一设置。

E003–E008 的训练 seed 为 20261012；E009–E010 为 20261013。split seed 均为 20261012。第二轮统一指定第一轮 A_control 的 split_manifest 路径。代码会重写包含 train_seed/source_manifest 的输出 manifest，因此完整文件 SHA 会随这些元数据改变；当前缺少 manifest，未独立验证跨轮样本成员或训练/验证重叠。

E002 只有 2 轮、每来源最多 32 张，仅用于链路检查；它与完整验证结果不能作性能差值。E001 具备四测试集指标，但实际训练划分等证据仍不完整，也不能与本轮内部验证直接比较。

## 选中 checkpoint 的验证结果

下列四项指标来自同一个 Sm 选中轮；没有分别为各指标挑最好 epoch。E_phi 是 mean E，不是 max/adaptive E。

| 记录 | 配方 / 迁移 | train seed | 最佳轮 | 宏 Sm | 宏 Fw | 宏 mean E | 宏 MAE | COD10K Sm | CAMO Sm |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [E003_cod_A_control](E003_cod_A_control.md) | control/无VDT | 20261012 | 10 | 0.798263 | 0.677055 | 0.848436 | 0.054341 | 0.782397 | 0.814129 |
| [E004_cod_A_schedule](E004_cod_A_schedule.md) | schedule/无VDT | 20261012 | 95 | 0.807636 | 0.704436 | 0.868485 | 0.047933 | 0.797346 | 0.817926 |
| [E005_cod_B_schedule](E005_cod_B_schedule.md) | schedule/full | 20261012 | 40 | 0.781063 | 0.648989 | 0.836460 | 0.055411 | 0.764136 | 0.797991 |
| [E006_cod_C_schedule](E006_cod_C_schedule.md) | schedule/full | 20261012 | 60 | 0.776925 | 0.649248 | 0.837467 | 0.056724 | 0.757121 | 0.796729 |
| [E007_transfer_B_encoder](E007_transfer_B_encoder.md) | schedule/encoder | 20261012 | 95 | 0.780768 | 0.656656 | 0.836831 | 0.053569 | 0.763424 | 0.798111 |
| [E008_transfer_C_encoder](E008_transfer_C_encoder.md) | schedule/encoder | 20261012 | 85 | 0.776117 | 0.652425 | 0.839120 | 0.055049 | 0.756106 | 0.796128 |
| [E009_transfer_A_control_repeat](E009_transfer_A_control_repeat.md) | control/无VDT | 20261013 | 10 | 0.807635 | 0.687991 | 0.861264 | 0.051713 | 0.788632 | 0.826638 |
| [E010_transfer_A_schedule_repeat](E010_transfer_A_schedule_repeat.md) | schedule/无VDT | 20261013 | 35 | 0.810328 | 0.694426 | 0.873200 | 0.050660 | 0.796969 | 0.823688 |

## 配方的第二个种子没有消除不确定性

| train seed | control Sm | schedule Sm | 配对 ΔSm |
|---|---:|---:|---:|
| 20261012 | 0.798263 | 0.807636 | +0.009373 |
| 20261013 | 0.807635 | 0.810328 | +0.002693 |

两种子 control 的 Sm 均值为 0.802949、样本标准差 0.006627；schedule 均值 0.808982、样本标准差 0.001904，配对增益均值 +0.006033。这些只是 n=2 描述统计，不是显著性或稳定性结论。

第二个 seed 的 CAMO Sm 从 0.826638 降到 0.823688（−0.002951），MAE 从 0.061013 升到 0.063005（+0.001992）；COD10K 的改善拉高了宏指标。第一次 schedule 的最佳轮为 95，第二次为 35，训练最佳时点也有波动。

## encoder-only 没有修复旧 VDT 权重的 Sm 差距

B 的源权重 SHA 与 E005 相同，C 与 E006 相同；来源最佳轮分别 61/93，迁移范围从 full 改为 encoder。以下同 seed 比较保留各运行自身 Sm 选中轮。

| 比较（后者−前者） | ΔSm | ΔFw | Δmean E | ΔMAE |
|---|---:|---:|---:|---:|
| E005 → E007 | -0.000296 | +0.007668 | +0.000371 | -0.001842 |
| E006 → E008 | -0.000808 | +0.003177 | +0.001653 | -0.001675 |

encoder-only 的 Fw/mean E/MAE 有改善，但 B/C 的 Sm 都略降。B_encoder/C_encoder 相对同 seed A_schedule 的 Sm 差距仍为 −0.026868/−0.031518。本次种子、权重与配方不支持“只重置 decoder/head 就足以修复”的假设；不能推广为 VDT、多模态或关系蒸馏本身永远无效。两轮并未训练新的 RGB teacher、显式跨模态关系、可靠性目标或 COD prior，因而不能检验这些设计稿机制。

## 证据清单与限制

- 本地只有上述九个 comparison.json：八个完整优化运行共 800 条 COD history、160 次正式验证；smoke 另有 16 条阶段 history。状态均记录 complete、test_status=skipped，test 为空。
- 所有最佳轮都核对为已评估轮次的 Sm argmax；每次来源计数、宏平均及选中轮指标一致，数值有限。
- 九份记录保存原始 comparison 的 SHA256。运行记录中的八个源码 SHA 均与登记时工作区一致；Git 基点为 074cc55b81c45dd5b4105f789f19c662e63eedfc，dirty=true。
- B/C 来源 checkpoint 的 SHA 在 full 与 encoder-only 两轮相同；没有本地权重，未重新计算权重文件 hash。
- 缺少本地 run.log、launch/exit_status、split manifest、独立 metrics/history、模型权重和逐图预测；内嵌 history 可核对，但无法独立证明进程退出码、成员列表、保存权重或逐图误差。
- 来源/运行 hash 与服务器路径见各运行记录；原始 JSON、模型和预测均不随本次 Git 提交。
- skip-test 时没有测试预测属于预期。所有本轮指标均为内部验证，不用于填写论文四测试集 SOTA 表；现有记录也没有显著性检验。

## 决定

将新配方暂作为 RGB 基线并保留旧配方、full/encoder-only 的负结果。冻结数据、选模、评估口径后，再开展独立的 TriPRD 机制研究与因果对照；当前文档更新不等于新机制实现。最终若使用完整 4040 张训练，必须先确定固定训练预算/最终权重规则，不能把参与训练的原验证集用于选模或按测试集挑 checkpoint。

后续应补归档 manifest、权重及日志，并在配置冻结后单独登记 eval-only 测试。新的正结果须同时考虑逐来源表现和多个训练种子，不能只报告宏平均、某个 seed 或每个测试集各自最佳的模型。
