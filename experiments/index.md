# Experiment index

实验登记规范见 [README](README.md)。日期为登记日期；每条记录单独说明运行时间及证据完整性。

| ID | Date | Change | Record | Decision |
|---|---|---|---|---|
| E001_triprd_full | 2026-10-15 | A/B/C/D 全量实验及四测试集补评估；comparison、16 条指标与 800 条 epoch 历史已核验 | [E001_triprd_full.md](E001_triprd_full.md) | 保留为基线；D 尚无稳定增益，先验证 COD 微调与迁移，完整日志、划分、权重和预测待补 |
| E002_gpu_smoke | 2026-10-16 | ResNet-50 A/B/C/D GPU 冒烟：各阶段 2 epoch、每来源最多 32 张、原尺寸正式 Sm 选模。 | [E002_gpu_smoke.md](E002_gpu_smoke.md) | 保留为链路验证；后续性能判断使用完整验证实验，不把 smoke 排序作为方法有效性证据。 |
| E003_cod_A_control | 2026-10-16 | A 对照：encoder/decoder 统一 lr=1e-4、无 scheduler；改用原尺寸宏 Sm 选模。 | [E003_cod_A_control.md](E003_cod_A_control.md) | 保留为配方对照；与 E004 同 seed 成对比较，并由 E009 检查第二个训练种子。 |
| E004_cod_A_schedule | 2026-10-16 | A 新配方：encoder LR×0.1、cosine、5 轮 warmup、最小 LR 比例 0.05。 | [E004_cod_A_schedule.md](E004_cod_A_schedule.md) | 保留为下一步 RGB 基线配方；结合 E010 第二个种子和逐来源结果判断，不依据测试集调整配置。 |
| E005_cod_B_schedule | 2026-10-16 | 复用 E001 B/vdt 权重并 full transfer，COD 使用与 E004 相同的新配方。 | [E005_cod_B_schedule.md](E005_cod_B_schedule.md) | 保留为同数据 RGB 预训练对照；E007 检查 encoder-only transfer 是否修复。 |
| E006_cod_C_schedule | 2026-10-16 | 复用 E001 C/vdt logits KD 权重并 full transfer，COD 使用统一新配方。 | [E006_cod_C_schedule.md](E006_cod_C_schedule.md) | 保留为 logits KD 对照，当前不支持其稳定优于 B 或 A；用 E008 排查 full transfer 影响。 |
| E007_transfer_B_encoder | 2026-10-16 | B 保持 seed=20261012 和新配方，仅将旧 VDT 权重复用范围从 full 改为 encoder。 | [E007_transfer_B_encoder.md](E007_transfer_B_encoder.md) | encoder-only 未修复 B 的宏 Sm 差距；保留负结果，不将其宣传为迁移解决方案。 |
| E008_transfer_C_encoder | 2026-10-16 | C 保持 seed=20261012 和新配方，仅将旧 VDT 权重复用范围从 full 改为 encoder。 | [E008_transfer_C_encoder.md](E008_transfer_C_encoder.md) | 保留负结果；后续方法研究必须验证知识本身的价值，不能只依赖更换权重加载范围。 |
| E009_transfer_A_control_repeat | 2026-10-16 | A 旧配方复跑：train seed 改为 20261013，固定 split seed=20261012 并指定上一轮 manifest。 | [E009_transfer_A_control_repeat.md](E009_transfer_A_control_repeat.md) | 保留为第二个种子的旧配方对照；有效配方对照为 E010−E009。 |
| E010_transfer_A_schedule_repeat | 2026-10-16 | A 新配方复跑：train seed=20261013，与 E009 使用同一划分和随机种子。 | [E010_transfer_A_schedule_repeat.md](E010_transfer_A_schedule_repeat.md) | 暂保留新配方为 RGB 基线；明确两种子证据有限及 CAMO 退步，冻结配置后再做正式测试。 |

GPU smoke 与两轮优化的联合分析见 [optimization_summary.md](optimization_summary.md)；该文件为汇总，不是新的训练运行 ID。
