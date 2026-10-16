# Experiment records

每次训练、评估或调参都使用一个唯一实验 ID。运行脚本保存原始证据；实验记录脚本把变更、结果和研究判断整理到台账中。

## 推荐流程

1. 为本次实验确定 ID，例如 `E001_baseline`、`E002_lr5e-5`。
2. 使用独立输出目录运行实验，例如 `results/E002_lr5e-5`，不要覆盖旧结果。
3. 运行：

```bash
python tools/record_experiment.py \
  --id E002_lr5e-5 \
  --run-dir results/E002_lr5e-5 \
  --change "learning rate: 1e-4 -> 5e-5" \
  --hypothesis "降低学习率可以减少验证集波动" \
  --tags optimizer,ablation
```

4. 编辑生成的 `experiments/E002_lr5e-5.md`，填写与上一轮的差值、失败案例和最终决定。

## 保留的证据

每个 `run-dir` 应至少保留 `run.log`、`comparison.json`、`split_manifest.json`、训练阶段的 `metrics.jsonl`、`history.json`、`best.pt` 和预测图。正式指标以 `comparison.json` 的 `test.<group>.<dataset>.official`（由 `evaluation/` 计算）为准，MAE 使用 `official.MAE`。外层 `dice`、`mae` 是快速预测统计，须与正式指标分列；训练 soft Dice 只用于收敛监控。

筛选阶段使用 `--skip-test` 时不生成预测图，也没有 test 指标，这是正常状态。记录器会从 `groups.<group>.cod_finetune` 提取选模指标、best epoch，以及该 epoch 的 `history[].val_official`，分别报告来源等权验证指标和各来源指标；不会用其他 epoch 的结果补齐空值。旧记录缺少这些字段时显示 `unavailable`。冻结后的 `--eval-only` 单独登记，并关联来源训练记录；不能把验证分数写入正式 test 表。旧版优化流程见 [优化实验说明](../tools/TRIPRD_OPTIMIZATION.md)。

关系版的 `--cod-train-mode full` 使用全部 COD 训练池，并在预先固定的末轮选取 checkpoint；记录中的 `selection_metric=fixed_last_epoch` 表示没有验证选模，不应填写或推断 best Sm。记录器对此显示 `not applicable`，保留 `selected_epoch`；正式四测试集分数仍以独立评估结果为准。本轮 E011–E014 实际采用 `split` 模式，尚未执行这项全量协议。

原始文件可保存在本地或服务器，不要求全部提交 Git。未随仓库提交的证据须在记录中注明存放位置、核验范围和 SHA256，避免将仅本地存在的文件写成仓库链接。

`experiments/index.md` 是所有实验的索引，单个 Markdown 记录保存变更、假设、协议、自动汇总结果、分析和下一步决定。

## 版本与分组

[旧版汇总](optimization_summary.md) 对应 E001–E010 的 mean-fusion/logit KD/特征关系实验及配方优化；[显式关系首轮汇总](relation_summary.md) 对应 E011–E014 的 `triprd_relation_v1`。两版的 A/B/C/D 含义不同，比较时须同时标明方法版本与实验 ID。关系版 A 为同结构无 VDT、B 为 VDT RGB 监督、C 为容量匹配的 RGB 教师关系 KD、D 为真实多模态教师关系 KD；四组只比较本轮匹配条件下的效应，不能沿用旧版组名解释。

关系版运行与证据保存要求见 [关系实验指南](../tools/TRIPRD_RELATION.md)。每次变更机制、随机种子、源权重或训练协议都使用新的 ID 和输出目录；保留本轮没有主指标净增益的结论，后续干预诊断不能覆盖现有记录。
