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

原始文件可保存在本地或服务器，不要求全部提交 Git。未随仓库提交的证据须在记录中注明存放位置、核验范围和 SHA256，避免将仅本地存在的文件写成仓库链接。

`experiments/index.md` 是所有实验的索引，单个 Markdown 记录保存变更、假设、协议、自动汇总结果、分析和下一步决定。
