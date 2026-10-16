# E003_cod_A_control

- Recorded at: 2026-10-16（Asia/Singapore，登记日期）
- Run directory: `results\optimization\cod\seed-20261012\A_control`
- Tags: validation-only,optimization,ablation

- Run started (UTC): `2026-10-15T03:16:53.662606+00:00`
- Run finished (UTC): `2026-10-15T07:41:43.146085+00:00`
- Status: comparison 记录 complete；本地未同步 launcher exit_status，未独立核验进程退出码。

## Change

A 对照：encoder/decoder 统一 lr=1e-4、无 scheduler；改用原尺寸宏 Sm 选模。

## Hypothesis

建立与新配方共享种子、划分和 Sm 选模规则的旧学习率对照。

## Protocol

本节 JSON 来自该运行 comparison 的 protocol，属于已记录参数；不是当前默认值推断。每个来源先逐图平均，再以 50%/50% 合成宏指标。Sm/Fw/mean E 越高越好，MAE 越低越好。E_phi 为 256 阈值曲线均值，不是 max/adaptive E；预测按 uint8→PIL 双线性恢复 GT 原尺寸，指标内部使用逐图 min-max 归一化。

max_samples=0 表示读取全部 COD 训练池后再划分：3232 张参与训练，808 张验证（COD10K 608、CAMO 200）。因此它不是在完整 4040 张上训练的最终模型。正式指标仅在 5、10…100 轮计算，best 指其中 20 个候选轮次的 Sm 最大值，不是未评估的其余 80 轮 Sm 最大值。

```json
{
  "vdt_root": "/data2/yyy/TriPRD/datasets/VDT-2048",
  "cod_root": "/data2/yyy/TriPRD/datasets/COD_benchmark",
  "out": "/data2/yyy/TriPRD/results/optimization/cod/seed-20261012/A_control",
  "groups": "A",
  "model_variant": "resnet50",
  "pretrained": true,
  "epochs": 100,
  "vdt_epochs": 100,
  "batch_size": 8,
  "image_size": 352,
  "base_channels": 256,
  "max_samples": 0,
  "max_test_samples": 0,
  "test_datasets": "COD10K,CAMO,CHAMELEON,NC4K",
  "val_fraction": 0.2,
  "seed": 20261012,
  "train_seed": 20261012,
  "split_seed": 20261012,
  "split_manifest": null,
  "learning_rate": 0.0001,
  "weight_decay": 0.0001,
  "device": "cuda",
  "num_workers": 0,
  "validation_metrics": true,
  "val_metrics_every": 5,
  "selection_metric": "S_alpha",
  "cod_encoder_lr_scale": 1.0,
  "cod_scheduler": "none",
  "cod_warmup_epochs": 0,
  "cod_min_lr_ratio": 0.05,
  "teacher_checkpoint": null,
  "vdt_checkpoint": [],
  "transfer_scope": "full",
  "lambda_logit": 1.0,
  "lambda_relation": 1.0,
  "lambda_boundary_kd": 0.5,
  "boundary_kd_type": "logit_mse",
  "relation_sampling": "legacy",
  "gate_mode": "disagreement",
  "skip_test": true,
  "eval_only": false,
  "cod_checkpoint": [],
  "rgb_normalization": "ImageNet",
  "train_augmentation": "horizontal_flip_p=0.5",
  "optimizer": "AdamW",
  "validation_source_weights": "equal",
  "validation_prediction": "uint8_then_PIL_bilinear_to_original_GT",
  "split_manifest_sha256": "629d64c03968f6870da44b0cf4edf3888b9cc96ba47edf6a0248085045e7101a",
  "sample_counts": {
    "vdt_train": 0,
    "vdt_val": 0,
    "cod_train": 3232,
    "cod_val": 808
  },
  "code_commit": "074cc55b81c45dd5b4105f789f19c662e63eedfc",
  "code_dirty": true,
  "source_sha256": {
    "tools/run_triprd_ab.py": "f0f549dc4b59942f4b501c71375fc4a9befb640cf6dbfb47d87eca409cb3ce56",
    "tools/triprd_training.py": "2536dfd13be0f555b6d478511d74471b4d2783800de9099e29835499a6c8b55b",
    "models/triprd.py": "fb7fc3d091cb6d993104c566b2bd6ff8e85a7d1dcc0dce8c94611ada4123d7eb",
    "models/triprd_full.py": "823a10a2b9f040d03eca58b02d758d34eeef360534298f7a823534bceed30065",
    "evaluation/validation.py": "15ea8cee017dc5f4feb7958e88c6808409d943a86d5c45ffba875d0f59b4168a",
    "evaluation/cod_metrics.py": "275d8192dd92292ddb0a67e9a6b0af36c4c29cc093367715d0946891e3da3e5f",
    "evaluation/camoformer_metrics.py": "0b99d035113861e4720f238eb2156e8dde74ea021010df79014eeeb0d66c5a5a",
    "tools/evaluate_cod.py": "d0b17fe18189e5d3bc16b92fca71765c9944f53a975e0124b5b5748ce7b8d1d7"
  },
  "python_version": "3.11.17",
  "torch_version": "2.7.1+cu128",
  "cuda_version": "12.8",
  "gpu_name": "NVIDIA GeForce RTX 4090"
}
```

本轮仅执行 COD 监督训练；没有新训练 teacher/VDT 阶段。protocol 保留的 vdt_epochs、KD 权重和 gate_mode 不表示本轮启用了相应蒸馏，B/C 的 VDT 预训练通过已有 checkpoint 复用。

## Validation selection

Official validation metrics below belong to the selected epoch; macro averages weight each source equally.

| Group | Best epoch | Selection metric | Selection score | S_alpha | F_beta_w | E_phi mean | MAE | Status / error |
|---|---:|---|---:|---:|---:|---:|---:|---|
| A | 10 | S_alpha | 0.798263 | 0.798263 | 0.677055 | 0.848436 | 0.054341 | ok |

### Selected-epoch validation by source

| Group | Source | Epoch | Samples | S_alpha | F_beta_w | E_phi mean | MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| A | COD10K | 10 | 608 | 0.782397 | 0.626379 | 0.843291 | 0.041554 |
| A | CAMO | 10 | 200 | 0.814129 | 0.727731 | 0.853582 | 0.067129 |

## Results

Test status: `skipped`；`test={}`。本次有正式验证指标，没有四测试集结果；跳过测试时未生成预测图属于预期。本记录不把验证指标填入测试表，也不与 E001 测试分数直接比较。

## Analysis

100 轮完整结束，最佳宏 Sm 出现在第 10 轮；第 100 轮 Sm 降至 0.779367。旧配方早期最佳、后期下降支持继续检查优化与泛化，但不能单凭曲线确定原因。

### Comparison with previous run

本记录建立链路/控制基线，没有同协议的前序成对运行；E001 测试分数与本记录验证分数不作数值差值。

### Failure cases and observations

本记录是新的验证基线，并非 E001 测试结果的重新评估；E001 的实际样本划分和部分超参数未完整归档。

未同步逐图预测或原图，无法识别具体失败样本、类别或边界形态；不从汇总分数虚构案例。

## Decision

保留为配方对照；与 E004 同 seed 成对比较，并由 E009 检查第二个训练种子。

## Artifacts and provenance

- 本地已读取的原始文件：`results/optimization/cod/seed-20261012/A_control/comparison.json`（不随 Git 提交）。
- comparison SHA256：`d913662cede2b41da0dfa91c85bc09479d0cb98f582e099f50652b457087513d`。
- JSON 记录的输出目录：`/data2/yyy/TriPRD/results/optimization/cod/seed-20261012/A_control`；不是已独立访问的服务器目录。
- 输出 split_manifest 整文件 SHA256：`629d64c03968f6870da44b0cf4edf3888b9cc96ba47edf6a0248085045e7101a`。
- 记录的 Git 基点：`074cc55b81c45dd5b4105f789f19c662e63eedfc`，`code_dirty=true`。
- protocol.source_sha256 的 8 个文件已逐个与 2026-10-16 当前本地源码计算比对，一致；该列表不覆盖 launcher、数据读取器及所有依赖，不能将 dirty 运行等同于只含基点提交。
- 本地本运行目录只有 comparison.json。run.log、launch.json、exit_status.json、split_manifest.json、分离的 metrics.jsonl/history.json 和模型权重未同步；history 内嵌于 comparison 已核验。
- 未独立核验服务器文件、样本成员/重叠、权重内容或保存的最佳 checkpoint；验证表反映 JSON 内部一致性，不能替代图像级复算。

### A checkpoint provenance

最终 COD checkpoint 路径记录：`/data2/yyy/TriPRD/results/optimization/cod/seed-20261012/A_control/A/cod/best.pt`；参数量 28,034,114。

无 VDT 权重复用；初始化依据 protocol 中的 ImageNet pretrained 设置。

相关汇总：[optimization_summary.md](optimization_summary.md)。
