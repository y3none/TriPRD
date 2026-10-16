# E008_transfer_C_encoder

- Recorded at: 2026-10-16（Asia/Singapore，登记日期）
- Run directory: `results\optimization\transfer\seed-20261012-repeat-20261013\C_encoder`
- Tags: validation-only,optimization,ablation

- Run started (UTC): `2026-10-15T09:15:53.140225+00:00`
- Run finished (UTC): `2026-10-15T13:40:00.833018+00:00`
- Status: comparison 记录 complete；本地未同步 launcher exit_status，未独立核验进程退出码。

## Change

C 保持 seed=20261012 和新配方，仅将旧 VDT 权重复用范围从 full 改为 encoder。

## Hypothesis

重置 decoder/head 能恢复 logits KD 编码器中的可迁移信息。

## Protocol

本节 JSON 来自该运行 comparison 的 protocol，属于已记录参数；不是当前默认值推断。每个来源先逐图平均，再以 50%/50% 合成宏指标。Sm/Fw/mean E 越高越好，MAE 越低越好。E_phi 为 256 阈值曲线均值，不是 max/adaptive E；预测按 uint8→PIL 双线性恢复 GT 原尺寸，指标内部使用逐图 min-max 归一化。

max_samples=0 表示读取全部 COD 训练池后再划分：3232 张参与训练，808 张验证（COD10K 608、CAMO 200）。因此它不是在完整 4040 张上训练的最终模型。正式指标仅在 5、10…100 轮计算，best 指其中 20 个候选轮次的 Sm 最大值，不是未评估的其余 80 轮 Sm 最大值。

```json
{
  "vdt_root": "/data2/yyy/TriPRD/datasets/VDT-2048",
  "cod_root": "/data2/yyy/TriPRD/datasets/COD_benchmark",
  "out": "/data2/yyy/TriPRD/results/optimization/transfer/seed-20261012-repeat-20261013/C_encoder",
  "groups": "C",
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
  "split_manifest": "/data2/yyy/TriPRD/results/optimization/cod/seed-20261012/A_control/split_manifest.json",
  "learning_rate": 0.0001,
  "weight_decay": 0.0001,
  "device": "cuda",
  "num_workers": 0,
  "validation_metrics": true,
  "val_metrics_every": 5,
  "selection_metric": "S_alpha",
  "cod_encoder_lr_scale": 0.1,
  "cod_scheduler": "cosine",
  "cod_warmup_epochs": 5,
  "cod_min_lr_ratio": 0.05,
  "teacher_checkpoint": null,
  "vdt_checkpoint": [
    "C=/data2/yyy/TriPRD/results/triprd_full/C/vdt/best.pt"
  ],
  "transfer_scope": "encoder",
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
  "split_manifest_sha256": "f505ccb43f42ebeaa49c85d7ccfd46357e005100a9d7ccbb2158bf704db7249f",
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
| C | 85 | S_alpha | 0.776117 | 0.776117 | 0.652425 | 0.839120 | 0.055049 | ok |

### Selected-epoch validation by source

| Group | Source | Epoch | Samples | S_alpha | F_beta_w | E_phi mean | MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| C | COD10K | 85 | 608 | 0.756106 | 0.592158 | 0.824091 | 0.043570 |
| C | CAMO | 85 | 200 | 0.796128 | 0.712691 | 0.854149 | 0.066528 |

## Results

Test status: `skipped`；`test={}`。本次有正式验证指标，没有四测试集结果；跳过测试时未生成预测图属于预期。本记录不把验证指标填入测试表，也不与 E001 测试分数直接比较。

## Analysis

相对 E006，宏 Sm -0.000808，Fw +0.003177、mean E +0.001653、MAE -0.001675。相对 E004，Sm 仍低 0.031518；因此 encoder-only 未修复主指标差距。

### Comparison with previous run

相对 [E006_cod_C_schedule](E006_cod_C_schedule.md)，下表为本记录选中轮指标减去对照选中轮指标：

| 来源 | ΔSm ↑ | ΔFw ↑ | Δmean E ↑ | ΔMAE ↓ |
|---|---:|---:|---:|---:|
| macro | -0.000808 | +0.003177 | +0.001653 | -0.001675 |
| COD10K | -0.001015 | +0.003949 | +0.000265 | -0.000210 |
| CAMO | -0.000600 | +0.002405 | +0.003041 | -0.003139 |

相对 [E004_cod_A_schedule](E004_cod_A_schedule.md)，下表为本记录选中轮指标减去对照选中轮指标：

| 来源 | ΔSm ↑ | ΔFw ↑ | Δmean E ↑ | ΔMAE ↓ |
|---|---:|---:|---:|---:|
| macro | -0.031518 | -0.052011 | -0.029365 | +0.007117 |
| COD10K | -0.041239 | -0.072438 | -0.040429 | +0.008334 |
| CAMO | -0.021797 | -0.031584 | -0.018301 | +0.005899 |


### Failure cases and observations

来源 C 权重 SHA、epoch 93 和预处理 metadata 与 E006 一致，仅迁移范围改变。第一波未指定输入 manifest（input=null）；第二波显式读取第一波 A_control 输出的 split_manifest.json，当前没有清单文件可独立核验成员。此结果针对旧教师和旧 logits KD 权重，不检验设计稿尚未实现的新关系机制。

未同步逐图预测或原图，无法识别具体失败样本、类别或边界形态；不从汇总分数虚构案例。

## Decision

保留负结果；后续方法研究必须验证知识本身的价值，不能只依赖更换权重加载范围。

## Artifacts and provenance

- 本地已读取的原始文件：`results/optimization/transfer/seed-20261012-repeat-20261013/C_encoder/comparison.json`（不随 Git 提交）。
- comparison SHA256：`5469ac91c46abc73b32132ab56405beb4551b2744314410eb35e41d3375c3dd0`。
- JSON 记录的输出目录：`/data2/yyy/TriPRD/results/optimization/transfer/seed-20261012-repeat-20261013/C_encoder`；不是已独立访问的服务器目录。
- 输出 split_manifest 整文件 SHA256：`f505ccb43f42ebeaa49c85d7ccfd46357e005100a9d7ccbb2158bf704db7249f`。
- 记录的 Git 基点：`074cc55b81c45dd5b4105f789f19c662e63eedfc`，`code_dirty=true`。
- protocol.source_sha256 的 8 个文件已逐个与 2026-10-16 当前本地源码计算比对，一致；该列表不覆盖 launcher、数据读取器及所有依赖，不能将 dirty 运行等同于只含基点提交。
- 本地本运行目录只有 comparison.json。run.log、launch.json、exit_status.json、split_manifest.json、分离的 metrics.jsonl/history.json 和模型权重未同步；history 内嵌于 comparison 已核验。
- 未独立核验服务器文件、样本成员/重叠、权重内容或保存的最佳 checkpoint；验证表反映 JSON 内部一致性，不能替代图像级复算。

输入 manifest 的记录路径：`/data2/yyy/TriPRD/results/optimization/cod/seed-20261012/A_control/split_manifest.json`。
输出 manifest 还包含 train_seed、source_manifest 等元数据；它的整文件 hash 不等于成员集合 hash。本轮 B/C 与 A repeat 的 hash 不同符合这些字段差异，不能据此认定划分变化，亦不能在缺少文件时声称成员已独立核验。

### C checkpoint provenance

最终 COD checkpoint 路径记录：`/data2/yyy/TriPRD/results/optimization/transfer/seed-20261012-repeat-20261013/C_encoder/C/cod/best.pt`；参数量 28,034,114。

VDT 来源 metadata（来源权重 SHA 是运行端记录值，本地无权重可重新计算）：

```json
{
  "checkpoint": "/data2/yyy/TriPRD/results/triprd_full/C/vdt/best.pt",
  "sha256": "476b3ab6ca9244850485278fd34acaabedc522c0d18c10a631428d07136395a2",
  "source_stage": "C/vdt",
  "source_seed": 20261012,
  "source_epoch": 93,
  "transfer_scope": "encoder",
  "source_protocol": {
    "model_variant": "resnet50",
    "pretrained": true,
    "rgb_normalization": "ImageNet",
    "train_augmentation": "horizontal_flip_p=0.5",
    "base_channels": 256,
    "image_size": 352,
    "seed": 20261012,
    "epochs": 100,
    "vdt_epochs": 100
  },
  "reused": true,
  "source_protocol_origin": "/data2/yyy/TriPRD/results/triprd_full/comparison.json",
  "preprocessing_verified": true
}
```

相关汇总：[optimization_summary.md](optimization_summary.md)。
