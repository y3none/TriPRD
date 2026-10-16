# E002_gpu_smoke

- Recorded at: 2026-10-16（Asia/Singapore，登记日期）
- Run directory: `results\optimization\gpu_smoke`
- Tags: validation-only,optimization,gpu-smoke

- Run started (UTC): `2026-10-15T02:23:09.100743+00:00`
- Run finished (UTC): `2026-10-15T02:24:40.311577+00:00`
- Status: comparison 记录 complete；本地未同步 launcher exit_status，未独立核验进程退出码。

## Change

ResNet-50 A/B/C/D GPU 冒烟：各阶段 2 epoch、每来源最多 32 张、原尺寸正式 Sm 选模。

## Hypothesis

验证训练、正式验证、选模和结果汇总链路可执行；不检验最终性能。

## Protocol

本节 JSON 来自该运行 comparison 的 protocol，属于已记录参数；不是当前默认值推断。每个来源先逐图平均，再以 50%/50% 合成宏指标。Sm/Fw/mean E 越高越好，MAE 越低越好。E_phi 为 256 阈值曲线均值，不是 max/adaptive E；预测按 uint8→PIL 双线性恢复 GT 原尺寸，指标内部使用逐图 min-max 归一化。

Smoke 使用 VDT train/val=26/6、COD train/val=52/12；COD 验证两个来源各 6 张。两轮正式 Sm 均参与选模。

```json
{
  "vdt_root": "/data2/yyy/TriPRD/datasets/VDT-2048",
  "cod_root": "/data2/yyy/TriPRD/datasets/COD_benchmark",
  "out": "results/optimization/gpu_smoke~",
  "groups": "A,B,C,D",
  "model_variant": "resnet50",
  "pretrained": true,
  "epochs": 2,
  "vdt_epochs": 2,
  "batch_size": 8,
  "image_size": 352,
  "base_channels": 256,
  "max_samples": 32,
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
  "val_metrics_every": 1,
  "selection_metric": "S_alpha",
  "cod_encoder_lr_scale": 0.1,
  "cod_scheduler": "cosine",
  "cod_warmup_epochs": 1,
  "cod_min_lr_ratio": 0.05,
  "teacher_checkpoint": null,
  "vdt_checkpoint": [],
  "transfer_scope": "full",
  "lambda_logit": 1.0,
  "lambda_relation": 1.0,
  "lambda_boundary_kd": 0.1,
  "boundary_kd_type": "prob_mse",
  "relation_sampling": "uniform",
  "gate_mode": "none",
  "skip_test": true,
  "eval_only": false,
  "cod_checkpoint": [],
  "rgb_normalization": "ImageNet",
  "train_augmentation": "horizontal_flip_p=0.5",
  "optimizer": "AdamW",
  "validation_source_weights": "equal",
  "validation_prediction": "uint8_then_PIL_bilinear_to_original_GT",
  "split_manifest_sha256": "6f0cddd80d03302ac555bb0fd10ce7b551cd11af82795e13b40a028dc5df8888",
  "sample_counts": {
    "vdt_train": 26,
    "vdt_val": 6,
    "cod_train": 52,
    "cod_val": 12
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

## Validation selection

Official validation metrics below belong to the selected epoch; macro averages weight each source equally.

| Group | Best epoch | Selection metric | Selection score | S_alpha | F_beta_w | E_phi mean | MAE | Status / error |
|---|---:|---|---:|---:|---:|---:|---:|---|
| A | 2 | S_alpha | 0.546774 | 0.546774 | 0.243589 | 0.421444 | 0.368921 | ok |
| B | 2 | S_alpha | 0.600643 | 0.600643 | 0.275384 | 0.496056 | 0.304048 | ok |
| C | 2 | S_alpha | 0.613843 | 0.613843 | 0.288284 | 0.507892 | 0.299489 | ok |
| D | 2 | S_alpha | 0.617172 | 0.617172 | 0.289243 | 0.513795 | 0.289696 | ok |

### Selected-epoch validation by source

| Group | Source | Epoch | Samples | S_alpha | F_beta_w | E_phi mean | MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| A | COD10K | 2 | 6 | 0.523813 | 0.222814 | 0.396840 | 0.390305 |
| A | CAMO | 2 | 6 | 0.569735 | 0.264365 | 0.446048 | 0.347537 |
| B | COD10K | 2 | 6 | 0.581269 | 0.249536 | 0.453286 | 0.320442 |
| B | CAMO | 2 | 6 | 0.620017 | 0.301231 | 0.538826 | 0.287654 |
| C | COD10K | 2 | 6 | 0.600965 | 0.267894 | 0.479988 | 0.311520 |
| C | CAMO | 2 | 6 | 0.626722 | 0.308673 | 0.535796 | 0.287458 |
| D | COD10K | 2 | 6 | 0.607410 | 0.269930 | 0.485702 | 0.299664 |
| D | CAMO | 2 | 6 | 0.626934 | 0.308556 | 0.541889 | 0.279729 |

## Results

Test status: `skipped`；`test={}`。本次有正式验证指标，没有四测试集结果；跳过测试时未生成预测图属于预期。本记录不把验证指标填入测试表，也不与 E001 测试分数直接比较。

## Analysis

8 个阶段各有连续 2 轮 history，共 16 条；四个 COD 阶段均每轮验证、均选中第 2 轮。教师使用 ImageNet 初始化并在本次重新执行 2 轮 VDT 训练，没有复用 teacher checkpoint；其验证 Dice 仅 0.119229。D 使用 uniform relation、prob_mse 边界 KD（权重 0.1）和 gate_mode=none；门控值约 1 属预期。

### Comparison with previous run

本记录建立链路/控制基线，没有同协议的前序成对运行；E001 测试分数与本记录验证分数不作数值差值。

### Failure cases and observations

本次仅支持 GPU smoke 链路完成。单 runner 的 A/B/C/D 顺序训练不能证明四卡 launcher 队列运行成功；没有复用已有 teacher/VDT 权重，也没有验证 disagreement gate。少样本、短训练和弱教师下的 D>C>B>A 排序不能推广。

未同步逐图预测或原图，无法识别具体失败样本、类别或边界形态；不从汇总分数虚构案例。

## Decision

保留为链路验证；后续性能判断使用完整验证实验，不把 smoke 排序作为方法有效性证据。

## Artifacts and provenance

- 本地已读取的原始文件：`results/optimization/gpu_smoke/comparison.json`（不随 Git 提交）。
- comparison SHA256：`8cabc5f2a0cc20d9b50de3f317ad7ed9553ce0b8f95f1e78dc608eb40a95df7e`。
- JSON 记录的输出目录：`results/optimization/gpu_smoke~`；不是已独立访问的服务器目录。
- 输出 split_manifest 整文件 SHA256：`6f0cddd80d03302ac555bb0fd10ce7b551cd11af82795e13b40a028dc5df8888`。
- 记录的 Git 基点：`074cc55b81c45dd5b4105f789f19c662e63eedfc`，`code_dirty=true`。
- protocol.source_sha256 的 8 个文件已逐个与 2026-10-16 当前本地源码计算比对，一致；该列表不覆盖 launcher、数据读取器及所有依赖，不能将 dirty 运行等同于只含基点提交。
- 本地本运行目录只有 comparison.json。run.log、launch.json、exit_status.json、split_manifest.json、分离的 metrics.jsonl/history.json 和模型权重未同步；history 内嵌于 comparison 已核验。
- 未独立核验服务器文件、样本成员/重叠、权重内容或保存的最佳 checkpoint；验证表反映 JSON 内部一致性，不能替代图像级复算。

注意输出路径尾随 `~`：本地文件夹名 gpu_smoke 不带 `~`，但服务器路径记录为 gpu_smoke~。后续定位文件须核对真实路径。

| 阶段 | 连续 history 轮数 | 选模指标 | 最佳轮 | 最佳值 |
|---|---:|---|---:|---:|
| teacher | 2 | dice | 2 | 0.119229 |
| A/cod | 2 | S_alpha | 2 | 0.546774 |
| B/vdt | 2 | dice | 2 | 0.156496 |
| B/cod | 2 | S_alpha | 2 | 0.600643 |
| C/vdt | 2 | dice | 2 | 0.109575 |
| C/cod | 2 | S_alpha | 2 | 0.613843 |
| D/vdt | 2 | dice | 2 | 0.125950 |
| D/cod | 2 | S_alpha | 2 | 0.617172 |

教师 checkpoint 路径记录：`/data2/yyy/TriPRD/results/optimization/gpu_smoke~/teacher/best.pt`。

### A checkpoint provenance

最终 COD checkpoint 路径记录：`/data2/yyy/TriPRD/results/optimization/gpu_smoke~/A/cod/best.pt`；参数量 28,034,114。

无 VDT 权重复用；初始化依据 protocol 中的 ImageNet pretrained 设置。

### B checkpoint provenance

最终 COD checkpoint 路径记录：`/data2/yyy/TriPRD/results/optimization/gpu_smoke~/B/cod/best.pt`；参数量 28,034,114。

VDT 来源 metadata（来源权重 SHA 是运行端记录值，本地无权重可重新计算）：

```json
{
  "checkpoint": "/data2/yyy/TriPRD/results/optimization/gpu_smoke~/B/vdt/best.pt",
  "sha256": "4c0dc40cc66929b7a638b1c52219ccc22b52b93f0bef60c0f1afd570625a9149",
  "transfer_scope": "full"
}
```

### C checkpoint provenance

最终 COD checkpoint 路径记录：`/data2/yyy/TriPRD/results/optimization/gpu_smoke~/C/cod/best.pt`；参数量 28,034,114。

VDT 来源 metadata（来源权重 SHA 是运行端记录值，本地无权重可重新计算）：

```json
{
  "checkpoint": "/data2/yyy/TriPRD/results/optimization/gpu_smoke~/C/vdt/best.pt",
  "sha256": "e32d90b64379f524c725e2764d3603233bfd8bbe0821909adf286fd4f061b46a",
  "transfer_scope": "full"
}
```

### D checkpoint provenance

最终 COD checkpoint 路径记录：`/data2/yyy/TriPRD/results/optimization/gpu_smoke~/D/cod/best.pt`；参数量 28,034,114。

VDT 来源 metadata（来源权重 SHA 是运行端记录值，本地无权重可重新计算）：

```json
{
  "checkpoint": "/data2/yyy/TriPRD/results/optimization/gpu_smoke~/D/vdt/best.pt",
  "sha256": "f6d246ee8585dcc724f80a0548abbc3252d9685ee8a34e21be6cdf6307e5a3c7",
  "transfer_scope": "full"
}
```

相关汇总：[optimization_summary.md](optimization_summary.md)。
