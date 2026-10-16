# E013_relation_C_rgb_relation

- Recorded at: 2026-10-16（Asia/Singapore，登记日期）
- Run directory: `results/optimization/relation/seed-20261012/C_rgb_relation`
- Tags: relation-v1,validation-only,matched-ablation
- Run started (UTC): `2026-10-16T02:11:45.986317+00:00`
- Run finished (UTC): `2026-10-16T09:45:18.910393+00:00`
- Status: comparison 记录 `complete`；进程退出码未独立核验。

## Change

三编码器 RGB/RGB/RGB 教师监督训练，再向冻结主干的 RGB 学生蒸馏局部关系分布，仅迁移关系分支。

## Hypothesis

与 D 容量匹配的纯 RGB 教师可分离教师容量与真实额外模态的贡献。

## Protocol

本节 JSON 原样转录本运行 comparison.protocol，非当前默认参数推断。四组为同结构 RGB 学生，参数量 28,138,562，其中关系分支 104,448；C/D 三编码器教师均为 75,154,626。A 不做 VDT；B 做 RGB 监督；C 与 D 仅在教师输入模态上作匹配比较（C 为 RGB/RGB/RGB，D 为 RGB/T/D）。学生在 VDT 阶段冻结 encoder 参数及 BN 状态，COD 阶段恢复训练；仅关系分支迁移，decoder/head 与主干以 A 相同初始状态重新构建。C/D 源学生使用 BCE+IoU 和权重 1、温度 1 的局部分类 KL；B 与所有 COD 阶段只有 BCE+IoU，不启用 logit/boundary KD。关系在 stride-16 特征上预测两组 3×3 邻域分布并参与残差聚合，关系维度 32。

COD 训练/划分 seed 均为 20261012；ResNet-50 ImageNet，352 输入，base channels 256，batch 8；AdamW lr=1e-4、weight decay=1e-4，encoder LR×0.1，cosine、warmup 5、min ratio 0.05。max_samples=0，本轮 cod_train_mode=split：3232 张训练、808 张验证（COD10K 608、CAMO 200），不是全部 4040 张的最终训练。B/C/D 的 VDT 为 838 训练、210 验证，各源阶段 100 轮；teacher/source 以缩放尺寸 soft Dice 选模。COD 为 100 轮，在第 5、10…100 轮计算正式验证指标，共 20 个候选 checkpoint；四组均选 epoch 70 的来源等权宏 Sm。未评估的轮次不能声称其 Sm 更低。

验证先按来源逐图平均，再以 COD10K/CAMO 各 50% 合成宏值。预测经 uint8→PIL 双线性恢复原 GT 尺寸；指标内部逐图 min-max 归一化。Sm/Fw/mean E 越高越好，MAE 越低越好；E_phi 是 256 阈值曲线均值，不是 max/adaptive E。

```json
{
  "groups": "C",
  "out": "/data2/yyy/TriPRD/results/optimization/relation/seed-20261012/C_rgb_relation",
  "vdt_root": "/data2/yyy/TriPRD/datasets/VDT-2048",
  "cod_root": "/data2/yyy/TriPRD/datasets/COD_benchmark",
  "model_variant": "resnet50",
  "pretrained": true,
  "base_channels": 256,
  "image_size": 352,
  "relation_dim": 32,
  "relation_kernel": 3,
  "relation_weight": 1.0,
  "relation_temperature": 1.0,
  "epochs": 100,
  "vdt_epochs": 100,
  "batch_size": 8,
  "max_samples": 0,
  "train_seed": 20261012,
  "split_seed": 20261012,
  "split_manifest": null,
  "val_fraction": 0.2,
  "val_metrics_every": 5,
  "selection_metric": "S_alpha",
  "cod_train_mode": "split",
  "learning_rate": 0.0001,
  "weight_decay": 0.0001,
  "cod_encoder_lr_scale": 0.1,
  "cod_scheduler": "cosine",
  "cod_warmup_epochs": 5,
  "cod_min_lr_ratio": 0.05,
  "device": "cuda",
  "teacher_checkpoint": [],
  "relation_checkpoint": [],
  "cod_checkpoint": [],
  "skip_test": true,
  "eval_only": false,
  "test_datasets": "COD10K,CAMO,CHAMELEON,NC4K",
  "max_test_samples": 0,
  "method": "triprd_relation_v1",
  "rgb_normalization": "ImageNet",
  "sample_counts": {
    "vdt_train": 838,
    "vdt_val": 210,
    "cod_train": 3232,
    "cod_val": 808
  },
  "split_manifest_sha256": "a781db0cc97fec81ed87eff0a07925efc287039a684bf68e815a54dbbcb5e3bc",
  "code_commit": "074cc55b81c45dd5b4105f789f19c662e63eedfc",
  "code_dirty": true,
  "source_sha256": {
    "models/triprd_relation.py": "ee67a807bfce6f17ad0b0a7007712843237d9ae722411c9ec0d830104dae0e5d",
    "models/triprd.py": "fb7fc3d091cb6d993104c566b2bd6ff8e85a7d1dcc0dce8c94611ada4123d7eb",
    "models/triprd_full.py": "823a10a2b9f040d03eca58b02d758d34eeef360534298f7a823534bceed30065",
    "tools/run_triprd_relation.py": "5ac638bfaa2777184c00e7c0704615fd149b0c84e5567cceed97ba55d6b5120b",
    "tools/triprd_relation_training.py": "f3601a16fcf744683ec325bb3ddcef97a593f034def87de118d497b04270bc62",
    "tools/run_triprd_ab.py": "f0f549dc4b59942f4b501c71375fc4a9befb640cf6dbfb47d87eca409cb3ce56",
    "tools/triprd_training.py": "2536dfd13be0f555b6d478511d74471b4d2783800de9099e29835499a6c8b55b",
    "evaluation/validation.py": "15ea8cee017dc5f4feb7958e88c6808409d943a86d5c45ffba875d0f59b4168a",
    "evaluation/cod_metrics.py": "275d8192dd92292ddb0a67e9a6b0af36c4c29cc093367715d0946891e3da3e5f",
    "evaluation/camoformer_metrics.py": "0b99d035113861e4720f238eb2156e8dde74ea021010df79014eeeb0d66c5a5a",
    "tools/evaluate_cod.py": "d0b17fe18189e5d3bc16b92fca71765c9944f53a975e0124b5b5748ce7b8d1d7",
    "datasets/cod_dataset.py": "acbb1a299bca85174319ab1a6badedb34986b2b1d7723fc0d8b66c8499e3eddd",
    "datasets/vdt2048.py": "5852b03e736f3950badffe0e76141ed9a561978d79ea4254a0a1e5aabce535b2"
  },
  "python_version": "3.11.17",
  "torch_version": "2.7.1+cu128",
  "cuda_version": "12.8",
  "gpu_name": "NVIDIA GeForce RTX 4090",
  "transfer_scope": "relation_branch_only",
  "source_backbone_frozen": true,
  "source_backbone_bn_frozen": true,
  "source_head_transferred": false,
  "teacher_modalities": {
    "C": "RGB/RGB/RGB",
    "D": "RGB/thermal/depth"
  },
  "thermal_depth_preprocessing": "[0,1], repeated to 3 channels; RGB uses rgb_normalization",
  "validation_source_weights": "equal",
  "validation_prediction": "uint8_then_PIL_bilinear_to_original_GT",
  "optimizer": "AdamW",
  "train_augmentation": "horizontal_flip_p=0.5"
}
```

## Validation selection

Official validation metrics below belong to the selected epoch; macro averages weight each source equally.

| Group | Best epoch | Selection metric | Selection score | S_alpha | F_beta_w | E_phi mean | MAE | Status / error |
|---|---:|---|---:|---:|---:|---:|---:|---|
| C | 70 | S_alpha | 0.807881 | 0.807881 | 0.701479 | 0.870593 | 0.048955 | ok |

### Selected-epoch validation by source

| Group | Source | Epoch | Samples | S_alpha | F_beta_w | E_phi mean | MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| C | COD10K | 70 | 608 | 0.798671 | 0.663328 | 0.871783 | 0.035980 |
| C | CAMO | 70 | 200 | 0.817091 | 0.739630 | 0.869404 | 0.061930 |

## Results

Test status: `skipped`；`test={}`。只有验证筛选结果，尚无 COD10K/CAMO/CHAMELEON/NC4K 测试集成绩；未生成测试预测属于预期。不与 E001 的测试表直接比较，也不以本轮验证分数宣称 SOTA。

## Analysis

相对 A，宏 Sm -0.000787、Fw -0.002388，mean E +0.001749、MAE -0.000037。相对 B 的 Sm 仅 +0.000117；单 seed 小差异不足以证明稳定蒸馏收益。

### Comparison with previous run

相对 [E011_relation_A_scratch](E011_relation_A_scratch.md)，本记录选中轮减去对照选中轮：

| 来源 | ΔSm ↑ | ΔFw ↑ | Δmean E ↑ | ΔMAE ↓ |
| --- | --- | --- | --- | --- |
| macro | -0.000787 | -0.002388 | +0.001749 | -0.000037 |
| COD10K | -0.000049 | -0.000112 | +0.003636 | +0.000164 |
| CAMO | -0.001524 | -0.004665 | -0.000139 | -0.000237 |

相对 [E012_relation_B_rgb_source](E012_relation_B_rgb_source.md)，本记录选中轮减去对照选中轮：

| 来源 | ΔSm ↑ | ΔFw ↑ | Δmean E ↑ | ΔMAE ↓ |
| --- | --- | --- | --- | --- |
| macro | +0.000117 | +0.000072 | +0.001089 | -0.000141 |
| COD10K | +0.001489 | +0.002707 | +0.000770 | -0.000001 |
| CAMO | -0.001255 | -0.002563 | +0.001407 | -0.000282 |

### Failure cases and observations

单训练 seed；未做显著性检验，没有逐图结果可做配对 bootstrap，不能写成稳定收益或稳定退步。四组在正式评估的 20 个候选轮中均于第 70 轮达到最佳宏 Sm，后续训练 loss 降低而未超过该 Sm，目前没有单纯延长训练的证据。

多模态教师的本轮 VDT soft Dice 高于 RGB 教师（0.904641 对 0.855075），但对应源学生为 0.827633 对 0.829510，且无 KD 的 B 为 0.830162。教师变强没有转化为本轮学生或 COD 主指标优势。关系归一化熵接近 1 提示平均邻域分布接近均匀；这是平均诊断，不能证明每个位置坍缩，也不能推断强制变尖必然有利。

代码中的教师还可通过多尺度 RGB/T/D 均值融合及 T/D value 特征获得分割收益，而学生 KD 目标只有局部关系分布；所以“教师更强”并不保证“关系目标包含更多有用知识”。这是基于实现的信息路径假设，不是已经由干预实验确认的因果结论。

## Decision

保留容量匹配的蒸馏对照；先检查教师是否依赖所蒸馏关系，暂不扩大训练。

相关联合分析及下一步见 [relation_summary.md](relation_summary.md)。

## Source-stage diagnostics

| 阶段 | Best epoch (Dice) | VDT val soft Dice | 自身归一化关系熵 | 冻结教师归一化关系熵 | Relation KL | Weighted KL |
| --- | --- | --- | --- | --- | --- | --- |
| teacher | 89 | 0.855075 | 0.995166 | — | 0.000000 | 0.000000 |
| source student | 89 | 0.829510 | 0.998245 | 0.995166 | 0.006672 | 0.006672 |

熵使用温度 1 的实际前向分布，按每个 query 的有效邻居数归一化，再跨关系类型、位置及样本平均；边界 padding 不作为有效邻居。C/D 的 KL 为 teacher→student，温度 1、权重 1。上表来自各阶段 Dice 选中轮的 val，不能混同 COD 原尺寸 Sm。

COD epoch 70 的 val relation entropy 为 `0.8577704701093164`，该阶段 relation_kd=`0.0`。

## Artifacts and provenance

- 本地原始文件：`results/optimization/relation/seed-20261012/C_rgb_relation/comparison.json`（不随 Git 提交）。
- comparison SHA256：`979e60517fe2649f8633e8b746942570192617209601c4cd8b4a5d9cdedc1860`。
- JSON 中输出目录：`/data2/yyy/TriPRD/results/optimization/relation/seed-20261012/C_rgb_relation`（未独立访问）。
- split_manifest 整文件 SHA256：`a781db0cc97fec81ed87eff0a07925efc287039a684bf68e815a54dbbcb5e3bc`。
- 记录的 COD checkpoint：`/data2/yyy/TriPRD/results/optimization/relation/seed-20261012/C_rgb_relation/C/cod/best.pt`。
- 记录的 COD checkpoint SHA256：`6aab8a3ae08ecbcbf067d2eb7059c0d232bca3f8cdddd978468016b04b9bc3e7`。

本地每个运行目录只有 comparison.json（内嵌训练 history）。没有同步 run.log、launch.json、exit_status.json、split_manifest.json、分离的 metrics.jsonl/history.json、模型权重或预测图。下述 checkpoint、artifact 与 manifest 哈希均为 JSON 记录值，未独立读取相应文件复核；未独立访问服务器，也未核验样本成员/重叠或逐图复算。status=complete 不能代替进程退出码。没有具体预测样本，不能编造失败图像或类别。

四份 protocol.source_sha256 相同，所列 13 个文件已在登记时逐个与本地源码 SHA256 比对，一致。该列表不覆盖 queue launcher 及所有依赖；服务器运行基点是 074cc55b81c45dd5b4105f789f19c662e63eedfc 且 code_dirty=true，不能把实验归因到一个未经验证的干净提交。

B/C/D 记录的 split_manifest 整文件哈希相同，A 哈希不同；A 的 VDT 样本数为 0，B/C/D 含 VDT 划分，因此整文件内容本就不完全相同。四组 split_seed、COD 样本数相同且源码路径一致，但在缺少 manifest 文件时不能把这些元数据当作 COD 成员已独立核验的证明。

### Initialization and transfer audit

以下为 JSON 内部的状态哈希与检查记录。四组 initial_state_sha256 完全一致；COD 初始 encoder/decoder/heads 与 A 相同，B/C/D 只有 relation_branch 改变。frozen_encoder_verified 的运行时检查覆盖 encoder 权重和 BN buffers，未在本地重新加载权重复查。relation_state_sha256 与 cod_initial_state_sha256.relation_branch 的序列化摘要函数不同，不要求两者字符串相等。

```json
{
  "initial_state_sha256": {
    "encoder": "0436eb3bcc7f0fbe18cc74a093bbe1d5d8b29d0ab2ff1c536246cb4597272486",
    "decoder": "d4efdb257524d63cb4ccd1a835a24a7d691b990d122cedf8b685066d301fd833",
    "heads": "d798730c17e58ad060efb93d38b173fd012d360302e213d731cd7a509d1dec61",
    "relation_branch": "ecba06ca00ac3ec85bb55c204bbd30c05c20efd2a657a7f791e87df4e4c2c17c"
  },
  "cod_initial_state_sha256": {
    "encoder": "0436eb3bcc7f0fbe18cc74a093bbe1d5d8b29d0ab2ff1c536246cb4597272486",
    "decoder": "d4efdb257524d63cb4ccd1a835a24a7d691b990d122cedf8b685066d301fd833",
    "heads": "d798730c17e58ad060efb93d38b173fd012d360302e213d731cd7a509d1dec61",
    "relation_branch": "73d38bb7dd6b9c4eb40927f79834ccd340bed23aaddf8d67099e2235a4e771ed"
  },
  "frozen_encoder_verified": true,
  "transfer": {
    "format": "triprd_relation_branch_v1",
    "source_stage": "C/vdt",
    "source_epoch": 89,
    "source_seed": 20261012,
    "source_checkpoint_path": "/data2/yyy/TriPRD/results/optimization/relation/seed-20261012/C_rgb_relation/C/vdt/best.pt",
    "source_checkpoint_sha256": "fb9462e8c87a86f0d8c7cfb0a67dd832eb1c2de0549c27f59a0216898ac862df",
    "relation_keys": [
      "keys.0.weight",
      "keys.1.weight",
      "output_projection.weight",
      "queries.0.weight",
      "queries.1.weight",
      "shared_projection.weight",
      "values.0.weight",
      "values.1.weight"
    ],
    "relation_state_sha256": "d7fc6f3e87c8b76b6bd18041b8f6c704f35b351f255553ee2c73a251bc840931",
    "artifact_path": "/data2/yyy/TriPRD/results/optimization/relation/seed-20261012/C_rgb_relation/C/vdt/relation.pt",
    "artifact_sha256": "b3a5d1625cad40a4cb34e03c5045430340d9fe757dea3287c6ac48de717809f9",
    "transfer_scope": "relation_branch",
    "preprocessing_verified": true
  },
  "teacher": {
    "checkpoint": "/data2/yyy/TriPRD/results/optimization/relation/seed-20261012/C_rgb_relation/C/teacher/best.pt",
    "sha256": "a80a61c6cdee7def941a156c6afb1079b07effa55b3dc0fec714f3099a5833f2",
    "parameters": 75154626,
    "teacher_kind": "rgb",
    "best_epoch": 89,
    "best_val_dice": 0.8550748223350162
  }
}
```

相关汇总：[relation_summary.md](relation_summary.md)。
