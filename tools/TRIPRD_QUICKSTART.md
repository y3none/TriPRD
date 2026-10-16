# TriPRD 可行性运行

**更新：2026-10-16。** E001 后的 COD 配方筛选和 encoder-only 迁移/第二种子 A 复验均已完成。入口、历史命令及执行状态见 [四卡优化实验说明](TRIPRD_OPTIMIZATION.md)，结果见 [优化实验汇总](../experiments/optimization_summary.md)。A 新配方在两个种子上提高来源等权 Sm，但仍有 CAMO 退步；encoder-only 未消除 B/C 负迁移。本轮没有新四测试集结果，也没有完整三种子确认结论。

本文保留上一版默认配方用于复现和排错。优化 runner 已支持正式验证 Sm 选模、固定数据划分、种子分离和权重复用；旧 wrapper 不会自动启用这些选项。当前 `--phase distill` 的 uniform Gram/概率边界 KD 消融已实现但未执行；[Paper Scope](../paper_scope.md) 中的显式跨模态关系教师、学生关系分支及仅迁移关系分支的下一版机制仍待实现，与该旧队列不同。研究计划见 [CVPR 2027 计划](../cvpr2027_submission_plan.md)。

这套脚本提供两种模式：`tiny` 使用 `models/triprd.py` 中的轻量 U-Net 风格网络，用于快速检查；`resnet50` 使用 ImageNet-compatible ResNet-50 和 352×352 输入。项目支持 `previous_performance.tex` 中的指标，但同一指标名称不代表训练协议、预训练资源和测试设置已与前人对齐；内部验证均值不能直接与论文测试集分数比较。
若只做最快的端到端 smoke，可在下面每条训练/测试命令中加上 `--image-size 64 --base-channels 8`；三阶段必须使用相同的这两个值。

运行环境至少需要 PyTorch、NumPy、Pillow；评测还需要项目已有的 `opencv-python`、`scipy` 和 `tqdm`。请先把 VDT-2048 和 COD benchmark 按 `configs/data_protocol.yaml` 的目录或 ZIP 形式准备好。

## 旧默认配方的一键运行（历史复现）

下面的历史对照脚本固定 VDT/COD 的 train/validation 划分，使用相同学生网络、初始化和 COD 预算，并按 validation Dice 选择 checkpoint。新实验使用优化说明中的验证筛选协议，不应把本段默认执行当作尚未完成的下一波：

```powershell
python tools/run_triprd_ab.py `
  --vdt-root datasets/VDT-2048 --cod-root datasets/COD_benchmark `
  --model-variant resnet50 --pretrained `
  --epochs 100 --vdt-epochs 100 --max-samples 0 `
  --image-size 352 --base-channels 256 --batch-size 2 `
  --out results/triprd_ab
```

输出包括 `split_manifest.json`、各组的 `metrics.jsonl` 和 `best.pt`、所请求测试集的 RGB-only 预测，以及 `comparison.json`。四组含义是：A 直接 COD RGB 训练；B VDT RGB 监督预训练后 COD 微调；C VDT logit KD 后 COD 微调；D 上一版融合特征关系/边界/disagreement 门控 KD 后 COD 微调。此处 D 不包含下一版显式关系教师或 learned uncertainty 模块。只有最后统一测试，不能用测试集选模型；方法筛选使用 `--skip-test`，冻结后才执行评估。

PowerShell wrapper 也会执行同一套 A/B/C/D 训练、RGB-only 测试和评测：

```powershell
.\tools\run_triprd_smoke.ps1
```

复现旧版 ResNet-50 配方可使用专用脚本：

```powershell
.\tools\run_triprd_full.ps1
```

该脚本默认使用 ResNet-50、ImageNet 初始化、352×352 输入、256 解码器通道、VDT/COD 各 100 个 epoch、全部训练样本和四个测试集；checkpoint 只按验证 Dice 选择。可用 `-VdtRoot`、`-CodRoot`、`-OutputRoot`、`-VdtEpochs` 和 `-CodEpochs` 覆盖默认值。

Linux 上使用 `CUDA_VISIBLE_DEVICES=3 bash tools/run_triprd_full.sh`；该 Bash 脚本将终端输出追加到输出目录的 `run.log` 并传递退出状态。PowerShell 脚本将输出显示在终端，需另行保留完整日志。

上面的 smoke 脚本默认是 64×64、8 通道、VDT/COD 各 2 个 epoch、每个训练源最多 64 张图；这只能检查链路和损失是否正常，不能作为正式效果结论。完整对照运行示例：

```powershell
.\tools\run_triprd_smoke.ps1 `
  -Epochs 20 -VdtEpochs 20 -MaxSamples 0 -MaxTestSamples 0 `
  -ImageSize 128 -BaseChannels 24 `
  -Datasets COD10K,CAMO,NC4K,CHAMELEON
```

使用四个官方测试集和完整训练样本上限的旧版 ResNet-50 示例：

```powershell
.\tools\run_triprd_smoke.ps1 `
  -ModelVariant resnet50 -Pretrained `
  -Epochs 100 -VdtEpochs 100 -MaxSamples 0 -MaxTestSamples 0 `
  -ImageSize 352 -BaseChannels 256 `
  -Datasets COD10K,CAMO,NC4K,CHAMELEON `
  -OutputRoot results/triprd_resnet50
```

`--pretrained` 需要本机缓存或可下载 torchvision 的 ResNet-50 权重；启用后 RGB 自动使用 ImageNet mean/std 归一化。若实验服务器不能联网，先手工放置权重；去掉该开关会改变实验条件，必须注明且不能与 ImageNet 初始化组混作同配方对照。这里使用 NC4K、COD10K-Test、CAMO-Test、CHAMELEON，以及 Sm、weighted F、mean E、MAE。比较前人 E 指标时仍须核对其使用 mean/max/adaptive 哪一版本。

`--max-samples 0` 仅取消样本截断，仍保留内部 train/validation 划分，不表示最终使用全部 4040 张 COD 训练图。全训练集固定预算模式仍待实现。官方已训练 COD checkpoint 可能见过内部验证图片，只适合独立复现其官方测试结果，不能直接作为该内部验证集上的公平基线。

跑完后可以把前人表格和 TriPRD-D 合并成 CSV：

```powershell
python tools/merge_prior_comparison.py `
  --prior previous_performance.tex `
  --triprd results/triprd_resnet50/comparison.json `
  --group D `
  --output results/prior_comparison.csv
```

## 1. 旧版分阶段 smoke（仅用于排错）

下面的 `train_triprd.py`、`test_triprd.py` 命令保留用于定位环境和数据问题；它们不是方法有效性对照。需要得出 A/B/C/D 科学结论时使用上面的 `run_triprd_ab.py`。

### 训练多模态教师

```powershell
python tools/train_triprd.py --stage teacher `
  --vdt-root datasets/VDT-2048 --epochs 2 --max-samples 64 `
  --out checkpoints/TriPRD_teacher.pt
```

教师训练阶段使用 VDT-2048 train 的 RGB、Thermal、Depth 和 mask。

### 训练 RGB 学生并蒸馏

```powershell
python tools/train_triprd.py --stage student `
  --vdt-root datasets/VDT-2048 `
  --teacher-checkpoint checkpoints/TriPRD_teacher.pt `
  --epochs 2 --max-samples 64 --out checkpoints/TriPRD_student.pt
```

教师在此阶段冻结，学生的主输入是 RGB；部署/测试时不需要教师或 T/D。上一版融合特征关系、边界和 disagreement 门控对照使用 `run_triprd_ab.py` 的 D 组；新的显式关系机制仍待实现。

### 在真实 COD 训练集上微调学生

```powershell
python tools/train_triprd.py --stage cod `
  --cod-root datasets/COD_benchmark `
  --student-checkpoint checkpoints/TriPRD_student.pt `
  --epochs 5 --out checkpoints/TriPRD_cod.pt
```

COD 微调只使用 COD10K-train 和 CAMO-train 的 RGB/mask。

### 测试并导出预测

先用小样本检查链路：

```powershell
python tools/test_triprd.py --checkpoint checkpoints/TriPRD_cod.pt `
  --data-root datasets/COD_benchmark `
  --dataset COD10K --max-samples 20 `
  --pred-root results/predictions/triprd
```

去掉 `--max-samples` 并重复 `--dataset` 可生成四个测试集的完整预测。目录结构为 `pred-root/<DATASET>/<sample_id>.png`。

### 评测

```powershell
python tools/evaluate_triprd.py --data-root datasets/COD_benchmark `
  --pred-root results/predictions/triprd --dataset COD10K
```

输出包含 `S_alpha`、`F_beta_w`、`E_phi`、`MAE`、`Boundary_F` 等项目统一指标，并写入 `results/triprd_metrics.json`。

正式结果前应移除 `--max-samples`，固定 checkpoint、seed、输入尺寸和数据 split；不要用 COD test 指标选择 checkpoint。
