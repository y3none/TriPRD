# TriPRD 可行性运行

这套脚本提供两种模式：`tiny` 使用 `models/triprd.py` 中的轻量 U-Net 风格网络，用于快速检查；`resnet50` 使用 ImageNet-compatible ResNet-50 和 352×352 输入，用于和 `previous_performance.tex` 的前人结果按同一指标口径比较。
若只做最快的端到端 smoke，可在下面每条训练/测试命令中加上 `--image-size 64 --base-channels 8`；三阶段必须使用相同的这两个值。

运行环境至少需要 PyTorch、NumPy、Pillow；评测还需要项目已有的 `opencv-python`、`scipy` 和 `tqdm`。请先把 VDT-2048 和 COD benchmark 按 `configs/data_protocol.yaml` 的目录或 ZIP 形式准备好。

## 一键运行

要验证方法本身是否有效，优先运行严格对照脚本。它固定 VDT/COD 的 train/validation 划分，使用相同学生网络、初始化和 COD 预算，并按 validation Dice 选择 checkpoint：

```powershell
python tools/run_triprd_ab.py `
  --vdt-root datasets/VDT-2048 --cod-root datasets/COD_benchmark `
  --model-variant resnet50 --pretrained `
  --epochs 100 --vdt-epochs 100 --max-samples 0 `
  --image-size 352 --base-channels 256 --batch-size 2 `
  --out results/triprd_ab
```

输出包括 `split_manifest.json`、各组的 `metrics.jsonl` 和 `best.pt`、所请求测试集的 RGB-only 预测，以及 `comparison.json`。四组含义是：A 直接 COD RGB 训练；B VDT RGB 监督预训练后 COD 微调；C VDT logit KD 后 COD 微调；D VDT 完整关系/边界/不确定性门控 KD 后 COD 微调。只有最后统一测试，不能用测试集选模型。

PowerShell wrapper 也会执行同一套 A/B/C/D 训练、RGB-only 测试和评测：

```powershell
.\tools\run_triprd_smoke.ps1
```

正式 ResNet-50 全量实验请使用专用脚本：

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

和前人表格直接对齐的完整 ResNet-50 版本：

```powershell
.\tools\run_triprd_smoke.ps1 `
  -ModelVariant resnet50 -Pretrained `
  -Epochs 100 -VdtEpochs 100 -MaxSamples 0 -MaxTestSamples 0 `
  -ImageSize 352 -BaseChannels 256 `
  -Datasets COD10K,CAMO,NC4K,CHAMELEON `
  -OutputRoot results/triprd_resnet50
```

`--pretrained` 需要本机已有 torchvision 的 ResNet-50 权重；启用后 RGB 自动使用 ImageNet mean/std 归一化。若实验服务器不能联网，先手工放置权重或去掉该开关，并在结果中注明未使用 ImageNet 初始化。这个版本的指标口径与 `previous_performance.tex` 一致：NC4K、COD10K-Test、CAMO-Test、CHAMELEON，以及 $S_m$、$alpha E$、$F_m^w$、$M$。

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

教师在此阶段冻结，学生的主输入是 RGB；部署/测试时不需要教师或 T/D。正式的关系、边界和不确定性门控对照请使用 `run_triprd_ab.py` 的 D 组。

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
