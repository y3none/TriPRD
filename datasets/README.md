# VDT2048 数据管线

VDT2048 数据不纳入 Git。仓库根目录的 `.gitignore` 已忽略 `datasets/`，因此建议把数据放在仓库外部，例如：

```text
data/VDT2048/
```

代码可以读取解压目录，也可以直接读取 `.zip` 文件。它会从目录名和文件名识别 `rgb`、`thermal`、`depth`、`mask` 四种模态，并按规范化样本 ID 配对。

## 数据下载

官方仓库：<https://github.com/VDT-2048/VDT-Dataset>

官方 README 提供的百度网盘地址：<https://pan.baidu.com/s/1JyFBtjlJGf4GE2zeciN1wQ?pwd=bipy>

提取码：`bipy`。

## 10 月 13 日检查命令

数据下载完成后，在 `TriPRD` 根目录运行：

```powershell
python tools/check_vdt2048_alignment.py `
  --root "..\data\VDT2048" `
  --split train `
  --max-samples 200 `
  --output results/vdt2048_alignment_train.json
```

生成四联图：

```powershell
python tools/visualize_vdt2048.py `
  --root "..\data\VDT2048" `
  --split train `
  --max-images 20 `
  --output visualizations/vdt2048_train
```

如果数据还没有下载，检查脚本会输出 `not_ready`，不会生成伪造的对齐结果；加上 `--require-data` 才会在数据缺失时返回失败码。

## 10 月 14 日 COD 数据检查

COD 数据可以直接读取压缩包，不需要先解压到仓库。当前计划使用：

```text
COD_benchmark/
├── TrainDataset.zip   # combined train pool: COD10K-train (3040) + CAMO-train (1000)
├── TestDataset.zip    # COD10K/CAMO/CHAMELEON test
└── NC4K.zip           # NC4K test
```

`TrainDataset.zip` follows the common COD release layout and contains 4040
training pairs. The loader exposes the official subsets separately: filenames
starting with `COD10K-` are COD10K-train (3040), and filenames starting with
`camourflage_` are CAMO-train (1000). Combining those two indexes reproduces
the standard 4040-image training pool.

运行统一检查：

```powershell
python tools/check_cod_datasets.py `
  --root datasets/COD_benchmark `
  --max-samples 200 `
  --output results/cod_dataset_report.json
```

检查器会验证 RGB 与 GT 是否一一对应、图片和 mask 尺寸是否一致、mask 是否为空、样本数是否符合官方划分，并断言 COD10K train/test、CAMO train/test、NC4K test、CHAMELEON test 的目录身份。设置 `--max-samples 0` 可读取每个数据集的全部样本。`read_pair(..., strict_size=True)` 默认拒绝尺寸不一致的源样本，避免静默改变标注。

CHAMELEON 的 `animal-62` 和 `animal-74` 沿用公开数据中的原始尺寸差异。官方 CODToolbox 的评测规则是保留 GT 尺寸、将预测图 resize 到 GT 尺寸后再计算指标；`evaluation.resize_prediction_to_gt` 提供同一规则，使用 bicubic 插值处理灰度预测图。检查器会把这类样本标记为 `ok_with_alignment`，同时保留 `size_mismatches` 记录。
