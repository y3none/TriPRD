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
