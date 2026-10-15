source ~/.bashrc


这台服务器有 **八张 RTX 4090，每张 24 GB 显存**。建议使用 **Python 3.11 + PyTorch 2.7.1 + torchvision 0.22.1，CUDA 12.8 版**，沿用上一条已核对的官方版本组合。

`nvidia-smi` 显示的 CUDA 13.3 是驱动支持能力，不要求 PyTorch 必须安装 CUDA 13.3 版。

在服务器终端执行：

```bash
conda create -n triprd python=3.11 pip -y
conda activate triprd

python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128

# 进入服务器上的 TriPRD 项目目录
cd /你的项目路径/TriPRD

python -m pip install -r requirements-triprd.txt
python -m pip install pytest
python -m pip check
```

安装后检查 GPU：

```bash
python -c "import torch, torchvision; print('torch:',torch.__version__); print('torchvision:',torchvision.__version__); print('CUDA:',torch.version.cuda); print('available:',torch.cuda.is_available()); print('GPUs:',torch.cuda.device_count()); print(torch.cuda.get_device_name(0)); x=torch.randn(512,512,device='cuda'); print('GPU calculation:',(x@x).mean().item())"
```

应看到 `available: True`、两张 GPU，以及正常完成的计算。

**当前两张卡都已有任务占用约 11–12 GB 显存**，GPU 0 利用率达到 99%。完整训练最好等一张卡空闲后启动；不要直接同时挤入现有任务。

先执行模型测试：

```bash
python -m pytest tests/test_triprd_model.py -q
```

再用一张空闲卡检查 ResNet-50 训练链路，例如选物理 GPU 1：

```bash
CUDA_VISIBLE_DEVICES=1 python tools/run_triprd_ab.py \
  --model-variant resnet50 --pretrained \
  --groups A,B,C,D \
  --epochs 1 --vdt-epochs 1 \
  --max-samples 16 --max-test-samples 4 \
  --batch-size 1 --image-size 352 --base-channels 256 \
  --test-datasets COD10K \
  --device cuda \
  --out results/resnet50_env_check
```

这次检查仍使用实际 ResNet-50 和 352 输入，只缩短数据量与轮数。第一次启用 `--pretrained` 时，torchvision 会尝试下载 ImageNet 权重。

**当前脚本是单 GPU 训练，不会自动利用两张卡。** 先验证单卡流程，再考虑两张卡分别跑不同种子；每次运行必须使用不同输出目录。