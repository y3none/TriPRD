# TriPRD Task v4

这是基于 `correction_stratified_probe_v1` 诊断重新实现的训练代码。新的可训练 DINO 尾部、RGB 细节支路、独立 SOD/COD 解码器和任务表征蒸馏见 [完整说明](tools/TRIPRD_TASK_V4.md)。代码包不包含数据、DINOv3 官方源码或预训练权重；使用已有资源路径。

## 在服务器新目录解压

保留旧项目，以免影响已运行实验的源码哈希。以下路径按实际位置调整：

```bash
unzip triprd_task_v4_2026-10-22.zip -d /data2/yyy/TriPRD_task_v4
cd /data2/yyy/TriPRD_task_v4
conda activate triprd-v3

export DINO_REPO=/data2/yyy/TriPRD/dinov3
export DINO_WEIGHTS=/home/yyy/data/TriPRD/dinov3/facebook--dinov3-vitb16-pretrain-lvd1689m/snapshots/master/model.safetensors
export VDT_ROOT=/data2/yyy/TriPRD/datasets/VDT-2048
export COD_ROOT=/data2/yyy/TriPRD/datasets/COD_benchmark
export OUTPUT_ROOT=/data2/yyy/TriPRD_task_v4/results/seed-20261012
export GPUS=2,3

# 先检查命令与路径，再运行 RGB 主体四组控制。
DRY_RUN=1 bash tools/run_triprd_task.sh baseline
bash tools/run_triprd_task.sh baseline
```

baseline 完成后，根据内部验证选择并固定 `TRAIN_BACKBONE_BLOCKS`（默认 4）和 `DETAIL_BRANCH`（默认 1）。接着按阶段运行：

```bash
bash tools/run_triprd_task.sh teachers
bash tools/run_triprd_task.sh screen
# 方案冻结之后才运行全量 4040 训练与四测试集评测：
bash tools/run_triprd_task.sh final
# 相同阶段、配置和代码的断点恢复：
RESUME=1 bash tools/run_triprd_task.sh screen
```

每卡运行一个独立实验。默认 384 分辨率、batch 8、100 epoch；它们是待验证的起始配方。若实际显存不足，在该阶段开始前统一调整全部控制组的 batch。多个配对 seed 使用独立输出根、固定 SPLIT_SEED 并设置相应 SEED；教师也重新训练。

## 核心文件

- `models/triprd_task.py`：RGB 学生、两种容量匹配教师。
- `models/triprd_task_loss.py`：任务监督、普通 KD 和可靠区域任务 KD。
- `tools/run_triprd_task.py`：训练、完整 epoch 恢复、独立最终评测。
- `tools/run_triprd_task_queue.py` / `.sh`：分阶段多卡实验调度。
- `tools/evaluate_triprd_task.py`：正式完整四测试集结果导出。

最终报告区分 Sm、weighted F、MAE 和 adaptive/mean/max E；严格校验全量训练成员、预测覆盖和同一 checkpoint。DINOv3 预训练及 VDT 额外监督必须披露，不能称为与仅用 COD 数据的方法资源相同。详细约定与外部参考见完整说明。

## 本地验证命令

已有依赖可直接复用。需要补齐时，在安装适合服务器 CUDA 的 PyTorch 后运行 `python -m pip install -r requirements-task-v4.txt`。

```bash
python -m pytest -p no:cacheprovider tests/test_task_model.py tests/test_task_loss.py tests/test_task_runner.py tests/test_task_protocol.py tests/test_task_queue.py
python tests/check_task_dinov3.py --dino-repo "$DINO_REPO" --device cpu
```

最后一个命令只使用随机初始化的真实 DINO 结构验证梯度，不下载预训练权重。功能测试和短程 CPU 训练不代表已经获得 SOTA 精度；正式提升需真实 CUDA 训练与配对实验验证。
