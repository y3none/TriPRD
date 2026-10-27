# TriPRD staged v5

本版针对 v4 screen 中持续联合蒸馏没有改善 COD 的结果，检验源域预训练后再做 COD-only 微调。方法标识为 `triprd_staged_v5`，不修改 v4 的模型、损失、训练、队列或评测源码。原 v4 checkpoint 和结果保留。

这是有配对对照和验收报告的新实验实现；功能测试通过不是精度提高的证据。当前已验证的增益来自 R3 RGB 主体，本版保留它，不声称新迁移方案已达到 SOTA。

## 实验组与流程

| 组 | 源阶段 | COD 阶段 |
|---|---|---|
| S0 | 无 | 通用初始化，COD-only |
| S1 | VDT RGB/GT 监督 | 只迁移共享模块，COD-only |
| S2 | 同源监督 + RGB 教师可靠输出/特征 KD | 同上 |
| S3 | 同源监督 + 多模态教师可靠输出/特征 KD | 同上 |

默认源阶段 20 轮，COD 阶段 100 轮。源预算在本轮开始前统一固定；此前两个教师的源验证最优轮均为 20，只是选择该起始预算的依据，不证明新学生 20 轮最优。三条源路线的成员、初始化、步数预算及选模规则一致。RGB/MM 两路使用同一蒸馏公式，KD 相对 GT 的权重保持 1；输出温度 2，特征权重 1，保留 v4 可靠性及区域归约。

源阶段只用 VDT TRAIN 838 张优化，VDT VAL 210 张按不含 KD 的监督损失选 checkpoint，不读取 COD 图像。COD 阶段重新构建同种子学生，只加载编码器、RGB 细节和共享任务特征等非 decoder 权重；两个 decoder 都保留通用初始化，COD decoder 迁移前后的 hash 必须一致。重建 optimizer、scheduler、AMP scaler 和随机状态；该阶段不读取 VDT 图像，不加载教师，也不添加任何源监督或 KD。

源阶段 20 轮正常约 2100 个 batch 更新，已经少于 v4 联合训练的 40400 个源 batch 暴露。本实验同时改变阶段组织和源预算，不能把相对 v4 的差值全部归因于“时机”。主要结论来自 S0/S1/S2/S3 的匹配控制。S1–S3 额外使用源数据/计算，与 S0 不是资源等价实验。

## 服务器运行

完整代码包解压到新的目录，复用现有环境、数据、DINO 权重和 v4 教师。不要覆盖原 v4 目录。

`tools/run_triprd_staged_server.sh` 已填写本次服务器的默认路径，可在新项目目录直接执行 `DRY_RUN=1 bash tools/run_triprd_staged_server.sh screen` 预览，再用 `nohup bash tools/run_triprd_staged_server.sh screen > screen.log 2>&1 &` 启动。若当前 shell 仍保留旧 v4 的 OUTPUT_ROOT，先将它设为新 v5 目录；队列会拒绝复用旧目录。下面给出完整显式配置。

```bash
unzip triprd_staged_v5_2026-10-25.zip -d /data2/yyy/TriPRD_staged_v5
cd /data2/yyy/TriPRD_staged_v5
conda activate triprd-v3

export OUTPUT_ROOT=/data2/yyy/TriPRD_staged_v5/results/seed-20261012
export TEACHER_ROOT=/data2/yyy/TriPRD_task_v4/results/seed-20261012/teachers
export SPLIT_MANIFEST=/data2/yyy/TriPRD_task_v4/results/seed-20261012/_data/split_manifest.json
export DINO_REPO=/data2/yyy/TriPRD/dinov3
export DINO_WEIGHTS=$DINO_REPO/facebook--dinov3-vitb16-pretrain-lvd1689m/snapshots/master/model.safetensors
export VDT_ROOT=/data2/yyy/TriPRD/datasets/VDT-2048
export COD_ROOT=/data2/yyy/TriPRD/datasets/COD_benchmark
export GPUS=2,3,0,1,6

mkdir -p "$OUTPUT_ROOT"
DRY_RUN=1 bash tools/run_triprd_staged.sh screen
nohup bash tools/run_triprd_staged.sh screen \
  > "$OUTPUT_ROOT/screen.log" 2>&1 &
tail -F "$OUTPUT_ROOT/screen.log"
```

`screen` 自动先并行完成 source/S1–S3，然后并行完成 screen/S0–S3。最多同时三组源训练或四组 COD 训练，一组一张卡；列表中的第五张卡可空闲，不会为了占满 GPU 增加无关实验。GPU 参数是物理编号，已有 CUDA_VISIBLE_DEVICES 限制仍必须遵守。

screen 完成后自动运行汇总；也可单独执行：

```bash
python tools/summarize_triprd_staged.py --output-root "$OUTPUT_ROOT"
cat "$OUTPUT_ROOT/_queue/source/status.json"
cat "$OUTPUT_ROOT/_queue/screen/status.json"
```

汇总报告位于 `OUTPUT_ROOT/analysis/staged_summary.md`，机器可读版为同目录 `staged_summary.json`。它同时给出按预设 macro Sm 选择 checkpoint 的完整指标，以及同 epoch 对比，不拼接不同轮次的指标最优值。保守的单种子筛查要求 S3 在选中 checkpoint 的 Sm 超过 S0/S1/S2、其他指标不退化，并在多数共同轮次 Sm 胜出。即便通过也仅是初步支持，需要多种子；未通过不会自动进入 final。

原教师严格校验原 v4 训练代码、架构、资源、源成员及训练种子；本版没有删除或伪造 teacher hash。若改训练种子，需要另训匹配教师，不能把现有教师当作新种子的匹配控制。划分种子保持不变。

## 恢复与正式评测

相同配置中断后：

```bash
RESUME=1 nohup bash tools/run_triprd_staged.sh screen \
  >> "$OUTPUT_ROOT/screen.log" 2>&1 &
```

每阶段保存完整 epoch 的 model/optimizer/scheduler/scaler/RNG/history。已完成的源阶段校验后复用，已完成的 COD 组校验后跳过。未完成的当前 epoch 会重跑。不要改原目录的 seed、预算、源码或 GPU 列表来绕过恢复检查；实验变化使用新输出根。异常断电/SIGKILL 遗留锁需要先核实对应进程，正常 SIGTERM 路径会清理自己的锁。

内部 screen 使用原 v4 的 COD 3232/808 成员，每 5 轮计算来源等权 macro Sm 选 checkpoint。正式 final 是单独显式操作，应在内部验证、多种子及方案冻结之后执行：

```bash
nohup bash tools/run_triprd_staged.sh final \
  > "$OUTPUT_ROOT/final.log" 2>&1 &
```

final 重新构建通用初始化的 COD 学生，仅复用已经冻结且只见过 VDT 的 source checkpoint，不加载 screen COD 权重。训练完整 COD10K 3040 + CAMO 1000，共 4040 张，固定最后一轮，并自动用同一 RGB-only 权重、相同单尺度推理/预测导出协议评测完整 COD10K、CAMO、CHAMELEON、NC4K。开发阶段的最优轮次不能用于逐测试集换模型。

正式评测独立验证新方法 checkpoint、源预训练来源、COD decoder 初始化证据、全量训练成员与完整预测覆盖。需区分 mean/adaptive/max E，并披露 DINOv3 通用预训练、VDT 数据与教师特权模态及计算成本；源域额外监督不能宣称与只用 COD 的方法资源相同。

## 代码与验证

- `tools/run_triprd_staged.py`：两个独立阶段、共享权重迁移、严格教师复用、checkpoint 与恢复。
- `tools/run_triprd_staged_queue.py` / `.sh`：阶段顺序、多 GPU、独立结果目录。
- `tools/summarize_triprd_staged.py`：配对条件检查与成效筛查。
- `tools/evaluate_triprd_staged.py`：正式结果验证与导出。

本地测试使用实际 CPU tiny 模型与合成图像输入验证行为；无法代替服务器 DINOv3/CUDA 精度实验。没有改动原始 v4 代码包中的 40 份文件。

协议参考：[ZoomNeXt 官方配置](https://github.com/lartpang/ZoomNeXt/blob/main/configs/icod_train.py) 使用 COD10K 3040 + CAMO 1000 训练并评测四个测试集；[CamoFormer 官方 test.py](https://github.com/HVision-NKU/CamoFormer/blob/main/test.py) 使用同一 checkpoint 评测四集并将概率乘 255 后 round 保存。不同论文的 epoch 和优化器可以不同，数据、评测定义和额外资源必须明确。

复核本版测试：

```bash
python -m pytest tests/test_staged_runner.py tests/test_staged_queue.py tests/test_staged_protocol.py tests/test_staged_summary.py -q
```
