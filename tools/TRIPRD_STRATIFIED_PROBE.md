# 边界诊断提取与区域平衡 MSE 对照

2026-10-22。针对用户提供的完整 path 诊断：D 的源域全局 positive MSE 优于 C，但正可靠边中背景占 97.884%、跨边界占 0.932%；D 跨边界 MSE 0.054864，劣于 C 0.044861、零修正 0.037977、常数 0.043291。D 关闭全部传播的 macro Sm 降低 0.012780，而仅关闭修正降低 0.000007。这些是本次内部验证的开发证据，不是正式 COD 测试结果。

本次提供两个独立入口：先从已生成的 JSON 提取遗漏的边界关系与反事实结果，再用两组短程源域训练检验区域归约。两者不自动串接，不改旧的模型/损失/源码、checkpoint、报告或门槛。

## 第一步：读取现有诊断，不重新推理或训练

上传补丁并在原项目根目录解压：

```bash
cd /data2/yyy/TriPRD
conda activate triprd-v3
unzip -o /path/to/triprd_stratified_probe_2026-10-22.zip -d .

export PATH_DIAGNOSTICS_ROOT=/data2/yyy/TriPRD/results/optimization/correction_path_diagnostic_v1

python tools/summarize_triprd_boundary_diagnostics.py \
  --input-root "$PATH_DIAGNOSTICS_ROOT" --groups C,D
```

默认只打印。可加 `--output-dir /data2/yyy/TriPRD/results/optimization/correction_boundary_summary_v1` 保存到新的独立目录。该目录不得已存在或与输入互相包含。工具只依赖 Python 标准库，不加载 torch、数据或 checkpoint；核对原 manifest 合约、完整内部验证成员、分组报告与逐图文件哈希。

输出包含各域、各尺度及 pooled 的跨边界基础/修正后 BCE、修正保留比例、裁剪率；VDT 额外包含前景/背景/跨边界的全边和正可靠边数、覆盖率、MSE 三基线，以及教师/学生基准组合的反事实 BCE。

反事实有两个方向：

- `student_base_plus_teacher_delta_bce`：将教师真实差值加到学生基础关系上，和学生 `base_bce` 比较。
- `teacher_rgb_plus_student_bce`：将学生预测差值加到教师 RGB 基础关系上，和 `teacher_rgb_bce` 比较。

这些是 **关系层面的 GT 辅助诊断**，没有替换分割预测，也不是部署时可用的额外输入。旧 JSON 的这些 BCE 聚合全部有效边，而 positive MSE 仅聚合可靠边；因此全边反事实恶化不足以判定可靠子集上的修正无效。新短程实验会额外测量正可靠子集上的对应 BCE。

先观察局部裁剪和实际关系质量。教师差值在学生基准上有益、学生预测却未达到时，区域 MSE 对照有针对性；若全边反事实不佳，先按上述支持范围限制解释，结合新实验的 positive-only 结果再判断基准兼容性。若关系改善而分割变化仍小，继续检查传播敏感性。本工具不给任意数值阈值，也不自动启动实验。

## 第二步：两个源域对照，只改变区域归约

| 组 | 正可靠边 KD | 其他设置 |
|---|---|---|
| pooled_mse | 所有边汇总 SSE / 边数 | 原 joint_positive_mse 控制 |
| stratified_mse | FG/BG/crossing 各自 MSE 后，对非空区域等权平均 | 只改变三类区域的权重 |

两组使用同一个教师、同一通用预训练和随机初始化、相同 VDT TRAIN/VAL 成员、增强、batch、优化器和学习率日程。学生从头初始化，不加载之前的 probe 或 COD checkpoint。

固定保留正可靠条件 `reliability > 0`、平方误差、倍率 5、原 source/kd/relation 权重以及分割/辅助/GT 关系监督。每一区域内部仍将当前 batch 的图像和两个尺度按边数汇总，**本次不同时改成尺度等权或图像等权**。区域无可靠边时不参加区域均值；整个 batch 都为空时返回可导零。

```text
L = source_weight * (source_supervised + kd_weight * 5 * MSE)

pooled MSE = total_squared_error / total_positive_edges

stratified MSE = mean over nonempty regions r of
                (sum_squared_error_r / positive_edges_r)
```

同样的倍率不代表梯度范数完全相同，训练日志会保留裁剪前梯度范数及 AMP 跳步信息（沿用原 optimizer-step 记录口径）。本对照不能被描述成独立控制了梯度大小。

默认各训练 10 轮，固定取末轮，不按新指标选最好轮次。源域验证按原频率记录；最后额外在完整 TRAIN 上进行无增强、固定 checkpoint 的评估，因此可以区分 TRAIN 可拟合性与 VAL 泛化。常数始终来自既有教师 checkpoint 中的 TRAIN 拟合值，不重新拟合、不用 VAL 调常数。

```bash
export RUN_ROOT=/data2/yyy/TriPRD/results/optimization/correction_4gpu_ampfix/seed-20261012
export OUTPUT_ROOT=/data2/yyy/TriPRD/results/optimization/correction_stratified_probe_v1
export GPUS=2,3
export PROBE_EPOCHS=10
export RESUME=0

DRY_RUN=1 bash tools/run_triprd_stratified_probe.sh
DRY_RUN=0 bash tools/run_triprd_stratified_probe.sh

cat "$OUTPUT_ROOT/summary.txt"
```

两张卡各跑一组，无 DDP；只有一张卡时设 `GPUS=2`，会顺序执行。若已设置 `CUDA_VISIBLE_DEVICES`，所选物理卡必须在其可见范围内。教师和预训练路径从原报告继承，不重新下载。实际模型和源域成员全部从原报告核验；缺失或不一致时应恢复对应文件，不修改哈希绕过校验。

正常中断后配置不变可设 `RESUME=1`。各轮 `last.pt` 保存模型、optimizer、scheduler、scaler、全局 RNG 和 DataLoader generator 状态，恢复从下一个完整 epoch 开始；若仅最终详细评估未完成，则从保存的末轮重新评估。已完成组校验后跳过。强制退出留下锁时，确认 launcher 和子进程均已结束后再处理锁文件。

## 如何判断这次对照

先查看 `summary.txt` / `summary.json` 中 TRAIN 与 VAL 的三类支持数量和覆盖率，确认训练侧是否也有背景支配。逐项看：

1. **跨边界 positive MSE** 是否同时优于新 pooled 控制及固定零/常数基线。
2. **实际修正后 GT BCE** 是否改善，分别报告全部有效边和正可靠边，避免支持范围不同造成误判。
3. **positive_student_base_plus_teacher_delta_bce** 相对 `positive_base_bce` 是否改善；再比较学生实际 `positive_corrected_bce`，区分真实差值本身不适配与学生拟合不到位。
4. TRAIN 改善、VAL 不改善，提示需要检查泛化；TRAIN/VAL 都未改善，不能直接用延长预算掩盖。在线训练 loss 是变动模型和增强 batch 的均值，不能与固定模型的完整 TRAIN 指标混用。
5. 前景、背景恢复是否被过度牺牲，以及总体关系质量是否退化。不要只挑新的跨边界指标宣布成功。

summary 差值为 **stratified − pooled**，MSE/BCE 负值更好。这里只描述配对开发结果，不声称显著性，也不自动进入 COD screen 或 full 训练。源域改善之后仍需要匹配的 COD 对照证明目标域收益。

```text
OUTPUT_ROOT/
  manifest.json
  summary.json
  summary.txt
  pooled_mse/{run.log,comparison.json,history.json,last.pt}
  stratified_mse/{run.log,comparison.json,history.json,last.pt}
```

## 本地验证范围

验证使用 PyTorch 2.8.0 CPU 的真实 tiny 模型和合成 I/O：独立逐边损失/梯度 oracle、原 pooled 训练权重和完整训练状态复现、固定末轮、精确恢复、训练/验证详细统计、旧文件不变和禁止 COD 访问。调度另用真实轻量子进程检查物理 GPU 隔离和 Bash 参数传递。真实服务器 DINOv3/CUDA 的边界恢复效果仍待实际运行，不把 CPU 功能测试当作精度结果。

本次新增 **32 项测试、23 个子测试通过**。另外，使用真实 tiny C/D path 诊断产物验证了只读边界摘要入口兼容；此前 MSE screen、path 诊断及其依赖的 19 个已记录文件哈希均保持不变。
