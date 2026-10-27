# 已运行的 screen 从两卡扩展到五卡

本补丁仅新增或更新扩容工具，不覆盖 v4 的模型、训练、评测或原队列源码，因此不改变教师/学生 checkpoint 的源码契约。它读取服务器现有 screen 计划，不重新填写训练参数，也不重新训练教师。

本次映射为 T0→GPU 2、T1→GPU 3、T2→GPU 0、T3→GPU 1、T4→GPU 6。GPU 参数使用物理编号，原来的两张卡保持在列表前两位。

## 使用

将补丁 ZIP 上传到服务器，在旧 v4 项目目录解压，或只把 `expand_triprd_task_screen.py` 放到已有 `tools/` 中。只更新扩容补丁文件；不要重新覆盖整份 v4 项目。

```bash
cd /data2/yyy/TriPRD_task_v4
conda activate triprd-v3
unzip -o triprd_task_v4_expand_5gpu_fix1_2026-10-23.zip

export OUTPUT_ROOT=/data2/yyy/TriPRD_task_v4/results/seed-20261012

# 只检查计划与资源，不停止训练。
python tools/expand_triprd_task_screen.py \
  --output-root "$OUTPUT_ROOT" --gpus 2,3,0,1,6

# 执行扩容。由工具核验并停止原队列，无需手动 kill 或删除锁。
nohup python -u tools/expand_triprd_task_screen.py \
  --output-root "$OUTPUT_ROOT" --gpus 2,3,0,1,6 --execute \
  > "$OUTPUT_ROOT/screen_expand.log" 2>&1 &

tail -F "$OUTPUT_ROOT/screen_expand.log"
```

如预览报告已有 `CUDA_VISIBLE_DEVICES` 限制，先检查服务器分配给当前会话的卡；本工具不自动扩大环境声明的可见范围。运行时保持本次确认的五张卡可用。

## 旧版出现 `/proc/<pid>/cwd` PermissionError 后恢复

旧版在遍历同用户进程时，会读取无关进程的工作目录；Linux 可能禁止此项读取。fix1 先按训练命令及输出目录筛选，扫描绝对输出路径时不需要读取 cwd。对真正需要身份核验的进程，权限不足仍会明确报错，不能据此认定进程已退出或删除锁。

更新上面的修复包后，使用文末带 `--execute --resume` 的命令重试。这次异常可能发生在原队列已停止、扩容计划尚未写入时；脚本支持从这个状态恢复，无需手动删除锁或使用 sudo。T0/T1 从已有的最近完整 epoch 继续，未完整保存的 epoch 会重跑。日志使用 `>>` 追加以保留失败现场。

## 保留什么

- 先核对旧计划、源码、数据划分和教师 checkpoint 哈希。
- 在 Linux `/proc` 中验证旧队列的用户、工作目录、脚本、阶段、输出目录及 PID 身份后，发送 SIGTERM，让旧队列停止其训练子进程。
- 等旧训练进程及其 DataLoader 进程组结束，额外扫描尚未登记到队列状态中的训练进程。存在任何同任务存活进程时拒绝重复启动。
- 只恢复经核验已死亡进程留下的锁，不删除 checkpoint。
- T0/T1 从 `last.pt` 保存的最近完整 epoch 恢复 optimizer、scheduler、scaler 和 RNG。尚未保存的当前 epoch 会重跑；若还没有完整 epoch，日志会明确显示从 0 开始。
- T2–T4 首次启动；已经完成的组核验 checkpoint 后跳过训练。
- 原 `run.log` 追加写入，原 screen plan 保留。新调度计划和迁移记录位于 `_queue/screen_expanded/`；模型输出仍在原来的 `screen/T0` 至 `screen/T4`。

监控新队列：

```bash
cat "$OUTPUT_ROOT/_queue/screen_expanded/status.json"
nvidia-smi
```

扩容后的队列中断时，用相同工具和卡列表恢复：

```bash
nohup python -u tools/expand_triprd_task_screen.py \
  --output-root "$OUTPUT_ROOT" --gpus 2,3,0,1,6 --execute --resume \
  >> "$OUTPUT_ROOT/screen_expand.log" 2>&1 &
```

扩容后不要再启动原来的两卡 `run_triprd_task.sh screen`。后续 `final` 可照常使用原脚本首次启动，设置 `GPUS=2,3,0,1,6`。

本地验证覆盖计划完整性、错误 PID/锁拒绝、真实五子进程调度和日志保留。开发机为 Windows，Linux 信号与 `/proc` 逻辑以受控测试验证；未连接服务器执行迁移或 GPU 训练。
