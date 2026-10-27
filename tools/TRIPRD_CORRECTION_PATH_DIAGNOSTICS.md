# 修正到分割输出的诊断

本诊断针对当前 positive-MSE screen：A 仅 COD、B 加源 RGB/GT、C 加完整关系 KD、D 加正可靠边修正 MSE。现有结果中 D 的源域 MSE 为 0.00260018，但 COD 上移除修正只改变约 0.000007 的 macro Sm。需要定位源域修正是否传入了实际的目标域分割。

新增工具仅加载既有 **best.pt**，不训练、不重选 checkpoint、不重新拟合常数。当前 C/D 对应 epoch 15，A/B 对应 epoch 10。读取训练池成员清单以核验谱系，实际诊断前向仅使用内部 COD VAL 和 VDT VAL，不读取官方测试样本。所有原模型、训练脚本、checkpoint 和报告保持原样。

## 在服务器运行

把新增诊断补丁包解压到当前项目根目录。它依赖已经运行成功的 AMP 修复、源域诊断和 MSE screen 文件；无需重新训练或下载权重。

```bash
cd /data2/yyy/TriPRD
conda activate triprd-v3
unzip -o /path/to/triprd_correction_path_diagnostics_2026-10-22.zip -d .

export RUN_ROOT=/data2/yyy/TriPRD/results/optimization/correction_4gpu_ampfix/seed-20261012
export DIAGNOSTICS_ROOT=/data2/yyy/TriPRD/results/optimization/correction_probe_diagnostic_v1
export SCREEN_ROOT=/data2/yyy/TriPRD/results/optimization/correction_mse_screen_v1
export OUTPUT_ROOT=/data2/yyy/TriPRD/results/optimization/correction_path_diagnostic_v1
export GPU=2
export GROUPS_TO_CHECK=A,B,C,D
export RESUME=0

# 完整核验输入和计划，不写诊断目录、不进行诊断推理。
DRY_RUN=1 bash tools/run_triprd_correction_path_diagnostics.sh

# 一张卡顺序评估，复用同一教师；每批四种干预共用学生 RGB 特征。
DRY_RUN=0 bash tools/run_triprd_correction_path_diagnostics.sh
```

默认完整覆盖 COD10K 608、CAMO 200、VDT 210 张验证图；实际数量以原成员清单为准。源域保持原 batch size/AMP，COD 逐图评估，以复现原指标口径。`GPU` 是物理 GPU 编号；进程内使用 `cuda:0`。只想优先检查 C/D 可首次设置 `GROUPS_TO_CHECK=C,D`，之后扩大组别须使用新输出目录。

中断后在相同配置、源码和输出目录下设置 `RESUME=1`。已完成组核验后跳过；未完成组从该组诊断开头重做，不承诺批次级恢复。异常强制终止可能留下 `diagnostic.lock`；确认其中 PID 及诊断进程已经退出后再清理，不应并行写同一个输出目录。

可用独立目录快速检查环境：

```bash
OUTPUT_ROOT=/data2/yyy/TriPRD/results/optimization/correction_path_smoke \
GROUPS_TO_CHECK=D DRY_RUN=0 bash tools/run_triprd_correction_path_diagnostics.sh \
  --limit-cod-per-source 2 --limit-vdt 2
```

显式限样本的部分标记为 partial，不用于完整指标复现或方法结论。正式诊断使用零限制的新目录。旧文件或哈希不匹配时恢复对应版本或文件，不手改哈希绕过核验。

## 三项检查及判断方式

### 1. 固定 checkpoint 的四种干预

| 模式 | 操作 | 回答的问题 |
|---|---|---|
| none | 正常推理 | 基准 |
| no_correction | 只移除额外修正，保留基础关系传播 | 修正是否有增量价值 |
| shuffle_correction | 每个有效 offset 内确定性空间置换原始修正 | 修正的空间位置是否重要 |
| identity | 关闭全部邻居传播 | 整个关系传播路径是否影响分割 |

所有模式共享原 checkpoint，不根据干预结果重新选择模型。差值统一使用 **none − intervention**，正 Sm 差表示正常模式更好。`identity` 是模型已有的传播开关；`shuffle` 是固定置换，不是多次随机实验。

- identity 明显下降，no_correction 基本不变：基础传播有价值，额外残差贡献有限。
- identity、no_correction 都基本不变：整个传播路径在当前指标上影响有限，继续检查浮点特征及预测。
- no_correction 明显下降，但 shuffle 不下降：可能主要利用修正的整体分布或强度，空间信息的证据不足。
- no_correction 和 shuffle 均下降：支持修正及其空间位置参与预测；仍需比较 A/C，参与不等于方法净收益。

同时记录逐图 Sm 差与原始浮点 probability 差；原 COD 指标先转 uint8、再按原图尺寸评估，小的浮点变化可能被量化隐藏。分来源报告均值、中位数、5/95 分位数和正负变化图数；这些都是固定 checkpoint 的描述性结果，不代表跨随机种子显著性。

### 2. 沿传播路径定位衰减

在 VDT、COD10K、CAMO 各自的 stride 8/16 上记录：

- 原始 `|correction|` 与实际 `|clip(base+correction)-base|`，以及裁剪比例。
- 基础关系和修正后关系相对 GT 的 BCE，分别看全边、前景内部、背景内部、跨边界。
- 传播模块实际输入/输出变化，以及干预相对正常模式的传播输出、decoder 输出、最终浮点 logits/probability 差。
- 源域学生基础关系与教师 RGB 基础关系的偏差，辅助判断教师差值能否直接加到学生基准上。

VDT 的修正明显、COD 的修正接近零，提示源到目标的表达差异；raw 明显而 applied 很小，提示裁剪/饱和；applied 明显但传播特征变化很小，提示传播对这些关系不敏感；关系 BCE 改善却分割基本不变，提示关系指标与分割收益脱节。不要只凭一个范数的绝对值设“通过阈值”；比较 A–D、两个域及干预差异更有解释力。

各层 hook 是真实前向轨迹；后一层的模式差也包含前面层的累积影响，不能当成该层独立因果贡献。

### 3. 拆解源域修正恢复

保留原 `source_recoverability` 的全局聚合，同时按尺度和 FG/BG/crossing 分解正可靠边的学生 MSE、零基线、固定 TRAIN 常数基线与边数。可靠边定义、常数和教师与原实验一致；不在 VAL 上重新拟合常数。

- D 只在背景/内部边改善，跨边界没有改善：整体 MSE 收益可能主要来自多数边。
- D 跨边界也改善，但 COD applied correction 很小：优先查域差与裁剪。
- D 修正准确，但学生修正后 GT 关系误差未改善：优先查教师与学生 base 不匹配。

空支持输出 `count=0, mean=null, max=null`，不能当作完美的零误差。pooled 是按有效边数汇总，不是对分组均值再取平均。

## 输出和读数顺序

```text
OUTPUT_ROOT/
  manifest.json
  summary.json
  diagnosis.txt
  A/{diagnostics.json,cod_per_image.jsonl}
  B/{diagnostics.json,cod_per_image.jsonl}
  C/{diagnostics.json,cod_per_image.jsonl}
  D/{diagnostics.json,cod_per_image.jsonl}
```

`diagnosis.txt` 是可直接粘贴回聊天的紧凑摘要，包含四种干预、复现差、各域的 raw/applied 修正和源域分层 MSE。运行后可以 `cat "$OUTPUT_ROOT/diagnosis.txt"`。

先看 `summary.json` 的 `reproduction`：完整诊断正常模式的 macro Sm 和 C/D positive MSE 应复现原结果（容许设备/精度产生的数值差，工具记录差值，不硬设判定阈值）。若偏差明显，先核对评估环境和输入，暂缓解释干预差。

再看 `metrics.identity` 与 `paired_S_alpha`，然后在各组 `diagnostics.json` 的 `path_statistics` 中沿 raw → applied → feature → logits 查找变化。`cod_per_image.jsonl` 保存逐图来源、四种模式 Sm 和浮点 probability 差，便于排查是否存在大幅正负变化相互抵消。

内部 `E_phi` 是 mean E；COD macro 是两个数据来源等权。关系诊断使用输入尺寸对应的 GT 网格，最终分割指标仍用原图 GT，二者不能混成同一个误差口径。

当前诊断定位显式推理路径；KD 在训练期间改变共享表示的效果仍需重训对照，不能通过一次推理关闭来完全排除。现有训练曲线的早期峰值与末期下降也保留为独立的训练日程问题，不在本工具里自动修改训练长度或学习率。

## 本地验证

PyTorch 2.8.0 CPU：新增核心/集成共 22 项测试和 8 个子测试通过，包含真实 tiny checkpoint 的 COD 指标与源域 MSE 复现、独立逐边手算、四模式一致、状态不变、空支持、CPU BF16、精确组级恢复和禁止训练/正式测试。Bash 入口语法及实际 `--help` 调用通过。旧 MSE screen 与依赖的 13 个已记录文件哈希均未变化。真实 DINOv3/CUDA 和完整服务器数据尚未执行，不能用本地测试代替服务器诊断结论。
