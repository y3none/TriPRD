# E001_triprd_full

本次登记 TriPRD A/B/C/D 全量实验及补评估结果。四组共 16 条测试集指标记录，每组报告 6473 张测试样本。完整蒸馏 D 尚未表现出相对 A 的稳定优势；保留 A 为区域分割基线、C 为 logits 蒸馏对照，下一轮先验证 COD 微调与迁移方式，再拆解 D。

- Recorded at: 2026-10-15（Asia/Shanghai）
- Run directory: `results/triprd_full`
- Tags: full-run, baseline, ablation, offline-evaluation
- Status: comparison、16 条正式指标及 800 条 epoch 历史已核验；完整日志、划分、权重和预测待补
- Updated at: 2026-10-15（补入 comparison.json 及教师历史终端片段）
- 实际训练及评估起止时间：未归档，不以本地 JSON 修改时间代替运行时间

## Change

从此前仅用于环境检查的 1 epoch、小样本实验推进到本次 A/B/C/D 完整实验。本地同步的 `results/triprd_full/comparison.json` 保存教师训练、B/C/D 的 VDT 预训练及 A/B/C/D 的 COD 训练，共 8 个阶段；每阶段 history 均连续覆盖 epoch 1–100。四个测试集已具备正式指标。

评估汇总代码已将自适应 F-measure 的读取从 `fm_result["adp"]` 修正为 `fm_result["fm"]["adp"]`。当前 comparison 的 16 个 `test.<group>.<dataset>.official` 均为成功结果，与四份独立 `official_metrics.json` 的指标逐字段一致。唯一不同的是 `model` 标签：comparison 统一为 `TriPRD-AB`，独立文件分别为 `TriPRD-A/B/C/D`，组别以外层 group 为准。当前文件不足以还原首次运行与补评估的时间顺序。

本次按 [实验记录规范](README.md) 更新同一实验 E001，以 comparison 为配置、训练历史及正式指标的主要来源，并用独立评估文件交叉核验；保留已有分析及实验 ID，不另登记一次实验。

## Hypothesis

VDT RGB 监督预训练可能改善 COD 的初始化；三模态教师的 logits 蒸馏可能在此基础上继续提高 RGB 学生的泛化；关系、边界和不确定性门控的完整组合预期进一步改善区域及边界质量。A→B、B→C、C→D 分别检验上述增量，D 还需同时与 A、B 比较。

## Protocol

### 组别定义

| Group | VDT 阶段 | COD 阶段及推理 |
|---|---|---|
| A | 无 VDT 预训练 | RGB 监督训练；RGB-only 推理 |
| B | RGB 学生监督预训练 | RGB 监督微调；RGB-only 推理 |
| C | 三模态教师 logits KD | RGB 监督微调；RGB-only 推理 |
| D | logits、关系、边界 KD 及分歧门控 | RGB 监督微调；RGB-only 推理 |

组别定义由 comparison 的 `groups.*.definition` 记录，并与当前 [runner](../tools/run_triprd_ab.py) 对应。KD 作用于 VDT 阶段，COD 阶段各组均使用监督损失。

### 运行配置与来源

| 项目 | 已知信息 | 核验状态 |
|---|---|---|
| VDT / COD epoch | 各 100；共 8 个阶段、800 条 epoch 记录 | protocol、阶段 epochs 及完整 history 一致 |
| 模型及初始化 | ResNet-50；pretrained=true | comparison.protocol 已记录 |
| 输入及解码器 | 352×352；256 channels | comparison.protocol 已记录 |
| Seed | 20261012 | comparison.protocol 已记录 |
| 归一化及增强 | ImageNet normalization；水平翻转 p=0.5 | comparison.protocol 已记录 |
| 参数量 | 教师 75,043,781；各学生 28,034,114 | teacher.parameters、groups.*.parameters |
| 教师校准温度 | 4.0 | teacher.calibration_temperature；不等同于已归档全部 KD 超参数 |
| Batch size | 当前默认 8 | comparison 未记录，实际值待核验 |
| 优化器 | 当前默认 AdamW，lr=1e-4，weight_decay=1e-4，无 scheduler | comparison 未记录，实际配置及代码版本待核验 |
| 数据划分 | 当前默认从 train 池按源划分 20% validation | 实际 val_fraction 未记录，缺 split_manifest.json |
| 训练数据覆盖 | max_samples、数据根目录及实际 train/val 样本数未记录 | 100 epoch 不能证明使用了全部训练样本；需运行实参与 split_manifest 核验 |
| Checkpoint 选择 | 当前 runner 按 validation SoftDice 最大值保存 best.pt | 8 阶段 best_val_dice 均与 history 最大值一致；权重文件未同步 |
| 启动命令及 GPU | 新终端片段末尾可见 `CUDA_VISIBLE_DEVICES=3 bash tools/run_triprd_full.sh` | 不能据此确认前文运行的实际命令或设备；缺完整日志及 GPU/依赖信息 |
| 测试样本 | NC4K 4121，COD10K 2026，CAMO 250，CHAMELEON 76 | comparison 与四组独立 JSON 的 num_samples 一致 |

comparison 保存的教师路径为 `/home/yyy/data/TriPRD/results/triprd_full/teacher/best.pt`；学生最终路径为 `/home/yyy/data/TriPRD/results/triprd_full/{A,B,C,D}/cod/best.pt`。这些是服务器路径记录，尚未验证对应权重文件内容。

当前 [Bash 脚本](../tools/run_triprd_full.sh) 与 [配置说明](../configs/triprd_resnet50_comparable.yaml) 可用于复核未记录的配置，不能将其默认值作为本次实参；runner 不读取该 YAML。

### 指标口径

- `S_alpha = S_m`；`F_beta_w = F_m_w`；`MAE = M`。
- `E_phi = alpha_E` 为阈值曲线的平均 E-measure；自适应值另存于 `E_phi_adaptive`。不与 max E 混用。
- `Boundary_F` 为项目 CamoFormer 边界区域处理后的 weighted F-measure，不是通用轮廓匹配 boundary F1。当前评估入口默认启用边界评价、boundary rate 为 15；JSON 未单独保存该参数。
- 正式 MAE 使用 `test.<group>.<dataset>.official.MAE`，与独立评估 JSON 一致；不能与外层小写 `mae` 的快速预测统计混用。
- 全部数值按 JSON 原值比较，表内保留 6 位小数。指标实现见 [evaluate_cod.py](../tools/evaluate_cod.py) 与 [evaluation](../evaluation/README.md)。

## Results

### 正式测试结果

| Group | Dataset | Samples | S_alpha ↑ | F_beta_w ↑ | E_phi mean ↑ | MAE ↓ | Boundary_F ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| A | NC4K | 4121 | 0.815043 | 0.724082 | 0.866373 | 0.059126 | 0.663734 |
| A | COD10K | 2026 | 0.792773 | 0.648107 | 0.859424 | 0.041040 | 0.605803 |
| A | CAMO | 250 | 0.751575 | 0.646179 | 0.807069 | 0.097431 | 0.595858 |
| A | CHAMELEON | 76 | 0.834322 | 0.742195 | 0.893254 | 0.048090 | 0.646660 |
| B | NC4K | 4121 | 0.813001 | 0.720284 | 0.861197 | 0.059212 | 0.656413 |
| B | COD10K | 2026 | 0.785113 | 0.635978 | 0.847451 | 0.041628 | 0.593674 |
| B | CAMO | 250 | 0.754900 | 0.649202 | 0.810776 | 0.095394 | 0.582535 |
| B | CHAMELEON | 76 | 0.828130 | 0.728709 | 0.872044 | 0.049326 | 0.630423 |
| C | NC4K | 4121 | 0.811437 | 0.715102 | 0.861197 | 0.059767 | 0.674004 |
| C | COD10K | 2026 | 0.779846 | 0.626453 | 0.844193 | 0.044795 | 0.616853 |
| C | CAMO | 250 | 0.732111 | 0.615836 | 0.783608 | 0.102990 | 0.579289 |
| C | CHAMELEON | 76 | 0.862038 | 0.783228 | 0.904248 | 0.038905 | 0.682454 |
| D | NC4K | 4121 | 0.816350 | 0.720780 | 0.865669 | 0.058595 | 0.669555 |
| D | COD10K | 2026 | 0.782287 | 0.627921 | 0.844585 | 0.042927 | 0.600468 |
| D | CAMO | 250 | 0.745370 | 0.633522 | 0.792148 | 0.100102 | 0.582871 |
| D | CHAMELEON | 76 | 0.827876 | 0.726413 | 0.891413 | 0.047754 | 0.630842 |

四组记录均包含四个预期测试集；样本数一致、指标均为有限值、所有别名与原字段一致。当前本地没有预测图，尚不能独立复核图片唯一覆盖和 checkpoint 与预测的对应关系。

### 训练收敛摘要

直接从 comparison 的 `teacher.history`、`groups.*.vdt_pretrain.history` 和 `groups.*.cod_finetune.history` 重算。8 个阶段均有连续的 100 条记录，所记 best_val_dice 均等于对应 history 中的最大 validation dice。下表替代此前基于已丢失日志的摘要；SoftDice 与正式测试指标分开报告。

| Stage | Best epoch | Best validation SoftDice | Epoch 100 train / val SoftDice |
|---|---:|---:|---:|
| teacher | 98 | 0.907944 | 0.973929 / 0.905798 |
| A/cod | 18 | 0.694850 | 0.939314 / 0.648663 |
| B/vdt | 61 | 0.849202 | 0.967949 / 0.833404 |
| B/cod | 22 | 0.679924 | 0.940154 / 0.647295 |
| C/vdt | 93 | 0.850809 | 0.965783 / 0.844380 |
| C/cod | 68 | 0.684146 | 0.934833 / 0.634463 |
| D/vdt | 46 | 0.850096 | 0.966884 / 0.845564 |
| D/cod | 18 | 0.689203 | 0.938868 / 0.664222 |

各 epoch 记录的 seconds 合计 **20.75 小时**，仅代表历史中记录的训练/验证 epoch 耗时，不是包含独立评估等流程的总运行时长。

### 蒸馏分项损失

D 的 VDT history 已保存分项损失。按当前 [loss 实现](../models/triprd.py) 的默认系数重建：`supervised = bce + iou + 0.5 × boundary`，`total = supervised + logit_kd + relation_kd + 0.5 × boundary_kd`；下列 epoch 的重建值与记录 total 的绝对差小于 1e-5。comparison 未单独保存损失系数，因此这里是与当前公式一致的核验，不将默认系数表述为独立记录的运行实参。

| D/vdt epoch | Supervised | Logit KD | Relation KD | 0.5 × Boundary KD | Total | Boundary KD / Total |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1.424542 | 8.261429 | 0.036219 | 36.460569 | 46.182759 | 78.95% |
| 5 | 0.607135 | 3.185042 | 0.027610 | 7.755179 | 11.574965 | 67.00% |
| 10 | 0.253588 | 0.249017 | 0.016928 | 0.579049 | 1.098582 | 52.71% |
| 46 | 0.097493 | 0.038768 | 0.003503 | 0.242571 | 0.382334 | 63.44% |
| 100 | 0.068061 | 0.028238 | 0.001744 | 0.169180 | 0.267224 | 63.31% |

边界 KD 的标量占比在首轮约 **78.95%**，在 D/vdt 最佳验证 epoch 46 仍约 **63.44%**，应优先单独检验该项的形式和权重。标量占比不等于梯度占比，也不能证明其导致最终测试退步。

100 轮 D/vdt 训练的 `gate_mean` 范围为 **0.999416435–0.999445073**，平均值始终接近 1，说明平均抑制幅度很小；需要门控分布或空间图才能判断局部筛选是否有效。教师校准温度记录为 4.0。

## Analysis

### 分数据集表现

- NC4K：D 的 Sm/MAE 最好，A 的 Fw/mean E 最好，C 的 Boundary_F 最好；D 相对 A 的 Sm 增益仅 +0.001307，尚无多种子证据判断其稳定性。
- COD10K：四项区域指标均为 A > B > D > C；D 相对 A 的 Fw 下降 0.020187。
- CAMO：四项区域指标均为 B > A > D > C；普通 VDT 预训练在这里有收益，不能概括为所有数据集均负迁移。
- CHAMELEON：C 的四项区域指标及 Boundary_F 均最好。该测试集仅 76 张，保留这一观察，但不将其外推为普遍收益。

### 本次组间差值

以下差值均为后者减去对照组的指标原值；Sm、Fw、mean E、Boundary_F 正值较好，MAE 负值较好。

#### D 相对 A

| Dataset | ΔSm | ΔFw | Δmean E | ΔMAE | ΔBoundary_F |
|---|---:|---:|---:|---:|---:|
| NC4K | +0.001307 | -0.003302 | -0.000704 | -0.000531 | +0.005821 |
| COD10K | -0.010486 | -0.020187 | -0.014838 | +0.001887 | -0.005336 |
| CAMO | -0.006205 | -0.012657 | -0.014921 | +0.002672 | -0.012988 |
| CHAMELEON | -0.006446 | -0.015783 | -0.001841 | -0.000336 | -0.015819 |

D 的 Fw、mean E 在四集全部低于 A；COD10K 和 CAMO 的四项区域指标全部退步。因此完整方法相对 RGB 基线的稳定收益尚未得到支持。

#### B 相对 A

| Dataset | ΔSm | ΔFw | Δmean E | ΔMAE | ΔBoundary_F |
|---|---:|---:|---:|---:|---:|
| NC4K | -0.002042 | -0.003798 | -0.005177 | +0.000086 | -0.007321 |
| COD10K | -0.007660 | -0.012129 | -0.011972 | +0.000588 | -0.012129 |
| CAMO | +0.003325 | +0.003023 | +0.003706 | -0.002036 | -0.013323 |
| CHAMELEON | -0.006192 | -0.013486 | -0.021210 | +0.001236 | -0.016237 |

B 仅在 CAMO 的四项区域指标均改善，另外三集整体退步。VDT 迁移方式值得先检查，但仅凭最终汇总不能确定负迁移的具体原因。

#### C 相对 B

| Dataset | ΔSm | ΔFw | Δmean E | ΔMAE | ΔBoundary_F |
|---|---:|---:|---:|---:|---:|
| NC4K | -0.001565 | -0.005182 | +0.000001 | +0.000555 | +0.017591 |
| COD10K | -0.005267 | -0.009525 | -0.003258 | +0.003167 | +0.023179 |
| CAMO | -0.022789 | -0.033367 | -0.027167 | +0.007596 | -0.003246 |
| CHAMELEON | +0.033908 | +0.054519 | +0.032204 | -0.010421 | +0.052031 |

C 在 CHAMELEON 明显改善，在 CAMO 明显退步；NC4K 的 mean E 差值约为 0.000000517，可视为基本持平，而 Sm/Fw/MAE 变差。logits KD 的效果具有数据集差异。

#### D 相对 B

| Dataset | ΔSm | ΔFw | Δmean E | ΔMAE | ΔBoundary_F |
|---|---:|---:|---:|---:|---:|
| NC4K | +0.003349 | +0.000496 | +0.004473 | -0.000617 | +0.013142 |
| COD10K | -0.002826 | -0.008057 | -0.002866 | +0.001299 | +0.006793 |
| CAMO | -0.009530 | -0.015680 | -0.018627 | +0.004708 | +0.000336 |
| CHAMELEON | -0.000254 | -0.002296 | +0.019369 | -0.001572 | +0.000418 |

D 在 NC4K 的四项区域指标均改善，但 COD10K 和 CAMO 均退步，CHAMELEON 有得有失。Boundary_F 四集均高于 B，但其中 CAMO 和 CHAMELEON 的增量很小，不能据此宣称完整 KD 全面有效。

#### D 相对 C

| Dataset | ΔSm | ΔFw | Δmean E | ΔMAE | ΔBoundary_F |
|---|---:|---:|---:|---:|---:|
| NC4K | +0.004914 | +0.005678 | +0.004472 | -0.001172 | -0.004449 |
| COD10K | +0.002441 | +0.001467 | +0.000392 | -0.001868 | -0.016386 |
| CAMO | +0.013259 | +0.017687 | +0.008540 | -0.002888 | +0.003581 |
| CHAMELEON | -0.034162 | -0.056815 | -0.012835 | +0.008849 | -0.051613 |

D 改善前三集的区域指标，却使 CHAMELEON 明显退步；Boundary_F 在三个数据集下降，Boundary_MAE 四集全部变差。C→D 同时改变关系、边界和门控，不能单独归因于某一项损失。

### Comparison with previous run

上一份可用报告为 [ResNet-50 环境检查](../results/resnet50_env_check_analysis.md)：每阶段 1 epoch、每个训练源最多 16 张、COD10K 测试仅 4 张，且正式指标汇总失败。本次是四个完整测试集的正式指标；数据规模、训练预算和指标口径不同，不计算跨实验的性能增幅，也不将环境检查的 B > D > A > C 排名视为需要保持的结论。本次作为后续全量实验的首条基线记录。

### Failure cases and observations

comparison 的训练历史显示后期 train SoftDice 约 0.94，而 COD validation 约 0.63–0.66，存在明显过拟合迹象。验证 SoftDice 排名 A > D > C > B 与 COD10K、CAMO 的正式测试排名不同；应检查原尺寸验证指标和来源权重，不能用测试集重新选 checkpoint。

当前实现还存在可独立检查的因素：[关系采样](../models/triprd.py)中邻近背景与外边界区域重复，每类默认取扫描顺序最前面的 8 个点；边界 KD 使用全图原始 logits MSE；当前 [训练器](../tools/run_triprd_ab.py)没有学习率调度。它们是后续消融对象，不是本次汇总指标已经证明的失败原因。

逐图预测和逐样本指标尚未同步，因此没有登记具体图像级失败案例。后续应从固定验证集导出对比图及逐图指标，再区分目标遗漏、误检和边界误差。

## Decision

保留本次结果作为基线；目前不接受“D 稳定优于 A/B”或“完整 KD 稳定改善边界”的结论。保留 A 作为区域分割基线、C 作为 logits KD 对照、B 作为额外 VDT 数据与训练预算对照。

下一轮按顺序进行，以下均为待实施计划：

1. 在固定验证划分上补原尺寸 Sm/Fw/mean E/MAE，分别报告 COD10K-val、CAMO-val；预先固定选模主指标与来源权重。
2. 增加已有教师/VDT 学生权重的复用入口，先以相同 COD 配方重跑 A′、B′、C′。优先验证编码器较小学习率及调度，再单独测试增强；不重复训练教师。
3. 若 B′仍落后 A′，比较完整权重迁移与仅迁移 encoder、重置 decoder/head。
4. 在统一配方下拆解 D：优先对比关闭边界 KD、降低边界 KD 权重与原设置，再单独检验关系和门控。依据本次边界损失占比与接近 1 的 gate_mean，补记各项梯度规模及门控分位数/空间图；关系区域/采样修正另作独立改动。
5. 只对胜出配置和必要基线做至少 3 个训练种子的复验。将 split_seed 与 train_seed 分开，报告均值和标准差，测试集仅用于最终报告。

每项改动先在预定验证指标上优于对应对照，且检查两个验证来源是否出现明显取舍；未确认收益则不扩展实验矩阵。单次汇总 JSON 不能提供训练方差或统计显著性。

## Artifacts

### 已保存的结果

- `results/triprd_full/comparison.json`（仅本地保存，不纳入 Git）：protocol、teacher/groups 元数据、8 阶段完整 history、16 条正式测试指标。
- `results/triprd_full/model_output.txt`（仅本地保存，不纳入 Git）：教师 epoch 41 尾部及完整 epoch 42–100 的终端 JSON 片段；59 条完整 history 与 comparison 对应项一致，不是完整运行日志。
- [A official_metrics.json](../results/triprd_full/A/official_metrics.json)
- [B official_metrics.json](../results/triprd_full/B/official_metrics.json)
- [C official_metrics.json](../results/triprd_full/C/official_metrics.json)
- [D official_metrics.json](../results/triprd_full/D/official_metrics.json)

| Source | SHA256 |
|---|---|
| comparison.json | `95757f85036674f1221967abf025aa54af7ebd32f1b0e23d521f7f99c105f500` |
| model_output.txt | `f671b416081ab869c1b8a81c50a86075561751377dc2ae78ae2c616618474419` |
| A/official_metrics.json | `7f5e5246d213e0952f5161f5798ba76dbb7c9dc57cf9edb4000dd066fbd14a93` |
| B/official_metrics.json | `ba8eed17f25720f8038c7193b478769c2bcb4d738512f7f520bacef9f78f2b53` |
| C/official_metrics.json | `ab2a212e19f07c281dc02a5ea936323e0c4e44011d4ecef4b333491edb7ad7e4` |
| D/official_metrics.json | `79782cf40b334e9e781634bdf73a8ee7d7b207e747aae5e929499df9abb4ca73` |

以上 SHA256 标识本次登记使用的文件内容，后续重算或更换文件时可复核。按归档决定，Git 仅保存本记录和四份独立评估 JSON；comparison 与终端原始输出保留在本地，指纹不表示这些文件已随仓库提交。

### 待从服务器回收的原始证据

| Artifact | 本地状态 | 用途 |
|---|---|---|
| run.log | 未同步；model_output.txt 仅为教师历史尾部片段 | 实际命令、运行起止时间、异常及完整输出 |
| split_manifest.json | 未同步 | 训练/验证划分及来源检查 |
| 各阶段 metrics.jsonl、history.json | 独立文件未同步；完整 history 已内嵌于 comparison | 补齐逐阶段归档，并与汇总交叉核验 |
| 各阶段 best.pt、last.pt | 权重文件未同步；教师及学生最终 best.pt 路径已有记录 | 权重、epoch、seed 与预测对应关系复核 |
| 各组 predictions 和逐图指标 | 未同步 | 预测覆盖、失败案例及配对分析 |

comparison 已补齐运行协议中的部分实参和全部 800 条 epoch 历史；正式指标、训练摘要与分项损失均已核验。完整日志、划分、权重和预测仍待归档。根目录原 `model_output.txt` 未保留；`results/model_output.txt` 是较早的调试及未完成训练记录，本次新增的 `results/triprd_full/model_output.txt` 也只是教师历史终端片段，两者均不能替代完整 run.log。
