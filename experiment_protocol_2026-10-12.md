# CVPR 2027 实验协议（2026-10-12 冻结版）

## 研究主线

真实 VDT2048 的 RGB-T-D 关系、边界和不确定性知识，蒸馏到只使用 RGB 的真实 COD 学生模型。

## 数据范围

### 多模态知识源

- VDT2048 官方 train split；
- RGB、Thermal、Depth、mask；
- 仅用于训练教师、训练学生和提取关系先验。

### COD 数据

- 训练：COD10K train、CAMO train；
- 测试：COD10K test、CAMO test、NC4K、CHAMELEON；
- COD 阶段只使用 RGB 和 mask。

### 禁止事项

- 不使用 PST900；
- 不使用合成 COD、伪 T/D；
- 不使用 VDT2048 test split；
- 不使用 COD 测试集图像调参或生成教师信号；
- 不跨官方场景或序列划分数据。

## 固定模型范围

- 教师：RGB、Thermal、Depth 三分支 + cross-modal relation module + mask/boundary/uncertainty heads；
- 学生：RGB encoder + relation/boundary/uncertainty heads + COD decoder；
- 最终推理只输入 RGB。

## 今日完成项

- [x] 创建实验目录结构；
- [x] 创建基础配置文件；
- [x] 创建实验结果记录表；
- [x] 冻结数据使用协议；
- [x] 建立明日数据对齐检查的入口。

## 后续日程

- 10/13：VDT2048 读取、RGB/T/D/mask 对齐检查和四联图；
- 10/14：COD10K、CAMO、NC4K、CHAMELEON 读取；
- 10/15：统一评价指标脚本；
- 10/16：E00 RGB baseline 训练。

