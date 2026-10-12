# 实验结果记录

`experiment_records.csv` 是所有主表和消融表的唯一结果入口。每次训练或测试完成后追加一行，不手工修改已记录的数字。

记录要求：

- 一个实验对应一个 `experiment_id`；
- 明确 dataset、split、seed、配置文件和 checkpoint；
- 指标统一记录 `S_alpha`、`F_beta_w`、`E_phi`、`MAE`、`Boundary_F`；
- 主结论至少使用 3 个随机种子；
- 配置、日志、checkpoint 和 CSV 行必须能相互追溯。

