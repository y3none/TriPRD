# CVPR 2027 TriPRD

This folder contains the complete experiment workspace for the RGB-only camouflaged object detection paper.

## Research assets

- `method_name.md`: official method name and filename convention`n- `paper_scope.md`: paper title, claim, method scope, and decisive experiments
- `cvpr2027_submission_plan.md`: schedule, milestones, and submission gates
- `research_plan.md`: full method and experiment design
- `experiment_protocol_2026-10-12.md`: frozen data and experiment protocol
- `configs/`: data, teacher, student, and COD fine-tuning configurations
- `datasets/`: dataset loaders and alignment checks
- `evaluation/`: unified COD metrics
- `tools/`: inspection and visualization utilities
- `results/`: experiment registry and result tables
- `checkpoints/`: saved model checkpoints
- `logs/`: training logs
- `visualizations/`: qualitative figures and reports

## Current status

The initial workspace, configurations, result schema, and data boundaries are frozen. The next task is VDT2048 loading, RGB/T/D/mask alignment validation, and four-panel visualization.

