# Method Name

## Method name

**TriPRD**

## Expanded name

**Tri-modal Privileged Relation Distillation**

## Task-specific project name

**TriPRD-COD**

`TriPRD` names the method mechanism. `COD` identifies the application task: camouflaged object detection. The repository and experiment assets use `TriPRD-COD` or `TriPRD` in filenames.

## Paper title

**TriPRD: Tri-modal Privileged Relation Distillation for RGB-only Camouflaged Object Detection**

## Name rationale

- **Tri-modal**: the teacher uses RGB, Thermal, and Depth.
- **Privileged**: Thermal and Depth are available during training only.
- **Relation Distillation**: the student receives target-internal, target-background, and boundary relation knowledge.

## Filename convention

Use `TriPRD` for method-level assets and `TriPRD-COD` for task-level outputs. Examples:

- `TriPRD_E13_relation_teacher_seed0.pt`
- `TriPRD_E26_full_student_seed0.yaml`
- `TriPRD-COD_COD10K_main_results.csv`
