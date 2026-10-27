# Method Name

Updated: 2026-10-27. The current implementation is [Completion v6](README_COMPLETION_V6.md).

## Project working name

**TriPRD**

`TriPRD` remains the repository and experiment working name. It is not a claim that every historical mechanism is present in the current implementation.

## Historical expanded name

**Tri-modal Privileged Relation Distillation**

This expansion described the earlier relation-distillation route. Completion v6 instead distills three-stage mask and boundary outputs under a common RGB cue, then transfers the complete RGB completion path to COD. It has no relation-matrix distillation loss. The historical expansion must not be used as v6's mechanism description.

## Task-specific project name

**TriPRD-COD**

`COD` identifies the application task: camouflaged object detection. The repository and experiment assets continue to use `TriPRD-COD` or `TriPRD` in filenames so historical records remain traceable.

## Historical working paper title

**TriPRD: Tri-modal Privileged Relation Distillation for RGB-only Camouflaged Object Detection**

This title belongs to the earlier proposal. No replacement paper title or acronym expansion has been selected. Describe the current mechanism as object-conditioned completion transfer with a privileged RGB/T/D source teacher until a final title is agreed.

## Historical name rationale

- **Tri-modal**: the teacher uses RGB, Thermal, and Depth.
- **Privileged**: Thermal and Depth are available during training only.
- **Relation Distillation**: the student receives target-internal, target-background, and boundary relation knowledge.

The first two terms still match v6's training setting. The third is historical: v6 uses output KL on masks and boundaries, not explicit target/background relation distributions. Implementing this new transfer route does not by itself establish a novel or effective contribution; see [Paper Scope](paper_scope.md).

## Filename convention

Use `TriPRD` for method-level assets and `TriPRD-COD` for task-level outputs. Examples:

- `TriPRD_E13_relation_teacher_seed0.pt`
- `TriPRD_E26_full_student_seed0.yaml`
- `TriPRD-COD_COD10K_main_results.csv`
