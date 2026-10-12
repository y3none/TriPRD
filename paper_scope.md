# Paper Scope

## Method name`n`n**TriPRD** — Tri-modal Privileged Relation Distillation`n`n## Working title

**TriPRD: Tri-modal Privileged Relation Distillation for RGB-only Camouflaged Object Detection**

## Research question

Can real RGB-T-D observations from VDT2048 teach a student that uses RGB only at inference time to preserve target-background structure, boundary reliability, and conflict-aware uncertainty for real camouflaged object detection?

## Fixed setting

- Teacher input: RGB, Thermal, and Depth from VDT2048.
- Student input: RGB only.
- Multimodal source: VDT2048 official train split.
- COD training: COD10K train and CAMO train.
- COD evaluation: COD10K test, CAMO test, NC4K, and CHAMELEON.
- Excluded: PST900, synthetic COD, pseudo modalities, and VDT2048 test split.

## Method components

1. Tri-modal relation teacher.
2. Relation distillation for target-internal and target-background structure.
3. Boundary reliability distillation.
4. Conflict-aware uncertainty gating.
5. RGB-only adaptation to real COD.

## Decisive experiments

- RGB teacher versus RGB-T-D relation teacher.
- Logit KD versus relation KD.
- Correct RGB/T/D pairing versus random pairing.
- Full model versus removal of relation, boundary, and uncertainty losses.
- Full model versus same-data RGB-only pretraining.

## Core claim to test

A real RGB-T-D teacher can transfer useful structural and reliability priors to an RGB-only COD student, producing consistent gains on at least three of the four real COD test sets while keeping RGB-only inference.

## Falsification conditions

The paper direction must be revised if the tri-modal teacher does not beat the RGB teacher, relation KD does not beat logit KD, correct pairing does not beat random pairing, or the full student does not improve real COD results.

