# Paper Scope

Updated: 2026-10-16. This is a research plan, not a claim that the proposed mechanisms are implemented or effective.

## Method and objective

**TriPRD — Tri-modal Privileged Relation Distillation**

Working title: **TriPRD: Tri-modal Privileged Relation Distillation for RGB-only Camouflaged Object Detection**.

The objective is to reach competitive state-of-the-art COD performance through TriPRD's own verified contribution. Standard pretrained backbones may be used. Replacing the complete model with an existing COD method and adding ordinary KD is not the main method. Strong COD systems remain comparison methods and possible secondary hosts for testing generality; their components and results must be attributed.

SOTA is a target, not a promised outcome. A stronger backbone, extra data, or renamed existing operators does not establish novelty.

## Fixed setting

- Teacher input: correctly paired real RGB, Thermal, and Depth from VDT2048 official train.
- Student input: RGB only, including at COD inference.
- COD training pool: COD10K train and CAMO train.
- COD test sets: COD10K test, CAMO test, NC4K, and CHAMELEON.
- Excluded: PST900, synthetic COD, pseudo modalities, and VDT2048 test in training or selection.
- Disclose VDT images/labels, teacher training, and backbone pretraining as training resources.

## Research question

Can spatial relations learned from real RGB/T/D saliency observations improve an RGB-only COD student's target/background reasoning across the task/domain shift, without replacing its general pretrained representation with a source-task segmentation model?

The hypothesis is that source-task segmentation specialization and transferable spatial structure should be separated during transfer. Existing negative-transfer results motivate this hypothesis; they do not establish its cause.

## Current implementation versus design

| Item | Current implementation | Not yet implemented |
|---|---|---|
| Teacher | Three ResNet-50 branches, feature averaging, segmentation/modality heads | Explicit RGB-to-T/D relations used by the teacher decoder |
| Relation KD | Self-similarity Gram matrices of decoded features; uniform sampling includes cross-region pairs | Direct cross-modal relation targets and student relation prediction used in decoding |
| Boundary | GT-derived morphological boundary supervision and boundary KD | Cross-modal boundary reliability |
| Uncertainty | Detached modality-prediction disagreement gate with temperature calibration | Student uncertainty learning or feature/edge conflict modeling |
| COD adaptation | Full/encoder initialization, then supervised COD training | Relation-branch-only transfer or explicit COD relation-prior loss |

The existing relation loss affects segmentation through shared feature gradients. What is missing is an explicit predicted relation participating in the segmentation forward path. Existing experiments therefore do not test every component in the original design sketch.

## Next minimal mechanism

Develop one relation-transfer mechanism before expanding boundary or uncertainty modules:

1. **Cross-modal relation teacher.** On aligned low-resolution grids, learn RGB-to-T and RGB-to-D pair relations. Learn projections jointly with a teacher decoder that uses these relations under source-task supervision. Direct dot products between independently trained, unaligned feature spaces do not establish meaningful relations.
2. **RGB relation predictor in the student.** Predict spatial relations from RGB features and use them in local residual aggregation before TriPRD's decoder. Distill detached teacher relation distributions on source images. Start with one scale and bounded local neighborhoods, not a full-resolution quadratic matrix.
3. **Transfer only the relation branch.** During VDT student training, freeze the ImageNet backbone, including BatchNorm running statistics, and train the relation branch plus a separate source head. At COD initialization, retain the same ImageNet backbone and identically initialized COD decoder/head as the control, and load only the relation branch. The backbone may subsequently be fine-tuned under the common COD recipe.

Transfer every learned projection/value parameter needed by the relation branch, with explicit checkpoint keys and metadata. Do not transfer the source segmentation head. Matched controls contain the same student relation branch so parameter count cannot explain their comparison.

This version uses COD RGB and COD labels to fine-tune the relation-aware student. It does not feed COD images into a real-T/D teacher or claim an additional COD prior loss. Relation-preservation regularization is a later ablation if forgetting is observed.

Local attention, residual aggregation, and KL distillation are established building blocks. The contribution to test is the content and utility of privileged cross-modal relations across the task/domain shift, not the invention of these operators.

## Experimental order

1. Verify relation forward/backward behavior, source pairing, frozen backbone state, and exact transfer keys. Measure GPU memory/time before full runs.
2. First matched-student screen: no VDT training; VDT RGB supervision; capacity-matched RGB-teacher relation KD; real RGB/T/D-teacher relation KD. Source-trained variants share images, source updates, source head supervision, target recipe, and initialization.
3. If multimodal transfer has a net gain, add direct GT-derived relation supervision, ordinary fused-feature Gram KD, and spatially shuffled relation targets. These test mask re-encoding, generic KD, and relation-distribution regularization.
4. Separately retrain a teacher/student control with fixed incorrect T/D pairing. This differs from shuffling relation targets after teacher training. Record corruption, label handling, and source budget.
5. Compare relation use in decoding against training-only relation regularization and a matched-capacity ordinary module. Confirm the effective mechanism on another backbone and paired seeds.

Screen on the fixed internal split with equal validation opportunities. Freeze configuration, duration, learning-rate horizon, and checkpoint rule before full COD training-pool retraining and final evaluation. Full-train/fixed-budget support remains an implementation task. Official COD-trained checkpoints must not be treated as clean baselines on internal validation images they have already seen.

## Falsification and reporting

- No advantage over a capacity-matched RGB teacher: do not claim extra value from T/D.
- No advantage over shuffled or GT-only relations: do not claim privileged spatial structure explains the gain.
- Good source relation prediction but no COD gain: investigate cross-task failure; do not conceal it by accumulating unrelated losses.
- Gains vanish on a stronger backbone or across seeds: narrow the claim and retain unfavorable results.
- Report the gain over the same student without TriPRD separately from absolute rankings against other methods.
- Report relation-branch inference cost despite removal of the multimodal teacher.
- SOTA, novelty relative to prior work, and CVPR readiness remain unverified until supported by experiments.

