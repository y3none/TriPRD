# Paper Scope

Updated: 2026-10-16. The minimal relation-transfer pipeline has completed its first matched A/B/C/D GPU screen: 100 COD epochs on the internal train/validation split, one seed. Multimodal D did not beat A or C on the predefined Sm criterion. Full training-pool retraining, final tests, stable gains, novelty, and SOTA status remain unverified. Future controls below are research plans unless marked implemented.

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

## Current implementation versus remaining design

| Item | Current implementation | Not yet implemented |
|---|---|---|
| Teacher | Local RGB-to-T/D relation aggregation using T/D values before decoding, with other scales still averaged; capacity-matched RGB-only control | Relation interventions; a proposed path where T/D affects only relation weights while values and decoder inputs remain RGB; additional scales or cross-modal reliability |
| Relation KD | New detached local relation-distribution KL; student-predicted relations used in decoding. Legacy fused-feature Gram KD remains separate | GT-only, shuffled-target, ordinary Gram-KD and training-only relation controls in the new pipeline |
| Boundary | Legacy GT morphological supervision and boundary KD; not part of the new relation objective | Cross-modal boundary reliability |
| Uncertainty | Legacy detached prediction-disagreement gate; not part of the new relation objective | Student uncertainty learning or feature/edge conflict modeling |
| COD adaptation | New relation-branch-only transfer with frozen source backbone/BN and independently initialized COD decoder; split selection and full-pool fixed-last-epoch modes | Explicit COD relation-prior loss, relation-preservation ablation |

E001–E010 used the legacy pipeline, where relation losses affect segmentation through shared feature gradients. They do not test the newly implemented explicit relation forward path. The independent [new experiment guide](tools/TRIPRD_RELATION.md) documents its different A/B/C/D definitions and commands; results must remain separated.

E011–E014 now archive the first screen of this explicit path; see the [relation experiment summary](experiments/relation_summary.md). With train/split seed 20261012, 3,232 COD training images and 808 validation images, source-equal Sm was A 0.808668, B 0.807764, C 0.807881, and D 0.807632. All selected epoch 70 from 100 epochs and skipped tests. The multimodal teacher improved VDT Dice over the RGB teacher (0.904641 versus 0.855075), but its source student and COD Sm did not improve over C. Small differences from one seed do not establish stable effects.

## Implemented minimal mechanism; effectiveness pending

The new pipeline isolates one relation-transfer mechanism before expanding boundary or uncertainty modules:

1. **Cross-modal relation teacher.** On aligned low-resolution grids, learn RGB-to-T and RGB-to-D pair relations. Learn projections jointly with a teacher decoder that uses these relations under source-task supervision. Direct dot products between independently trained, unaligned feature spaces do not establish meaningful relations.
2. **RGB relation predictor in the student.** Predict spatial relations from RGB features and use them in local residual aggregation before TriPRD's decoder. Distill detached teacher relation distributions on source images. Start with one scale and bounded local neighborhoods, not a full-resolution quadratic matrix.
3. **Transfer only the relation branch.** During VDT student training, freeze the ImageNet backbone, including BatchNorm running statistics, and train the relation branch plus a separate source head. At COD initialization, retain the same ImageNet backbone and identically initialized COD decoder/head as the control, and load only the relation branch. The backbone may subsequently be fine-tuned under the common COD recipe.

Transfer every learned projection/value parameter needed by the relation branch, with explicit checkpoint keys and metadata. Do not transfer the source segmentation head. Matched controls contain the same student relation branch so parameter count cannot explain their comparison.

This version uses COD RGB and COD labels to fine-tune the relation-aware student. It does not feed COD images into a real-T/D teacher or claim an additional COD prior loss. Relation-preservation regularization is a later ablation if forgetting is observed.

Local attention, residual aggregation, and KL distillation are established building blocks. The contribution to test is the content and utility of privileged cross-modal relations across the task/domain shift, not the invention of these operators.

## Experimental order

1. The first matched-student screen is completed: no VDT training; VDT RGB supervision; capacity-matched RGB-teacher relation KD; real RGB/T/D-teacher relation KD. Source-trained variants share images, source updates, source head supervision, target recipe, and initialization. Comparison metadata reports successful frozen-backbone/BN and branch-only transfer checks; actual checkpoint files and split membership still need independent verification.
2. Before extending this recipe, use saved teachers on the same VDT validation samples to compare learned relations, uniform relations, shuffled relations, and disabled relation residuals. Mean relation entropy near one motivates this check but does not prove all relations are ineffective. The intervention tooling is not implemented yet.
3. If interventions barely change segmentation, test a teacher in which T/D affects only relation weights while values and other decoder inputs remain RGB. Existing multimodal value features and mean-fused scales may bypass the distilled target; this is a hypothesis, not an established cause. This proposed information-path change is not implemented or evaluated.
4. If multimodal transfer has a net gain, add direct GT-derived relation supervision, ordinary fused-feature Gram KD, and spatially shuffled relation targets. These test mask re-encoding, generic KD, and relation-distribution regularization.
5. Separately retrain a teacher/student control with fixed incorrect T/D pairing. This differs from perturbing relations after teacher training. Record corruption, label handling, and source budget.
6. Compare relation use in decoding against training-only relation regularization and a matched-capacity ordinary module. Confirm an effective mechanism on another backbone and paired seeds; do not automatically scale up the unsuccessful first recipe.

Screen on the fixed internal split with equal validation opportunities. Freeze configuration, duration, learning-rate horizon, and checkpoint rule before full COD training-pool retraining and final evaluation. The new runner supports full 4,040-image COD training with fixed last-epoch weights and validates complete test prediction coverage; these paths still require real-data execution. Official COD-trained checkpoints must not be treated as clean baselines on internal validation images they have already seen.

## Falsification and reporting

- No advantage over a capacity-matched RGB teacher: do not claim extra value from T/D.
- No advantage over shuffled or GT-only relations: do not claim privileged spatial structure explains the gain.
- Good source relation prediction but no COD gain: investigate cross-task failure; do not conceal it by accumulating unrelated losses.
- Gains vanish on a stronger backbone or across seeds: narrow the claim and retain unfavorable results.
- Report the gain over the same student without TriPRD separately from absolute rankings against other methods.
- Report relation-branch inference cost despite removal of the multimodal teacher.
- SOTA, novelty relative to prior work, and CVPR readiness remain unverified until supported by experiments.

