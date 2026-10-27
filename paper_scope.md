# Paper Scope

Updated: 2026-10-27. [Completion v6](README_COMPLETION_V6.md) is the current implementation. Its first DINOv3 ViT-B/16 teacher/source/COD run and four-set official evaluation are complete. The [result analysis](experiments/completion_v6_result_analysis_2026-10-27.md) verifies report/code/provenance consistency, but does not establish a matched multimodal net benefit, stable gains, novelty or SOTA. Local trained checkpoints and prediction PNGs have not been received, so weights and metrics have not been independently recomputed.

## Method and objective

**TriPRD** remains the project working name.

The historical expansion **Tri-modal Privileged Relation Distillation** and its corresponding working paper title describe the earlier relation route. Completion v6 distills stage mask/boundary outputs, not relation matrices. No replacement paper title has been selected; see [method_name.md](method_name.md).

The objective is to reach competitive state-of-the-art COD performance through TriPRD's own verified contribution. Standard pretrained backbones may be used. Replacing the complete model with an existing COD method and adding ordinary KD is not the main method. Strong COD systems remain comparison methods and possible secondary hosts for testing generality; their components and results must be attributed.

**Explicit user constraint (2026-10-19):** E2Net is the team's separate work. This project must be an independent original method and must not be built on E2Net's method modules, decoder, COD weights, or complete implementation. E2Net is a comparison only. Public general-purpose pretrained encoders remain eligible; their use alone is not a contribution.

SOTA is a target, not a promised outcome. A stronger backbone, extra data, or renamed existing operators does not establish novelty.

## Fixed setting

- Teacher input: correctly paired real RGB, Thermal, and Depth from VDT2048 official train.
- Student input: RGB only, including at COD inference.
- COD training pool: COD10K train and CAMO train.
- COD test sets: COD10K test, CAMO test, NC4K, and CHAMELEON.
- Excluded: PST900, synthetic COD, pseudo modalities, and VDT2048 test in training or selection.
- Disclose VDT images/labels, teacher training, and backbone pretraining as training resources.

## Research question

Can a real RGB/T/D source teacher demonstrate object-conditioned mask completion under the same RGB cue as its student, and can the complete learned RGB completion path improve COD after supervised target adaptation?

The candidate contribution is the content and transfer of privileged completion behavior across the saliency-to-camouflage task shift. General pretrained features, standard KL, boundary supervision, foreground/background prototypes and coarse-to-fine decoding are not contributions by themselves. Implementation of this route does not establish its originality or empirical utility.

## Current completion v6 implementation

- Generic DINOv3 ViT-B/16 RGB features, projected multilevel features and a real stride4/8 RGB detail branch. The last four DINO blocks and final norm are trainable; earlier backbone parameters remain frozen. No E2Net components or COD checkpoint initialize this model.
- A real RGB/T/D teacher shares DINO encoder weights across modalities, uses separate T/D adapters and fuses modality features at each scale. Its coarse locator is RGB-only. The source RGB student starts from generic initialization and learns from its own GT supervision plus a frozen teacher.
- The student's predicted RGB cue, including any source-training perturbation, is supplied unchanged to the teacher. Soft foreground/background prototypes, pixel similarities and global context condition a full-image stride16/8/4 refiner; predicted masks and boundaries enter the next stage. There is no GT cue, hard ROI or test-time oracle prompt.
- KD matches three stages of mask and boundary outputs using teacher-correctness-weighted Bernoulli KL and a fixed pixel denominator. It does not match hidden fused features or relation matrices, use RTD-minus-R advantage gates, or reconstruct T/D.
- COD inherits all source-student state except the fresh COD locator, including the entire refiner and mask/boundary heads. Three locator-only epochs precede joint adaptation of the allowed RGB representation and completion path. Source cues stop gradients; generated COD cues allow final-mask gradients to reach the locator.
- The COD stage uses only COD RGB/GT. It has no live multimodal teacher, source batches or KD. Deployment is RGB-only and single-pass. This is supervised target adaptation, not zero-shot transfer or an established domain-invariant completion function.

## Completed v6 evidence and reporting boundary

The first run used VDT 838 train / 210 validation images for both teacher and source student, with 100/20 training epochs and selected source epochs 29/16. COD used COD10K3040 + CAMO1000, 384-pixel inputs and fixed final epoch 100, with no target validation or test checkpoint selection. One checkpoint was evaluated on all 6473 images across COD10K, CAMO, NC4K and CHAMELEON; see the [original benchmark](results/TriPRD_completion_v6/reports/cod/benchmark.md).

The report records 340 transferred tensors, an intact source refiner digest and an unchanged fresh COD locator. Source output KD converges, but this alone does not show that T/D information helps COD. In the frozen-completion warmup, improving the locator only slightly improves final-mask training loss; the large improvement appears after joint adaptation. This supports the need for target adaptation, not a causal claim that the transferred refiner outperforms a COD-only model.

All four Sm/Fw scores remain below the project's recorded E2Net reference. This is a performance reference, not a matched DINOv2-versus-DINOv3 experiment. No same-architecture v6 no-source or RGB-teacher formal result is currently available, so backbone, extra source data, generic KD and T/D contributions remain unseparated. Earlier v4/v5 internal-validation scores use different samples and checkpoint rules and are not direct v6 test baselines.

The student has about 90.94M parameters and the teacher about 95.17M. RGB-only inference removes auxiliary-modality processing, but a lightweight-model or speedup claim requires measured deployment cost. Report teacher/source training and VDT labels in addition to DINO pretraining resources.

Current claim boundaries:

- Do not claim extra T/D value without a matched RGB teacher comparison, or source-transfer value without a matched student lacking source training.
- Do not attribute gains over unrelated methods directly to the proposed transfer mechanism or to DINOv3 versus DINOv2.
- Do not describe v6 as relation distillation, signed relation correction, hidden-feature KD or relation-only transfer; these belong to historical variants.
- Report absolute benchmark performance separately from incremental mechanism gains, retain unfavorable results and establish variability before claiming stability.
- Follow the fixed development/full-training/test separation. Repeated test-guided structure selection cannot be presented as untouched final evaluation.

## Historical research and implementations

The sections below preserve earlier hypotheses, version-specific controls and pending diagnostics for reproduction. They are not the current v6 method or a claim that their experimental results transfer to it. [Task v4](README_TASK_V4.md) and [Staged v5](README_STAGED_V5.md) are historical routes with incompatible checkpoints.

As of 2026-10-19, relation_v1 and structure_v2 had completed matched single-seed screens without an established multimodal net benefit. In structure_v2, E exceeds A by only 0.000123 macro validation Sm and is worse on the other three macro metrics; see the [result analysis](experiments/structure_v2_analysis_2026-10-19.md). The [independent-method plan](literature/cvpr_independent_method_plan_2026-10-19.md) led to the [correction_v3 implementation](tools/TRIPRD_CORRECTION.md) and its CPU integration checks.

The earlier relation-stage research question was whether spatial relations learned from real RGB/T/D saliency observations improve RGB-only COD reasoning while separating source-task segmentation specialization from transferable spatial structure. Negative-transfer results motivated that hypothesis without establishing its cause. V6 changes the transfer scope by retaining the complete source RGB completion path.

## Archived correction v3 implementation

- One generic DINOv3 RGB backbone with trainable multi-level adapters, relation correction, and independent SOD/COD output heads; no E2Net components or COD checkpoint initialization. The default generic backbone is frozen; the shared RGB adapters and decoding path are trained jointly.
- One teacher shares the generic backbone across modalities and caches modality features for equally supervised R/RT/RD/RTD coalitions. T/D affect relations; propagated values remain RGB. Missing-modality tokens are trained, not introduced only at evaluation.
- GT-supported positive relation improvements define source correction reliability. RGB predicts signed probability corrections; ordinary full relation KD remains a matched control. No T/D reconstruction or COD pseudo-modality generation is used.
- B–F receive matched source RGB/GT supervision and source batches. E/F additionally gate their transfer losses using a detached positive cosine of transfer and COD training gradients on shared adapters. This is a batch-level approximation, not per-pixel causal selection or a generalization guarantee.
- Source-validation recovery is measured against zero and TRAIN-fitted spatially constant corrections. COD validation uses original-size source-macro Sm; full-pool training fixes the last epoch. Test data never supplies gradient selection.
- The 2026-10-20 staged optimization entry now checks a fixed-budget, source-only student probe before the matched A/B/C/D screen. Probe checkpoints never initialize formal students. Teacher relation supervision is equally weighted across the four coalitions, and student corrections start at zero. These implementation fixes do not establish an empirical performance improvement.
- A COD-only, B RGB joint, C full KD, D reliable correction, E selected correction, F selected full KD share student architecture and initialization. Post-selection correction removal and spatial permutation do not reselect checkpoints.

All algorithmic efficacy and novelty claims remain hypotheses. Generic pretraining, modality dropout, residual prediction and gradient reweighting have precedents; implementation alone does not establish an original contribution.

## Archived structure v2 candidate

The [structure guide](tools/TRIPRD_STRUCTURE.md) defines the new implementation and five-group protocol. It changes the knowledge target and the source-to-target interface:

- Independently supervised thermal-image color appearance, relative depth continuity, and annotation compatibility at ResNet strides 8/16. Thermal images are pseudo-color observations, not calibrated temperatures; binary annotation compatibility is not instance identity.
- Teacher modalities affect edges only. Propagated values and other segmentation inputs remain RGB, with no direct mean-fused modality bypass.
- A separate frozen ImageNet reference encoder and its source-trained edge heads form the complete transferred function. The trainable COD encoder, semantic graph, propagation and decoder are separately initialized; the source function stays fixed during COD training. Unused final stages of reference/auxiliary encoders are removed.
- COD has its own supervised semantic edges and context-dependent selection, including identity. Source-only interventions distinguish fixed knowledge from the graph machinery trained on COD.
- New A/B/C/D/E separate no source, mask-only source, direct observed-edge supervision, RGB-teacher refinement with observed labels, and matched RGB/T/D teacher inputs. D already uses privileged labels. These definitions do not replace or relabel the archived v1 groups.

The reference path increases deployment cost. Absolute improvement over the older smaller model and the incremental contribution of privileged information must be reported separately. Depth target scaling and preservation of pseudo-color thermal information have been checked on real VDT Train observations, but their usefulness for COD remains a hypothesis.

## Archived relation v1 implementation and remaining diagnostics

| Item | Current implementation | Not yet implemented |
|---|---|---|
| Teacher | Local RGB-to-T/D relation aggregation using T/D values before decoding, with other scales still averaged; capacity-matched RGB-only control | Relation interventions; a proposed path where T/D affects only relation weights while values and decoder inputs remain RGB; additional scales or cross-modal reliability |
| Relation KD | New detached local relation-distribution KL; student-predicted relations used in decoding. Legacy fused-feature Gram KD remains separate | GT-only, shuffled-target, ordinary Gram-KD and training-only relation controls in the new pipeline |
| Boundary | Legacy GT morphological supervision and boundary KD; not part of the new relation objective | Cross-modal boundary reliability |
| Uncertainty | Legacy detached prediction-disagreement gate; not part of the new relation objective | Student uncertainty learning or feature/edge conflict modeling |
| COD adaptation | New relation-branch-only transfer with frozen source backbone/BN and independently initialized COD decoder; split selection and full-pool fixed-last-epoch modes | Explicit COD relation-prior loss, relation-preservation ablation |

E001–E010 used the legacy pipeline, where relation losses affect segmentation through shared feature gradients. They do not test the newly implemented explicit relation forward path. The independent [new experiment guide](tools/TRIPRD_RELATION.md) documents its different A/B/C/D definitions and commands; results must remain separated.

E011–E014 now archive the first screen of this explicit path; see the [relation experiment summary](experiments/relation_summary.md). With train/split seed 20261012, 3,232 COD training images and 808 validation images, source-equal Sm was A 0.808668, B 0.807764, C 0.807881, and D 0.807632. All selected epoch 70 from 100 epochs and skipped tests. The multimodal teacher improved VDT Dice over the RGB teacher (0.904641 versus 0.855075), but its source student and COD Sm did not improve over C. Small differences from one seed do not establish stable effects.

## Archived relation v1 mechanism

The new pipeline isolates one relation-transfer mechanism before expanding boundary or uncertainty modules:

1. **Cross-modal relation teacher.** On aligned low-resolution grids, learn RGB-to-T and RGB-to-D pair relations. Learn projections jointly with a teacher decoder that uses these relations under source-task supervision. Direct dot products between independently trained, unaligned feature spaces do not establish meaningful relations.
2. **RGB relation predictor in the student.** Predict spatial relations from RGB features and use them in local residual aggregation before TriPRD's decoder. Distill detached teacher relation distributions on source images. Start with one scale and bounded local neighborhoods, not a full-resolution quadratic matrix.
3. **Transfer only the relation branch.** During VDT student training, freeze the ImageNet backbone, including BatchNorm running statistics, and train the relation branch plus a separate source head. At COD initialization, retain the same ImageNet backbone and identically initialized COD decoder/head as the control, and load only the relation branch. The backbone may subsequently be fine-tuned under the common COD recipe.

Transfer every learned projection/value parameter needed by the relation branch, with explicit checkpoint keys and metadata. Do not transfer the source segmentation head. Matched controls contain the same student relation branch so parameter count cannot explain their comparison.

This version uses COD RGB and COD labels to fine-tune the relation-aware student. It does not feed COD images into a real-T/D teacher or claim an additional COD prior loss. Relation-preservation regularization is a later ablation if forgetting is observed.

Local attention, residual aggregation, and KL distillation are established building blocks. The contribution to test is the content and utility of privileged cross-modal relations across the task/domain shift, not the invention of these operators.

## Archived relation v1 diagnostic order

1. The first matched-student screen is completed: no VDT training; VDT RGB supervision; capacity-matched RGB-teacher relation KD; real RGB/T/D-teacher relation KD. Source-trained variants share images, source updates, source head supervision, target recipe, and initialization. Comparison metadata reports successful frozen-backbone/BN and branch-only transfer checks; actual checkpoint files and split membership still need independent verification.
2. Before extending this recipe, use saved teachers on the same VDT validation samples to compare learned relations, uniform relations, shuffled relations, and disabled relation residuals. Mean relation entropy near one motivates this check but does not prove all relations are ineffective. The intervention tooling is not implemented yet.
3. If interventions barely change segmentation, test a teacher in which T/D affects only relation weights while values and other decoder inputs remain RGB. Existing multimodal value features and mean-fused scales may bypass the distilled target; this is a hypothesis, not an established cause. This proposed information-path change is not implemented or evaluated.
4. If multimodal transfer has a net gain, add direct GT-derived relation supervision, ordinary fused-feature Gram KD, and spatially shuffled relation targets. These test mask re-encoding, generic KD, and relation-distribution regularization.
5. Separately retrain a teacher/student control with fixed incorrect T/D pairing. This differs from perturbing relations after teacher training. Record corruption, label handling, and source budget.
6. Compare relation use in decoding against training-only relation regularization and a matched-capacity ordinary module. Confirm an effective mechanism on another backbone and paired seeds; do not automatically scale up the unsuccessful first recipe.

Screen on the fixed internal split with equal validation opportunities. Freeze configuration, duration, learning-rate horizon, and checkpoint rule before full COD training-pool retraining and final evaluation. The new runner supports full 4,040-image COD training with fixed last-epoch weights and validates complete test prediction coverage; these paths still require real-data execution. Official COD-trained checkpoints must not be treated as clean baselines on internal validation images they have already seen.

## Archived relation-stage falsification and reporting

- No advantage over a capacity-matched RGB teacher: do not claim extra value from T/D.
- No advantage over shuffled or GT-only relations: do not claim privileged spatial structure explains the gain.
- Good source relation prediction but no COD gain: investigate cross-task failure; do not conceal it by accumulating unrelated losses.
- Gains vanish on a stronger backbone or across seeds: narrow the claim and retain unfavorable results.
- Report the gain over the same student without TriPRD separately from absolute rankings against other methods.
- Report relation-branch inference cost despite removal of the multimodal teacher.
- SOTA, novelty relative to prior work, and CVPR readiness remain unverified until supported by experiments.

