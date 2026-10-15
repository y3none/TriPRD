# TriPRD

TriPRD is a tri-modal teacher and RGB student framework for camouflaged object detection. The teacher uses RGB, thermal, and depth inputs during training; the student uses RGB only for deployment.

## Implementation baseline

The initial A/B/C/D pipeline was introduced in commit `75adf91` (`Add TriPRD A/B/C/D experiment pipeline and results`). It includes the ResNet-50 comparable model, the controlled A/B/C/D runner, training and evaluation utilities, tests, configuration files, and the environment-check report in [`results/resnet50_env_check_analysis.md`](results/resnet50_env_check_analysis.md).

The smoke experiment used ImageNet-pretrained ResNet-50 with one epoch, at most 16 training samples, batch size 1, 352×352 inputs, and four COD10K test images. Its results were:

| Group | COD10K Dice | COD10K MAE |
|---|---:|---:|
| A | 0.3845 | 0.4607 |
| B | **0.4116** | **0.4434** |
| C | 0.3798 | 0.5191 |
| D | 0.3911 | 0.4451 |

The observed order was B > D > A > C. These numbers are an environment and pipeline check, not a final benchmark: the sample count and training budget are intentionally small. The official-metric aggregation bug found in that run has been fixed in the current code.

## A/B/C/D protocol

- **A — COD RGB supervised baseline:** train the RGB student directly on COD data.
- **B — RGB supervised pretraining:** pretrain the RGB student on VDT RGB masks, then fine-tune it on COD.
- **C — Logit distillation:** train a tri-modal RGB/thermal/depth teacher, distill its output logits into the RGB student on VDT, then fine-tune on COD.
- **D — Full distillation:** use logit, feature-relation, and boundary distillation with an uncertainty gate derived from teacher modality disagreement, then fine-tune on COD.

## Quick start

Install the dependencies listed in [`requirements-triprd.txt`](requirements-triprd.txt), then run the controlled experiment with:

```bash
python tools/run_triprd_ab.py --model-variant resnet50 --pretrained \
  --groups A,B,C,D --epochs 5 --vdt-epochs 5 \
  --image-size 352 --base-channels 256 --device cuda
```

Use the dataset-root options when the datasets are stored outside the default `datasets/` directory. See [`tools/TRIPRD_QUICKSTART.md`](tools/TRIPRD_QUICKSTART.md) for the full protocol and evaluation commands.

For the full paper-comparable Windows run, use:

```powershell
.\tools\run_triprd_full.ps1
```

On Ubuntu/Linux, use the Bash wrapper:

```bash
CUDA_VISIBLE_DEVICES=3 bash tools/run_triprd_full.sh
```

For a repeatable record of every experiment or tuning change, see
[`experiments/README.md`](experiments/README.md). After a run, create its
record with `python tools/record_experiment.py`; the command writes a Markdown
record and updates `experiments/index.md` without changing the raw run output.

## Training and evaluation tools

The experiment tooling includes the following improvements:

- Fixed official adaptive F-measure aggregation by reading `fm["adp"]` from the result returned by the evaluation module.
- Routed prediction resizing and COD metrics through `evaluation/cod_metrics.py`, which wraps the vendored CamoFormer metric implementation.
- Added `tools/run_triprd_full.ps1` for the full Windows experiment and `tools/run_triprd_full.sh` for Ubuntu/Linux.
- Avoided Bash's reserved `GROUPS` variable by using `TRIPRD_GROUPS` for A/B/C/D selection.
- Added Bash `run.log` capture, unbuffered Python output, and exit-status propagation; the runner prints compact one-line epoch summaries on both platforms.
- Added validation Dice, best epoch, stage ETA, checkpoint, and test-metric summaries to the terminal while keeping detailed histories in `metrics.jsonl`, `history.json`, and `comparison.json`.
- Added experiment records with separate official and quick metrics, explicit evaluation failures, and protection against duplicate or invalid experiment IDs.

## Full experiment results

The first full A/B/C/D experiment is registered as
[`E001_triprd_full`](experiments/E001_triprd_full.md) on **2026-10-15 (Asia/Shanghai)**.
The [experiment index](experiments/index.md) tracks subsequent runs.
The record incorporates the locally verified `results/triprd_full/comparison.json`:
run metadata, all eight stage histories and official test results. The raw
comparison file and terminal log are excluded from Git; their SHA256 hashes
are retained in the experiment record.
All 16 official metric records were cross-checked against the offline evaluation files:
[A](results/triprd_full/A/official_metrics.json),
[B](results/triprd_full/B/official_metrics.json),
[C](results/triprd_full/C/official_metrics.json), and
[D](results/triprd_full/D/official_metrics.json).

**Protocol and provenance**

- Output directory: `results/triprd_full`.
- Test sets: NC4K 4,121; COD10K 2,026; CAMO 250; CHAMELEON 76.
  Each group reports 6,473 samples.
- Recorded protocol: ImageNet-pretrained ResNet-50, 352×352 input,
  256 decoder channels, seed `20261012`, ImageNet normalization and
  horizontal flipping with probability 0.5.
- All eight stages contain continuous histories for epochs 1–100
  (800 epoch records). Each recorded best validation Dice matches its
  history maximum.
- Recorded parameters: teacher 75,043,781; each student 28,034,114.
  Teacher calibration temperature: 4.0. Server paths to the teacher and
  final student `best.pt` files are recorded, but weights are not synced.
- Batch size 8, AdamW with learning rate and weight decay `1e-4`, no
  scheduler, and validation fraction 0.2 are current code defaults;
  comparison does not record these run settings. The actual split still
  needs `split_manifest.json`.
- Training sample limits, dataset roots and actual train/validation counts
  are not recorded. The 100-epoch histories alone do not establish that all
  training samples were used.
- GPU, runtime/code versions, exact timestamps and the actual launch command
  remain unverified. The local `results/triprd_full/model_output.txt`
  contains only a terminal fragment of teacher history, not a complete run log.
  Standalone stage history files, checkpoints and predictions are still absent.
- Official results are read from `test.<group>.<dataset>.official`.
  The comparison uses a shared `TriPRD-AB` model label; use the outer group
  key to distinguish A/B/C/D. Metrics match the independent evaluation files.

**Test results**

`S_alpha` is Sm, `F_beta_w` is weighted F-measure, and `E_phi` is mean
E-measure across thresholds, not adaptive or maximum E-measure.
`Boundary_F` is the evaluator's weighted F-measure within the GT boundary
region, not a generic contour-matching boundary F1. MAE is the official
evaluator value (`official.MAE`), separate from the outer quick-statistic
`mae`. Values are rounded to six decimal places.

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

**Analysis**

- **NC4K:** D leads Sm and MAE; A leads Fw and mean E; C leads Boundary_F.
  D's Sm gain over A is only `+0.001307`.
- **COD10K:** A > B > D > C on all four regional metrics (Sm, Fw, mean E, MAE).
- **CAMO:** B > A > D > C on those four regional metrics.
- **CHAMELEON:** C leads all four regional metrics and Boundary_F.
  This test set contains only 76 images.
- **B versus A:** VDT RGB pretraining improves the four regional metrics on
  CAMO but regresses overall on the other three datasets.
- **C versus B:** CHAMELEON improves substantially (Sm `+0.033908`,
  Fw `+0.054519`), while CAMO regresses (Sm `-0.022789`,
  Fw `-0.033367`). Distillation does not provide a uniform benefit.
- **D versus B:** D improves all four regional metrics on NC4K, regresses on
  COD10K/CAMO, and has mixed results on CHAMELEON. Boundary_F is higher on
  all four sets, but this is not a general regional-performance gain.
- **D versus A:** Fw and mean E are lower on every dataset; the COD10K
  differences are Sm `-0.010486` and Fw `-0.020187`.
- **D versus C:** regional metrics improve on NC4K/COD10K/CAMO but decline
  on CHAMELEON. Boundary_F declines on three datasets, and Boundary_MAE
  increases on all four. Relation, boundary and gating effects need separate
  ablations before assigning a cause.

The following results are recalculated from all eight stage histories in
comparison. SoftDice is reported separately from official test metrics.

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

Recorded epoch durations sum to **20.75 hours**; this is not the total runtime
including independent evaluation and other work.

The D/VDT loss history also adds two observations:

- Using the current loss formula, the weighted boundary KD term accounts for
  **78.95%** of total scalar loss at epoch 1 and **63.44%** at its best
  validation epoch 46. These proportions do not establish gradient dominance
  or explain test regressions on their own.
- Training `gate_mean` stays within **0.999416435–0.999445073**, close to 1.
  Its average suppression is small; spatial maps or distributions are needed
  to assess local filtering.

The validation SoftDice order A > D > C > B differs from the test rankings.
Late training also showed a large train/validation gap. Image-level failure
cases have not been inspected because predictions are not available locally.
The earlier one-epoch smoke check used different samples and metrics, so no
performance delta against it is claimed.

**Decision and next experiment**

Keep this run as the full-run baseline. The results do not establish a stable
advantage for full distillation over A or B. Preserve A as the regional baseline,
B as the VDT data/budget control, and C as the logit-distillation control.

Next, add original-size validation metrics and a fixed checkpoint-selection
rule, then reuse VDT weights to compare A′/B′/C′ under a common improved COD
fine-tuning recipe. Test encoder-only transfer if full transfer remains harmful.
After that, isolate D's boundary, relation and gating contributions, starting
with boundary KD removal and weight reduction given the observed loss scale.
Record gradient scales and gate distributions, then verify the winning
configuration with at least three training seeds while keeping the data split
fixed. These changes are planned, not yet implemented.

The [experiment record](experiments/E001_triprd_full.md) contains the complete group
deltas, training summary, evidence inventory and decision criteria. A single
run of aggregate metrics does not establish statistical significance.
