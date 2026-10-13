# TriPRD

TriPRD is a tri-modal teacher and RGB student framework for camouflaged object detection. The teacher uses RGB, thermal, and depth inputs during training; the student uses RGB only for deployment.

## Latest submission

The current implementation was submitted in commit `75adf91` (`Add TriPRD A/B/C/D experiment pipeline and results`). It adds the ResNet-50 comparable model, the controlled A/B/C/D runner, training and evaluation utilities, tests, configuration files, and the environment-check report in [`results/resnet50_env_check_analysis.md`](results/resnet50_env_check_analysis.md).

The smoke experiment used ImageNet-pretrained ResNet-50 with one epoch, at most 16 training samples, batch size 1, 352×352 inputs, and four COD10K test images. Its results were:

| Group | COD10K Dice | COD10K MAE |
|---|---:|---:|
| A | 0.3845 | 0.4607 |
| B | **0.4116** | **0.4434** |
| C | 0.3798 | 0.5191 |
| D | 0.3911 | 0.4451 |

The observed order was B > D > A > C. These numbers are an environment and pipeline check, not a final benchmark: the sample count and training budget are intentionally small. The report also records the known official-metric aggregation issue (`'adp'`) encountered in this run.

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
