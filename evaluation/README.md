# COD evaluation

The metric implementation is the official CamoFormer `evaltools/metrics.py`
vendored unchanged at [`camoformer_metrics.py`](camoformer_metrics.py). The
project entry point at [`tools/evaluate_cod.py`](../tools/evaluate_cod.py)
keeps those metric classes and reads GT masks from the project's COD ZIP files.

Install the evaluation dependencies in the experiment environment:

```powershell
pip install numpy scipy opencv-python tqdm pillow
```

Predictions keep one directory per test dataset:

```text
Prediction/CamoFormer/COD10K/0001.png
Prediction/CamoFormer/CAMO/0001.png
Prediction/CamoFormer/NC4K/0001.png
Prediction/CamoFormer/CHAMELEON/0001.png
```

Prediction filenames must match the GT filename stem in the corresponding
dataset archive. The evaluator uses GT as the spatial reference and resizes
the prediction in memory with CamoFormer's default OpenCV linear interpolation;
it does not rewrite prediction files.

Run all four COD test sets and append one row per dataset to the experiment
registry:

```powershell
python tools/evaluate_cod.py `
  --data-root datasets/COD_benchmark `
  --pred-root Prediction/CamoFormer `
  --model CamoFormer `
  --BR on `
  --br-rate 15 `
  --experiment-id E05-CamoFormer `
  --record-csv results/experiment_records.csv
```

For a small smoke run, add `--dataset COD10K --max-samples 20 --no-record`.
The JSON printed to the terminal contains `S_alpha`, `F_beta_w`, `E_phi`,
`MAE`, and `Boundary_F`; the CSV receives the same five fields.

The upstream CamoFormer source is available at:
<https://github.com/HVision-NKU/CamoFormer/tree/main/evaltools>.
