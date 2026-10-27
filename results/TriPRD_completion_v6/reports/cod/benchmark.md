# TriPRD completion v6 official test results

RGB-only inference; full COD10K3040 + CAMO1000 training; fixed last epoch.
Additional VDT source training and a multimodal teacher are part of the training resources.

| Dataset | N | Sm ↑ | Fw ↑ | Emean ↑ | Emax ↑ | Eadaptive ↑ | MAE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| NC4K | 4121 | 0.913445 | 0.882477 | 0.958109 | 0.963412 | 0.959900 | 0.022695 |
| COD10K | 2026 | 0.895018 | 0.831094 | 0.952530 | 0.959975 | 0.949825 | 0.016974 |
| CAMO | 250 | 0.899703 | 0.871965 | 0.950531 | 0.955912 | 0.954325 | 0.032771 |
| CHAMELEON | 76 | 0.922565 | 0.882944 | 0.959777 | 0.968633 | 0.968087 | 0.018206 |

Checkpoint SHA256: `0d667a1ccbc1370f49e8633ea458e196fb7be361cf079d36a41919d3e6bd9534`
Prediction protocol: `triprd_task_v4_fp32_logits_bilinear_align_false_sigmoid_round_uint8_png_v1`

The E-measure variants are separate; these rows do not establish a SOTA claim.
