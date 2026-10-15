# Phase 1 baseline results

One seed: value [95% lesion-bootstrap CI]. Several seeds: mean ± SD.

## random

| test set | model | balanced_acc | macro_f1 | macro_auroc | mel_auroc | mel_sens_at_spec95 | mel_sens_valthr | ece |
|---|---|---|---|---|---|---|---|---|
| test | dinov2_probe | 0.680 [0.619, 0.752] | 0.618 [0.559, 0.665] | 0.941 [0.931, 0.953] | 0.846 [0.806, 0.880] | 0.383 [0.263, 0.482] | 0.419 [0.331, 0.510] | 0.032 [0.029, 0.059] |

## site_rosendahl

| test set | model | balanced_acc | macro_f1 | macro_auroc | mel_auroc | mel_sens_at_spec95 | mel_sens_valthr | ece |
|---|---|---|---|---|---|---|---|---|
| test_rosendahl | convnext_tiny | 0.736 [0.683, 0.793] | 0.736 [0.674, 0.794] | 0.915 [0.886, 0.941] | 0.869 [0.814, 0.915] | 0.444 [0.245, 0.660] | 0.463 [0.312, 0.620] | 0.129 [0.089, 0.183] |
| test_rosendahl | dinov2_probe | 0.619 [0.553, 0.681] | 0.612 [0.543, 0.670] | 0.852 [0.814, 0.887] | 0.787 [0.714, 0.853] | 0.370 [0.167, 0.511] | 0.167 [0.070, 0.295] | 0.119 [0.097, 0.179] |
| test_vidir_modern | convnext_tiny | 0.683 [0.626, 0.738] | 0.634 [0.572, 0.695] | 0.866 [0.833, 0.898] | 0.772 [0.707, 0.836] | 0.218 [0.109, 0.421] | 0.307 [0.202, 0.419] | 0.173 [0.133, 0.224] |
| test_vidir_modern | dinov2_probe | 0.571 [0.507, 0.635] | 0.577 [0.506, 0.640] | 0.820 [0.781, 0.857] | 0.690 [0.605, 0.767] | 0.287 [0.133, 0.396] | 0.119 [0.052, 0.193] | 0.065 [0.049, 0.120] |

## site_vidir_modern

| test set | model | balanced_acc | macro_f1 | macro_auroc | mel_auroc | mel_sens_at_spec95 | mel_sens_valthr | ece |
|---|---|---|---|---|---|---|---|---|
| test_rosendahl | convnext_tiny | 0.628 [0.560, 0.699] | 0.626 [0.554, 0.689] | 0.843 [0.804, 0.884] | 0.779 [0.701, 0.851] | 0.296 [0.143, 0.479] | 0.241 [0.113, 0.375] | 0.244 [0.191, 0.312] |
| test_rosendahl | dinov2_probe | 0.552 [0.484, 0.621] | 0.533 [0.462, 0.598] | 0.826 [0.784, 0.862] | 0.814 [0.741, 0.879] | 0.333 [0.203, 0.500] | 0.296 [0.187, 0.409] | 0.070 [0.059, 0.134] |
| test_vidir_modern | convnext_tiny | 0.780 [0.715, 0.836] | 0.740 [0.675, 0.793] | 0.926 [0.900, 0.949] | 0.868 [0.810, 0.921] | 0.535 [0.387, 0.740] | 0.406 [0.277, 0.526] | 0.136 [0.105, 0.187] |
| test_vidir_modern | dinov2_probe | 0.733 [0.680, 0.785] | 0.679 [0.612, 0.734] | 0.892 [0.861, 0.921] | 0.778 [0.690, 0.855] | 0.396 [0.268, 0.546] | 0.347 [0.220, 0.468] | 0.113 [0.088, 0.158] |

## Site-shift gap (melanoma AUROC, in-site minus cross-site, same model)

| trained on | model | in-site | cross-site | gap |
|---|---|---|---|---|
| rosendahl | convnext_tiny | 0.869 | 0.772 | +0.097 |
| rosendahl | dinov2_probe | 0.787 | 0.690 | +0.097 |
| vidir_modern | convnext_tiny | 0.868 | 0.779 | +0.089 |
| vidir_modern | dinov2_probe | 0.778 | 0.814 | -0.036 |
