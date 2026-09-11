# Experiment protocol and results

## Recorded environment

| Component | Version / setting |
|---|---|
| Python | 3.10.9 |
| scikit-learn | 1.7.2 |
| NumPy | 2.2.6 |
| SciPy | 1.15.3 |
| Training data | 32,561 examples, 123 features |
| Test data | 16,281 examples, 123 features |
| Validation | StratifiedKFold, 3 splits, shuffle=True, random_state=42 |
| Fold sizes | 10,854; 10,854; 10,853 validation examples |
| Preprocessing | No extra scaling, selection, or resampling; CSR converted to dense |
| SVC options | tol=0.001, shrinking=True, probability=False, class_weight=None |
| Runtime options | cache_size=512 MB, max_iter=-1, six worker processes |

Both input files use `n_features=123`; automatic inference on the test file alone would produce only 122 columns. All models share the same fold split. Means are arithmetic averages of the three held-out fold accuracies, displayed to four decimal places. Selection uses full precision. All 90 fold fits have `fit_status=0` and no recorded warnings.

## Table 1 · Linear kernel

| C | Fold 1 (%) | Fold 2 (%) | Fold 3 (%) | Mean (%) |
|---:|---:|---:|---:|---:|
| 0.01 | 84.4481 | 84.3099 | 84.3730 | 84.3770 |
| 0.05 | 84.7890 | 84.5587 | 84.6586 | 84.6688 |
| 0.1 | 84.8443 | 84.6785 | 84.7415 | 84.7548 |
| 0.5 | 84.8719 | 84.6324 | 84.7139 | 84.7394 |
| 1 | 84.8627 | 84.6232 | 84.7508 | 84.7456 |

## Table 2 · RBF kernel

The kernel is `exp(-gamma * ||x - z||²)`. Thus `gamma = 1 / (2 sigma²)` for the width convention in the lecture. The requested numeric grid values are passed directly to SVC's `gamma` argument.

| C / gamma | 0.01 | 0.05 | 0.1 | 0.5 | 1 |
|---:|---:|---:|---:|---:|---:|
| 0.01 | 75.9190 | 81.9784 | 81.9877 | 75.9190 | 75.9190 |
| 0.05 | 83.1209 | 83.5816 | 83.4495 | 78.9841 | 75.9190 |
| 0.1 | 83.7351 | 83.9348 | 83.9133 | 80.5933 | 76.1832 |
| 0.5 | 84.3432 | 84.4968 | 84.6381 | 83.1946 | 78.9626 |
| 1 | 84.4599 | 84.7363 | 84.7271 | 83.5939 | 79.9730 |

## Table 3 · Selection and final test

| Quantity | Recorded result |
|---|---:|
| Selected kernel | Linear |
| C | 0.1 |
| Mean validation accuracy | 84.7548% |
| Best RBF setting | C=1, gamma=0.05 |
| Best RBF validation accuracy | 84.7363% |
| Final test accuracy | 85.0132% |
| Correct test predictions | 13,841 / 16,281 |

The validation gap between the best kernels is only **0.0184 percentage points**. Selection is specific to this grid and fold assignment; it does not establish general superiority of linear kernels. The selected configuration was saved before the final test labels were loaded by the runner.

Final confusion matrix (rows = true labels, columns = predictions):

| True / predicted | -1 | +1 |
|---|---:|---:|
| -1 | 11,579 | 856 |
| +1 | 1,584 | 2,262 |

## Evidence and provenance

- [Full cross-validation records](../question2/results/cv_results.json)
- [Selection record](../question2/results/selected_config.json)
- [Final test record](../question2/results/final_test.json)
- [Environment and input hashes](../question2/results/metadata.json)
- [Original execution log](../question2/experiment.log)
- [Saved validation indices](../question2/results/cv_splits.npz)
- [Saved test predictions](../question2/results/test_predictions.npy)
- [Cross-validation screenshot](../assets/q2_cv.png) and [test screenshot](../assets/q2_test.png)

The log and screenshots retain the original local-run context. Terminal evidence displays verified saved records; it is not a screenshot of rerunning all 90 fits. The full training script is supplied separately.

### Official input-file SHA-256 hashes

- `a9a`: `f5d5ffd8d865ff41328e7ee043e4b020816914ff6843ff15b98905ddbedce906`
- `a9a.t`: `1f448a153f0320399a7e40836eb207655b0bde0f21fc941cc472193daa9f5de9`

### Preserved PDF SHA-256 hashes

- `report/AI6102_Assignment.pdf`: `9e254b5016d6ad80abb1e11ce71b2d9e838490b088880f13fbeaaad3a2cb6ec4`
- `assignment/original-assignment.pdf`: `2c4665978aaee49245546bfee4f2ab81f18e54fbe347daf181ed7d0081dc737f`
