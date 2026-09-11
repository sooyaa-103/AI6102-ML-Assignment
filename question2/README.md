# Q2 experiment

See the [repository overview](../README.md#reproduce-or-verify) for environment setup and commands, and the [experiment protocol](../docs/EXPERIMENTS.md) for the full result tables and settings.

- `run_experiment.py`: reruns all 30 configurations over the same three folds, selects by mean validation accuracy, then fits and tests the selected model.
- `download_data.py`: downloads missing official data and checks file hashes.
- `show_evidence.py cv`: verifies and displays saved cross-validation records; `cv` is the default.
- `show_evidence.py test`: verifies saved predictions against test labels and the original execution log. Requires the downloaded data and numerical dependencies.
- `results/metadata.json`: recorded environment, class counts, protocol, SVC settings, and data hashes.
- `results/cv_results.json`: complete grid, including each fold's counts and convergence status.
- `results/selected_config.json`: selection made before final test evaluation.
- `results/final_test.json`: final held-out evaluation.
- `results/cv_splits.npz`: saved validation indices.
- `results/test_predictions.npy`: saved final predictions.
- `experiment.log`: original run output, retained as evidence.

The full runner overwrites local result files. Committed records preserve the run reported in the PDF; use Git to review changes before replacing them with a new run.
