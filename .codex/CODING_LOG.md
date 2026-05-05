# Coding Log

This file keeps durable project context because Codex chats are not synced across machines.

Note: I do not have access to full historical chat metadata from earlier Codex sessions. The chronology below is reconstructed from the repository's Git history, current code, and the local `.codex` notes. Dates are commit dates where available.

## Current State - 2026-05-05

### Current goal
Get RandLA-Net / Open3D-ML training stable on SemanticKITTI.

### Current config decisions
- Pipeline: `SemSegExt`.
- Model: `RandLANetFlat`.
- Dataset: `SemanticKITTIFlat`.
- Training mode default: `train+eval`.
- `dataset.dataset_path` is mandatory via Hydra override; no hardcoded machine-specific path in YAML.
- Model trains on 19 compact SemanticKITTI classes.
- SemanticKITTI train ID `0` / `unlabeled` is ignored via `ignored_label_inds: [0]`.
- Predictions must be restored from compact class IDs to SemanticKITTI train IDs before visualization/export.
- Current optimizer LR: `0.001`.
- Current scheduler gamma: `0.9886`.
- Current batch sizes: `1` for train/val/test.
- Current data-loader config: `num_workers: 4`, `prefetch_factor: 2`, `persist_workers: true`, `pin_memory: false`.
- `clean: false` by default; `clean: true` removes old checkpoint/log and cache dirs before a run.
- Validation split `08` is also used as `test_split` until real SemanticKITTI test handling is added.

### Current open questions
- Is `num_workers: 4` stable on both local machines and the cluster, or should debug runs keep an override with `num_workers: 0`?
- Is eval instability caused by sampling, class imbalance, label mapping, or config?
- Are Open3D-style class weights from raw class counts still the right baseline, or should an unweighted baseline be compared first?
- Should `in_channels: 3` stay fixed for the stable baseline, or should intensity be tested later as a controlled ablation?
- Should eval-only runs require explicit `model.ckpt_path` instead of using the latest checkpoint automatically?

### Current next steps
1. Run `pytest` to confirm label/loss/projection contracts still pass.
2. Add a tiny debug config or documented command recipe for local smoke tests.
3. Run a short train smoke test with `dataset.dataset_path=...` and confirm checkpoint + TensorBoard outputs are created.
4. Compare smoke behavior with `num_workers=0` and current `num_workers=4`.
5. Add a SLURM smoke-test script once the local smoke command is stable.
6. Only then tune LR / scheduler / class weighting.

### Useful run notes
- Local train/eval smoke command shape:
  `python -m lidar_owl.main dataset.dataset_path=/path/to/sequences pipeline.max_epoch=1`
- Eval-only command shape:
  `python -m lidar_owl.main mode=eval dataset.dataset_path=/path/to/sequences model.ckpt_path=/path/to/ckpt.pth`
- For reproducibility, prefer explicit Hydra overrides in notes/scripts instead of editing machine-specific paths into YAML.

## Chronological Notes

### 2025-10-16 - Initial repo and environment baseline
- `c5d08e7` initial commit.
- `9585b3f`, `69e7a40` established stable Python/CUDA/Torch config.

### 2025-10-17 - README update
- `af80739` updated README.

### 2025-10-19 - Environment and data loading
- `3feb50b` fixed environment issues and tested data loading.

### 2025-10-21 - Model loading
- `dc46715` got model loading working.

### 2025-10-23 - Training on Mac
- `8225585` got training working on Mac.

### 2025-10-24 - Trainer and TODOs
- `33f6539` added trainer/TODOs and tested training.

### 2025-11-08 - Run modes
- `e4513dd` added argparse support for mode and debug flag.

### 2025-11-12 - Logging starts
- `e22ac88`, `7c116db` made logging partially work.

### 2025-11-14 - BEV visualization
- `557208e`, `6fd17d1` got BEV visualization working, with remaining color/dimension issues.

### 2025-11-15 - SemanticKITTI colors and prediction mapping
- `742d848` added SemanticKITTI colors.
- Prediction mapping with ignored labels was still buggy.

### 2025-11-17 - Semantic segmentation pipeline tested
- `c7ce21d` reported semantic-segmentation pipeline fully tested with metrics, loss, and visualization.

### 2025-11-18 - Config cleanup and Hydra
- `6d12b87` decluttered config files and tested training.
- `3532281` incorporated Hydra; training ran, checkpoint loading still had a TODO.
- `9db7778` removed old YAMLs and updated `.gitignore`.

### 2025-11-19 - Checkpoint and pipeline refactoring
- `162c26d`, `fe47715` worked on checkpoint saving, pipeline `run/test` refactoring, and `ConfigDict` fixes.
- `d0ac8d0` added clean run mode.

### 2025-11-20 - Dataset path config
- `648ebbe` moved dataset path handling into config.

### 2025-11-24 - Refactoring
- `db1f420` added fixes, refactoring, and TODO cleanup.

### 2026-03-09 - Flat CE and first tests
- `779e58d` implemented basic flat cross entropy and resolved configurable loss handling.
- `6593204` added first loss/dataset tests, TOML setup, and import cleanup.
- `9bae4ad` fixed README header formatting.

### 2026-03-16 - Label mapping
- `194f4e5` got label mapping mostly correct; mIoU still looked odd.

### 2026-03-17 - Save-log cleanup
- `363f3a0` cleanup.
- `dccd54e` cleaned up `save_logs`; train visualization was the main remaining use.
- `0c91292` noted that BEV visualization should be cleaned up for val/test separately instead of overloading train `save_logs`.

### 2026-03-19 - Test run TensorBoard stub
- `246e07c` cleaned up train `save_logs` and added a `run_test` stub for TensorBoard testing.

### 2026-03-21 - Test predictions and debug visualization
- `f1b8977` code was running, but predictions in test run were off.
- `61c50b2` made small fixes and noted train/val visualization should be checked.
- `6201ba2` added debug train visualization.

### 2026-03-22 - `run_test` behavior
- `77165f8` made `run_test` work like the Open3D-ML equivalent.
- `b853b8a` minor cleanup.

### 2026-03-24 - Cleanup
- `e114eec` cleanup.

### 2026-03-25 - Class count fix
- `4413fdf` fixed `num_classes`.

### 2026-03-26 - Eval metric extension
- `d7599b6` added metric class extension for the eval routine.

### 2026-04-11 - Loss calls and tests
- `f75777a` fixed loss calls.
- `aeeb45a` cleaned up loss tests.

### 2026-04-30 - Helix and Open3D environment
- `e45d51a` made adjustments for Helix.
- `ce60dfa` updated the Conda environment for compatibility with Open3D `0.18`.

### 2026-05-05 - Local Codex project notes
- Added `.codex/AGENTS.md`, `.codex/TODO.md`, `.codex/CODING_LOG.md`, and `.codex/config.toml` for durable local context.
- Documented current project constraints:
  - small, reviewable changes;
  - preserve experiment reproducibility;
  - avoid hardcoded absolute paths;
  - prioritize stable baselines, clean evaluation, and reproducible experiments over adding new methods.
- Current uncommitted config change in `configs/pipeline/semseg_ext.yaml`:
  - `num_workers` changed from `0` to `4`;
  - added `prefetch_factor: 2`;
  - added `persist_workers: true`;
  - kept `pin_memory: false`.

## Established Technical Contracts

### SemanticKITTI label contract
- Dataset labels include train ID `0` for ignored/unlabeled points.
- Open3D-ML training/eval uses compact class IDs after ignored labels are filtered.
- Model output has 19 channels/classes.
- Visualization/export must restore predicted compact IDs back to SemanticKITTI train IDs.
- Metrics must compare compact predictions against compact labels, not compact predictions against raw train IDs.

### Local wrappers and registries
- `SemanticKITTIFlat` wraps Open3D-ML SemanticKITTI and returns samples with `feat: None`.
- `RandLANetFlat` wraps Open3D-ML RandLA-Net.
- Local resolvers register `SemanticKITTIFlat` and `RandLANetFlat` while still allowing fallback to Open3D-ML registries.

### Loss contract
- `CrossEntropyFlat` expects ignored points to have already been removed.
- Label compaction happens in `BaseFlatAdapter.get_loss()` through Open3D-ML valid-label filtering.
- When `model.loss.class_weights: true`, dataset class counts are converted to Open3D-style CE class weights before loss construction.

### Eval/logging contract
- Eval resolves the latest checkpoint if `model.ckpt_path` is unset.
- TensorBoard eval logs go under `pipeline.eval_sum_dir`.
- Projection images should use restored prediction labels so colors match SemanticKITTI train IDs.
- Test metrics should filter labels the same way as training loss.

### Regression test coverage
- SemanticKITTI name/palette contract.
- Projection behavior with ignored labels and visibility masks.
- Compact prediction restoration.
- Custom loss adapter path and Open3D fallback path.
- MRO contract: local adapter must precede Open3D model class.
