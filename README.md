# LiDAR-OWL (open world learning) — PhD Monorepo by M30

This repository is the monorepo for my dissertation and builds on Open3D-ML. It contains benchmarks, implementations and configurations for (hierarchical) LiDAR semantic segmentation, uncertainty estimation and anomaly/novelty detection.

Key points
- Built on Open3D-ML (see dependencies in env.yaml).
- Entry points: e.g. `lidar_owl.main` (see `src/lidar_owl/main.py`).
- Tests are included (see `tests/`).
- Environment and project configs are in `env.yaml`.

Quickstart
1. Create the conda environment:
   - conda env create -f env.yaml
2. Activate and develop:
   - conda activate lidar-owl
   - Run scripts, e.g. `python -m lidar_owl.main`
3. Run tests:
   - pytest

# Contents:
Benchmarks (outline)
1. SemSeg Performance
   - Dataset: any
   - Metrics: mIoU, runtime
2. Uncertainty Estimation
   - Train & val on same dataset (real: KITTI / STF / nuScenes, sim: CARLA)
   - Metrics: ECE, NLL (?), AUSE, uIoU
3. Uncertainty Disentangling
   - CARLA: aleatoric, epistemic & label uncertainty
   - Metrics: TBD
4. Domain Generalization / Robustness
   - Train / val on different datasets (e.g. KITTI → STF)
   - Metric: mIoU
5. Anomaly / Novelty Detection
   - KITTI, STF & nuScenes with 1–2 masked classes
   - Metrics: CER, FPR95, AUROC / AUPRC
6. Optional: Hierarchical SemSeg — integrate into other benchmarks
   - Datasets with label hierarchies
   - Metrics: hIoU, average confidence, information content/flow (entropy, MI, …), SR, CER

Models & Baselines
- SemSeg backbones: those available in Open3D-ML
- Uncertainty baselines: Ensembles, Logit Sampling, DDU 
- Anomaly baselines: OE, CP, CMAV
- Metric learning: prototypes, memory bank, hinge loss, objectosphere, convex-hull, hierarchical variants; negative sampling: random, hard, informed

Dissertation Contents:
- Preliminary Study - Uncertainty Types
- Chapter 1 — Hierarchical Uncertainty Estimation
- Chapter 2 — Hierarchical Novelty Detection