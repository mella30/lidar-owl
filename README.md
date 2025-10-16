# lidar-owl

Short description
A research and development repository for LiDAR semantic segmentation with a focus on uncertainty estimation, anomaly/novelty detection, and hierarchical semantic segmentation. Built on top of open3d-ml and semseg models. Configuration via Hydra. Development setup with Conda.

Core ideas
- Base stack: open3d-ml (point-cloud backbones) + semseg models for LiDAR point clouds.
- Research topics:
  - Semantic segmentation (including hierarchical / multi-scale labels)
  - Uncertainty estimation (e.g., MC-Dropout, Deep Ensembles, Evidential methods, Temperature Scaling, Test-time Augmentation)
  - Anomaly / Novelty detection (reconstruction-based, density-based, centroid-distance, energy-based approaches)
  - Benchmarks: multiple datasets and evaluation metrics (see below)

Installation (Conda, recommended)
1. Create and activate the Conda environment:
```bash
conda env create -f environment.yml
conda activate lidar-owl
```
2. If needed, install additional packages (GPU drivers / CUDA-compatible PyTorch). Install open3d-ml either from pip or from source for specific versions:
```bash
pip install -r requirements.txt
# or
pip install open3d-ml
```
The file environment.yml in the project root contains base dependencies (Python, PyTorch, open3d-ml, etc.).

Configuration (Hydra)
- Configuration files live under conf/ or configs/.
- Example run:
```bash
python -m src.lidar_owl.main --config-name=default
# with overrides:
python -m src.lidar_owl.main dataset=semantic_kitti experiment=segmentation
```
Hydra enables reproducible experiment configs, sweeping, and versioned outputs (logs/checkpoints per run).

Usage
- Train / evaluate / infer via the CLI in src/lidar_owl/main.py (Hydra-driven).
- For cluster jobs using Slurm, activate the Conda env in the job script:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate lidar-owl
```
If required for the university cluster, consider migrating to pyproject + Docker/Singularity for containerized images.

Tests
- Unit tests with pytest:
```bash
pytest
```
(Tests located in tests/)

Datasets & Benchmarks (examples)
- Datasets: SemanticKITTI, nuScenes (lidarseg), KITTI-360, SemanticPOSS, Paris-Lille-3D (extendable)
- Metrics:
  - Segmentation: mean IoU, per-class IoU, precision, recall, F1, overall accuracy
  - Uncertainty / Calibration: ECE (Expected Calibration Error), NLL (Negative Log-Likelihood)
  - Anomaly / Novelty: AUROC, AUPR, FPR@95TPR

Design notes
- Modular structure: separate backbone implementations (open3d-ml) from segmentation heads and uncertainty modules.
- Pipeline sketch: dataset → backbone → segmentation head → uncertainty module → evaluation (segmentation + anomaly metrics).
- Experiment tracking: Hydra outputs + optional logging (TensorBoard, Weights & Biases).

Contributing
- Contributions welcome. Open PRs for small, tested changes and add tests where applicable.
- Use issue templates for bug reports and feature requests.

License
MIT — see LICENSE.