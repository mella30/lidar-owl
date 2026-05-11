# TODO

## Immediate
- [x] benchmark semseg performance (own vs open3dml sample script)
- [ ] save preprocessed npy files in workspace & copy into TMP dir
- [ ] implement early stopping & resume training (max time on cluster: 40h)
- [x] log git hash for each run
- [ ] remove all .get() accesses in dicts and replace with [<key>]


## Later
- [ ] hierarchical models, derive from ml3d base model
- [ ] uncertainty models
- [ ] anomaly models
- [ ] hierarchical losses, metric learning (NOTE: feature space in model config needs to be overridden for metric learning)
- [ ] carla dataset (make nr classes, cmap dependent on dataset)
- [ ] metric & runtime visu