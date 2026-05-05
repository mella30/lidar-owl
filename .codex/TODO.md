# TODO

## Immediate
- [x] `configs/pipeline/semseg_ext.yaml:4` — pin_memory: false  # TODO: test if faster, then allocate more CPU RAM
- [ ] `src/lidar_owl/pipelines.py:19` — # TODO: implement early stopping
- [ ] `src/lidar_owl/main.py:11` — # TODO: resume training (max time on cluster: 40h)
- [ ] `src/lidar_owl/pipelines.py:20` — # TODO: log git hash for each run


## Later
- [ ] `src/lidar_owl/models.py:1` — # TODO: hierarchical models, derive from ml3d base model
- [ ] `src/lidar_owl/models.py:3` — # TODO: uncertainty models
- [ ] `src/lidar_owl/models.py:4` — # TODO: anomaly models
- [ ] `src/lidar_owl/losses.py:1` — # TODO: hierarchical losses, metric learning
- [ ] `configs/model/randlanet.yaml:9` — dim_output: [16, 64, 128, 256]  # TODO: needs to be overridden for metric learning
- [ ] `src/lidar_owl/datasets.py:3` — # TODO: carla dataset
- [ ] `src/lidar_owl/pipelines.py:25` — self.color_map = log.semkitti_cmap(self.num_classes)  # TODO: depends on dataset!
- [ ] `configs/model/randlanet.yaml:6` — num_classes: 19  # TODO: dataset dependent
- [ ] `src/lidar_owl/log.py:1` — # TODO: visu, runtime