# TODO: class mapping for hierarchical trainings
# TODO: mask managing (via yaml)
# TODO: carla dataset
from pathlib import Path
import open3d
import open3d.ml.torch as ml3d
import yaml
import numpy as np

# open3d-ml dataset wrapper
class SemanticKITTISplitFlat(open3d._ml3d.datasets.semantickitti.SemanticKITTISplit):
    def get_data(self, idx):
        sample = super().get_data(idx)
        # remove intensity from feature channel (should not be used for training since it's quite uncalibrated)
        sample["feat"] = None 
        return sample

class SemanticKITTIFlat(ml3d.datasets.SemanticKITTI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        resource = Path(open3d._ml3d.__file__).parent / "datasets" / "_resources" / "semantic-kitti.yaml"
        self.class_config = yaml.safe_load(resource.read_text())

        # sensor FOV for projection visualizations
        self.range_size = self.cfg.cfg_dict["sensor_fov"]

        # label metadata for visualization and class mapping
        self.train_id_to_raw_id = {
            int(train_id): int(raw_id)
            for train_id, raw_id in self.class_config["learning_map_inv"].items()
        }
        raw_id_to_name = {
            int(raw_id): name for raw_id, name in self.class_config["labels"].items()
        }
        raw_id_to_color = {
            int(raw_id): np.asarray(color, dtype=np.float32) / 255.0
            for raw_id, color in self.class_config["color_map"].items()
        }
        num_train_ids = max(self.train_id_to_raw_id) + 1
        self.label_to_names = {
            train_id: raw_id_to_name[raw_id]
            for train_id, raw_id in self.train_id_to_raw_id.items()
        }
        self.label_names = [
            self.label_to_names[train_id] for train_id in range(num_train_ids)
        ]
        self.label_colors = np.zeros((num_train_ids, 3), dtype=np.float32)
        for train_id, raw_id in self.train_id_to_raw_id.items():
            self.label_colors[train_id] = raw_id_to_color[raw_id]

    def get_label_names(self, num_classes: int, compact: bool = False, ignored_label_inds=()):
        names = self.label_names[:num_classes + len(ignored_label_inds)] if compact else self.label_names[:num_classes]
        if compact:
            ignored = {int(label) for label in ignored_label_inds if int(label) >= 0}
            names = [name for idx, name in enumerate(names) if idx not in ignored]
        return names[:num_classes]

    def get_label_colors(self, num_classes: int):
        return self.label_colors[:int(num_classes)]
    
    def get_split(self, split):
        return SemanticKITTISplitFlat(self, split=split)
