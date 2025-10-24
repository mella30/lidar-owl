import os
import open3d.ml.utils as ml3d_utils

from train import Trainer

# DATA_DIR = '/data/'
DATA_DIR = '/Users/m30/data/'

def main():
    # TODO: use hydra
    cfg_file = os.path.join('configs', 'randlanet_semantickitti.yaml')
    cfg = ml3d_utils.Config.load_from_file(cfg_file)
    
    # construct a dataset by specifying dataset_path
    cfg.dataset['dataset_path'] = os.path.join(DATA_DIR, 'datasets', 'public_datasets', 'semantic_kitti')

    semseg_trainer = Trainer(cfg)
    semseg_trainer.train()

if __name__ == "__main__":
    main()