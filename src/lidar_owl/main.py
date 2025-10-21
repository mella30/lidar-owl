import os
import open3d.ml.torch as ml3d_torch
import open3d.ml as ml3d

DATA_DIR = '/data/'

def main():
    # read config file
    # config = yaml.safe_load(open(os.path.join('configs', 'pointpillars_kitti.yaml')))
    cfg_file = os.path.join('configs', 'randlanet_semantickitti.yaml')
    cfg = ml3d.utils.Config.load_from_file(cfg_file)
    
    # construct a dataset by specifying dataset_path
    cfg.dataset['dataset_path'] = os.path.join(DATA_DIR, 'datasets', 'public_datasets', 'semantic_kitti')
    dataset = ml3d_torch.datasets.SemanticKITTI(**cfg.dataset)

    model = ml3d_torch.models.RandLANet(**cfg.model)

    pipeline = ml3d_torch.pipelines.SemanticSegmentation(model=model, dataset=dataset, max_epoch=100)
    pipeline.run_train()

    print("Training and testing completed.")


if __name__ == "__main__":
    main()