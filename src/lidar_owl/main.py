import sys
import argparse
from typing import Any

import open3d.ml.utils as ml3d_utils

from train import Trainer
from eval import Evaluator


def parse_args():
    argparser = argparse.ArgumentParser(description="LiDAR Owl entry point")
    argparser.add_argument(
        "--mode",
        "-m",
        type=str,
        required=True,
        help="Defines the usage mode: "
        "- train (provide empty log dir, default: log), "
        "- test (provide model dir, default: log_dir/model)",
    )
    argparser.add_argument(
        "--debug",
        "-db",
        action="store_true",
        required=False,
        default=False,
        help="Flag to control debugging mode. Default: False",
    )
    argparser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to the YAML config file (no default).",
    )
    
    return argparser.parse_args()

def handle_config(cfg: dict[str, Any], debug:bool=False):
    # TODO: handle hierarchical model stuff

    # dataset stuff
    if debug:
        cfg['dataset']['training_split'] = ['08']
        cfg['dataset']['validation_split'] = ['08']
    else:
        cfg['dataset']['training_split'] = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
        cfg['dataset']['validation_split'] = ['08']

    # use preprocessed dataset cache if cache dir is given
    cfg['dataset']['use_cache'] = True if cfg.get("dataset", {}).get("cache_dir") else False

    # model stuff  
    model_cfg = cfg.setdefault("model", {})
    model_cfg['num_classes'] = 19  # TODO: depends on dataset
    model_cfg['in_channels'] = 3  # never use intensity to train
    try:
        model_name = model_cfg.get("name")
        # RandLaNet specifics
        if model_name and model_name.lower() == "randlanetflat":
            model_cfg['num_neighbors'] = 16
            model_cfg['num_layers'] = 4
            model_cfg['num_points'] = 45056
            model_cfg['sub_sampling_ratio'] = [4, 4, 4, 4]
            model_cfg['dim_features'] = 8
            model_cfg['grid_size'] = 0.06
            # ensure nested augment cfg exists before setting defaults
            recenter_cfg = model_cfg.setdefault("augment", {}).setdefault("recenter", {})
            recenter_cfg.setdefault("dim", [0, 1])
    except KeyError as ex:
        raise KeyError("mandatory field model.name is missing") from ex
    
    # pipeline stuff
    cfg['pipeline']['num_workers'] = 0  # known ml3d bug 
    cfg['pipeline']['pin_memory'] = False  # known ml3d bug 
    
    return cfg


def main():
    args = parse_args()

    # TODO: use hydra
    # set default config values
    cfg = handle_config(ml3d_utils.Config.load_from_file(args.config), args.debug)

    if args.mode == "train":
        semseg_trainer = Trainer(cfg)
        semseg_trainer.train()

    if args.mode == "eval":
        semseg_evaluator = Evaluator(cfg)
        semseg_evaluator.eval()

if __name__ == "__main__":
    main()
