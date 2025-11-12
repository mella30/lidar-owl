import argparse

import open3d.ml.utils as ml3d_utils

from train import Trainer
from test import Tester


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


def main():
    args = parse_args()

    # TODO: use hydra
    cfg = ml3d_utils.Config.load_from_file(args.config)

    if args.mode == "train":
        semseg_trainer = Trainer(cfg)
        semseg_trainer.train()

    if args.mode == "test":
        semseg_tester = Tester(cfg)
        semseg_tester.test()

if __name__ == "__main__":
    main()
