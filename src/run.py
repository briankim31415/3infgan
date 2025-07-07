"""
RUN.PY

This file contains the main runner class for the Infinite GAN.

It contains the following functions:
- parse_args: Create a parser and add arguments for command line.
- set_def_args: Set default config parameters to argument values.
- main: Main runner class for Infinite GANs.
"""


import argparse
from datetime import datetime
from .train import train
from .utils import load_config_file, load_csv_cfgs, overwrite_cfg, get_device


def parse_args():
    """Create a parser and add arguments for command line."""
    parser = argparse.ArgumentParser(description="Run Infinite GAN training.")
    parser.add_argument("--use_wandb", action="store_true", help="Enable wandb logging.")
    parser.add_argument("--online", action="store_true", help="Set wandb mode to online")
    parser.add_argument("--cfg_name", type=str, default="default", help="Config file name.")
    parser.add_argument("--multirun_cfg", type=str, default=None, help="Multi-run config file.")
    return parser.parse_args()

def set_def_args(config, args):
    """Set default config parameters to argument values."""
    config.use_wandb = args.use_wandb
    config.wandb_online = args.online
    config.device = get_device()
    config.timestamp = datetime.now().strftime("%m/%d")


def main():
    """Main runner class for Infinite GANs."""
    # Get argument parser
    args = parse_args()

    # Get default config file
    config = load_config_file()

    # Load arguments
    set_def_args(config, args)

    # Overwrite any parameters
    if args.cfg_name != "default":
        ow_cfg = load_config_file(args.cfg_name)
        config = overwrite_cfg(config, ow_cfg)

    # Get multi-run configs
    config_runs = []
    if args.multirun_cfg is not None:
        multi_run_cfgs = load_csv_cfgs(args.multirun_cfg)
        for i, cfg in enumerate(multi_run_cfgs):
            add_cfg = overwrite_cfg(config, cfg)
            config_runs.append(add_cfg)
    else:
        config_runs.append(config)
    
    # Iterate through config(s)
    for run_cfg in config_runs:

        # Run training
        generator, discriminator = train(run_cfg)

        # Save models (todo later)
    


if __name__ == "__main__":
    main()