################
# Main Run Class.
################
from .train import train
from .utils import parse_args, load_config_file, load_csv_cfgs, overwrite_cfg, set_def_args

'''
Train loop
Init wandb(utils)/close
cfg file selector
"Store final models" >> later
'''


def main():
    # Get args
    args = parse_args()

    # Get default config file
    config = load_config_file()
    set_def_args(config, args)

    # Overwrite any parameters
    if args.cfg_name != "default":
        ow_cfg = load_config_file(args.cfg_name)
        config = overwrite_cfg(config, ow_cfg)

    # Get multi-run configs
    config_runs = []
    if args.multirun_cfg is not None:
        multi_run_cfgs = load_csv_cfgs(args.multirun_cfg)
        for cfg in multi_run_cfgs:
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