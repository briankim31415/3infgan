import wandb
import torch
import os
import yaml
import csv
import argparse
import datetime
from types import SimpleNamespace

###################
# TorchSDE standard helper objects.
###################
import torch

class LipSwish(torch.nn.Module):
    def forward(self, x):
        return 0.909 * torch.nn.functional.silu(x)

class MLP(torch.nn.Module):
    def __init__(self, in_size, out_size, mlp_size, num_layers, tanh):
        super().__init__()

        model = [torch.nn.Linear(in_size, mlp_size),
                 LipSwish()]
        for _ in range(num_layers - 1):
            model.append(torch.nn.Linear(mlp_size, mlp_size))
            ###################
            # LipSwish activations are useful to constrain the Lipschitz constant of the discriminator.
            # (For simplicity we additionally use them in the generator, but that's less important.)
            ###################
            model.append(LipSwish())
        model.append(torch.nn.Linear(mlp_size, out_size))
        if tanh:
            model.append(torch.nn.Tanh())
        self._model = torch.nn.Sequential(*model)

    def forward(self, x):
        return self._model(x)

def get_device():
    if torch.cuda.is_available():
        print("Using CUDA.")
        return 'cuda'
    else:
        print("Warning: CUDA not available; falling back to CPU.")
        return 'cpu'

def plot_wandb_samples():
    pass

def start_wandb(cfg):
    if cfg.use_wandb:
        wandb.login()
        wandb.init(
                project=cfg.wandb_proj,
                config=vars(cfg),
                name=f'{cfg.wandb_name}_[{cfg.timestamp}]',
                tags=[cfg.config_name, cfg.wandb_date_tag],
                mode="online" if cfg.wandb_online else "offline"
            )

def close_wandb(cfg):
    if cfg.use_wandb:
        wandb.finish()

def remove_ts(data):
    """
    Removes the first feature (assumed to be timestep) from the last dimension of the tensor.
    Expects data of shape [batch_size, time_steps, features].
    """

    # TODO Check if this is right
    return data[:, :, 1:]

def append_ts(ts, data):
    """
    Appends the timestep tensor as the first feature along the last dimension.
    Expects timestep of shape [batch_size, time_steps, 1] and data of shape [batch_size, time_steps, features-1].
    """

    # TODO Check if this is right (adds to every trajectory?)
    return torch.cat((ts, data), dim=-1)

def add_time_channel(ts: torch.Tensor, trajectories: torch.Tensor) -> torch.Tensor:
    """Add time as first channel to trajectories.
    
    Args:
        ts: Time points of shape (num_time_steps,)
        trajectories: Trajectories of shape (batch, time, channels)
    
    Returns:
        Time-augmented trajectories of shape (batch, time, channels+1)
    """
    batch_size, num_steps, _ = trajectories.shape
    
    # Expand time to match batch dimension
    time_channel = ts.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 1)
    
    # Concatenate time as first channel
    return torch.cat([time_channel, trajectories], dim=-1)

def load_config_file(cfg_name="default"):
    """
    Loads the configuration file from the /confs directory,
    which is at the same level as the parent of this file's directory (/src),
    and returns it as a SimpleNamespace.
    """
    current_file_dir = os.path.dirname(os.path.abspath(__file__))  # /src
    project_root = os.path.dirname(current_file_dir)  # project root
    config_path = os.path.join(project_root, "confs", f"{cfg_name}.yaml")

    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)

    return SimpleNamespace(**config_dict)

def overwrite_cfg(config, ow_cfg):
    """
    Overwrites values in the default configuration with those in the provided config.
    Both arguments are expected to be SimpleNamespace instances.
    """
    output_cfg = SimpleNamespace(**vars(config))
    for key, value in vars(ow_cfg).items():
        if not hasattr(output_cfg, key):
            raise KeyError(f"Key '{key}' not found in configuration.")
        setattr(output_cfg, key, value)
    return output_cfg


def load_csv_cfgs(csv_name):
    """
    Loads a CSV file and returns a list of SimpleNamespace objects.
    Each row is converted into a SimpleNamespace where columns are keys.

    Args:
        csv_name (str): Path to the CSV file.

    Returns:
        List[SimpleNamespace]: List of config objects.
    """
    cfg_list = []
    current_file_dir = os.path.dirname(os.path.abspath(__file__))  # /src
    project_root = os.path.dirname(current_file_dir)  # project root
    csv_path = os.path.join(project_root, "confs", f"{csv_name}.csv")
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            converted_row = {
                k: int(v) if v.isdigit()
                else float(v) if v.replace('.', '', 1).isdigit() and v.count('.') < 2
                else v
                for k, v in row.items()
            }
            cfg = SimpleNamespace(**converted_row)
            cfg_list.append(cfg)
    return cfg_list

def parse_args():
    parser = argparse.ArgumentParser(description="Run Infinite GAN training.")
    parser.add_argument("--use_wandb", action="store_true", help="Enable wandb logging.")
    parser.add_argument("--online", action="store_true", help="Set wandb mode to online")
    parser.add_argument("--cfg_name", type=str, default="default", help="Config file name.")
    parser.add_argument("--multirun_cfg", type=str, default=None, help="Multi-run config file.")
    return parser.parse_args()

def set_def_args(config, args):
    config.use_wandb = args.use_wandb
    config.wandb_online = args.online
    config.device = get_device()
    config.timestamp = datetime.now().strftime("%m/%d")
