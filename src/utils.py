import wandb
import torch
import os
import yaml
import csv
import random
import torchcde
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

# Remove time from first channel of trajectories
def remove_time_channel(trajectories: torch.Tensor) -> torch.Tensor:
    """Remove time from first channel of trajectories.

    Args:
        trajectories: Time-augmented trajectories of shape (batch, time, channels+1)

    Returns:
        Trajectories of shape (batch, time, channels) with time channel removed
    """
    return trajectories[:, :, 1:]

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

def set_seed(seed=0):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_data_csv(csv_name):
    """Get source data from .csv"""
    file_path = f'./data/{csv_name}.csv'
    return pd.read_csv(file_path)