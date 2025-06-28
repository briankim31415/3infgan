import torch
import yaml
import csv
import random
import numpy as np
import pandas as pd
from types import SimpleNamespace

###################
# TorchSDE standard helper objects.
###################

class LipSwish(torch.nn.Module):
    """Approximation of LipSwish activation function."""
    def forward(self, x):
        return 0.909 * torch.nn.functional.silu(x)

class MLP(torch.nn.Module):
    """MLP neural net class object for generator and discriminator."""
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
            # model.append(LipSwish())
            model.append(torch.nn.Softplus()) # Used in paper (Appendix)
        model.append(torch.nn.Linear(mlp_size, out_size))
        if tanh:
            model.append(torch.nn.Tanh())
        self._model = torch.nn.Sequential(*model)

    def forward(self, x):
        return self._model(x)


###################
# Infinite GAN helper functions.
###################

def set_seed(seed=0):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_device():
    """Determines the appropriate device for PyTorch computations."""
    if torch.cuda.is_available():
        print("Using CUDA.")
        return 'cuda'
    else:
        print("Warning: CUDA not available; falling back to CPU.")
        return 'cpu'

def add_time_channel(ts: torch.Tensor, trajectories: torch.Tensor) -> torch.Tensor:
    """
    Add time as first channel to trajectories.
    
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

def remove_time_channel(trajectories: torch.Tensor) -> torch.Tensor:
    """
    Remove time from first channel of trajectories.

    Args:
        trajectories: Time-augmented trajectories of shape (batch, time, channels+1)

    Returns:
        Trajectories of shape (batch, time, channels) with time channel removed
    """
    return trajectories[:, :, 1:]


###################
# Config loading helper functions.
###################

def load_config_file(cfg_name="default"):
    """Loads the configuration file from the /confs directory."""
    # Get path of config file
    config_path = f'./confs/{cfg_name}.yaml'

    # Open the config file
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)

    # Return config as a SimpleNamespace
    return SimpleNamespace(**config_dict)

def overwrite_cfg(config, ow_cfg):
    """Overwrites values in the first config with those in the overwrite config."""
    # Create a new output SimpleNamespace so values don't carry over
    output_cfg = SimpleNamespace(**vars(config))

    # Iterate through each overwrite parameter
    for key, value in vars(ow_cfg).items():

        # Check if parameter is valid
        if not hasattr(output_cfg, key):
            raise KeyError(f"Key '{key}' not found in configuration.")
        
        # Overwrite parameter
        setattr(output_cfg, key, value)
    
    # Return new overwritten config
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
    # Get path of csv
    csv_path = f'./confs{csv_name}.csv'

    # Iterate through the csv file
    cfg_list = []
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:

            # Convert numerical values from strings to numbers (int/float)
            converted_row = {
                k: int(v) if v.isdigit()
                else float(v) if v.replace('.', '', 1).isdigit() and v.count('.') < 2
                else v
                for k, v in row.items()
            }

            # Create new SimpleNamespace for the given config
            cfg = SimpleNamespace(**converted_row)
            cfg_list.append(cfg)

    # Return list of all configs in csv
    return cfg_list

def get_data_csv(csv_name):
    """Get data from data_source.csv"""
    # Get path of csv
    file_path = f'./data/{csv_name}.csv'

    # Read and return data from csv
    return pd.read_csv(file_path)