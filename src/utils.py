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

def plot_wandb_samples(cfg, ts, generator, data_loader, step, avg=False):
    num_plot_samples = cfg.num_plot_samples

    # Get real and fake data
    real_data = data_loader.next()
    assert num_plot_samples <= real_data.size(0)
    real_data = torchcde.LinearInterpolation(real_data).evaluate(ts)

    generator.eval()
    with torch.no_grad():
        generated_data = generator(ts, real_data.size(0)).cpu()
    generated_data = torchcde.LinearInterpolation(generated_data).evaluate(ts)
    generator.train()


    real_data = real_data[:num_plot_samples]
    generated_data = generated_data[:num_plot_samples]

    # Plot samples for each feature
    # real_data = real_data.permute(2,0,1)  # Reshape to (features, num_plot_samples, sample_size)
    # generated_data = generated_data.permute(2,0,1)    # Reshape to (features, num_plot_samples, sample_size)
    # num_features = 1 # TODO: change for multi features
    # for i in range(num_features):

    # Get data for each feature and denormalize
    real_data = real_data * data_loader.std[0] + data_loader.mean[0]
    generated_data = generated_data * data_loader.std[0].cpu() + data_loader.mean[0].cpu()

    for j, real_sample_ in enumerate(real_data):
        kwargs = {'label': 'Real'} if j == 0 else {}
        plt.plot(ts.cpu(), real_sample_[:, 1].cpu(), color='dodgerblue', linewidth=0.5, alpha=0.7, **kwargs)
    for j, generated_sample_ in enumerate(generated_data):
        kwargs = {'label': 'Generated'} if j == 0 else {}
        plt.plot(ts.cpu(), generated_sample_[:, 1].cpu(), color='crimson', linewidth=0.5, alpha=0.7, **kwargs)
    plt.legend()
    plt.title(f"{'[AVG] ' if avg else ''}Step {step}: {num_plot_samples} samples from both real and generated distributions.")
    # plt.savefig(f'{cfg.output_dir}/samples_{i+1}.png')
    fig = wandb.Image(plt)
    wandb.log({'Averaged model samples': fig} if avg else {'Samples': fig})
    plt.close()

def start_wandb(cfg):
    if cfg.use_wandb:
        wandb.login()
        wandb.init(
                project=cfg.wandb_proj,
                config=vars(cfg),
                name=f'{cfg.config_name}_[{cfg.timestamp}]',
                tags=[cfg.config_name, cfg.timestamp],
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

def set_seed(seed=0):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_data_csv(csv_name):
    ''' Get source data from .csv '''
    file_path = f'./data/{csv_name}.csv'
    return pd.read_csv(file_path)