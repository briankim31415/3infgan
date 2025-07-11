"""
DATA.PY

This file contains the data class for the Infinite GAN.
It is used to load the config dataset and create a torch dataloader for it.

It contains the following classes:
- OrnsteinUhlenbeckSDE: A class for the Ornstein-Uhlenbeck SDE.
- Data: A class object that contains the dataloader and other data-related functions.

It contains the following functions:
- create_dataloader: Creates a dataloader for the dataset.
- next: Returns the next batch of samples.
- get_fake_samples: Returns the denormalized fake samples with time channel.
- get_real_samples: Returns the denormalized real samples with time channel.
- geolife_dataloader: Custom dataloader for the Geolife dataset.
"""


import torch
import torchcde
import torchsde
import numpy as np
from .utils import get_data_csv, add_time_channel, remove_time_channel

class OrnsteinUhlenbeckSDE(torch.nn.Module):
    """
    Ornstein-Uhlenbeck SDE class.

    Args:
        mu: Mean of the SDE.
        theta: Speed of mean reversion.
        sigma: Volatility of the SDE.
        t_size: Time steps.
    """
    def __init__(self, mu, theta, sigma, t_size, sde_type='ito', noise_type='scalar'):
        super().__init__()
        self.register_buffer('mu', torch.as_tensor(mu))
        self.register_buffer('theta', torch.as_tensor(theta))
        self.register_buffer('sigma', torch.as_tensor(sigma))
        self.t_size = t_size
        self.sde_type = sde_type
        self.noise_type = noise_type

    def f(self, t, y):
        """Drift function of the Ornstein-Uhlenbeck SDE."""
        return self.mu * t - self.theta * y

    def g(self, t, y):
        """Diffusion function of the Ornstein-Uhlenbeck SDE."""
        return self.sigma.expand(y.size(0), 1, 1) * (2 * t / self.t_size)

class Data():
    """
    Data class.

    Args:
        cfg: Configuration object.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.mean = []
        self.std = []
        self.cols = []
        self.ts = torch.linspace(0, cfg.t_size - 1, cfg.t_size, device=cfg.device)

        self.data_size, self.dataloader = self.create_dataloader()
        self.infinite_train_dataloader = (elem for it in iter(lambda: self.dataloader, None) for elem in it)

    def create_dataloader(self):
        """Create a dataloader for the dataset."""
        if self.cfg.data_source == "ou_proc":
            # Generate Ornstein-Uhlenbeck process
            sde_gen = OrnsteinUhlenbeckSDE(mu=0.02, theta=0.1, sigma=0.4, t_size=self.cfg.t_size).to(self.cfg.device)
            y0 = torch.rand(self.cfg.dataset_size, device=self.cfg.device).unsqueeze(-1) * 2 - 1
            ts = torch.linspace(0, self.cfg.t_size - 1, self.cfg.t_size, device=self.cfg.device)
            ys = torchsde.sdeint(sde_gen, y0, ts, dt=1e-1) # 64, 8192, 1
        elif self.cfg.data_source == "geolife":
            # Custom data loading for Geolife dataset
            ys = self.geolife_dataloader(self.cfg.t_size)
        else:
            # Read from csv dataset
            df = get_data_csv(self.cfg.data_source)
            if self.cfg.data_col != "None":
                # If config selects only 1 column for data
                df = df[self.cfg.data_col]
            else:
                # Read all columns of data
                self.cols = df.columns.to_list()
            raw_data = df.to_numpy()

            # Ensure raw_data is always 2D: [T, num_channels]
            if raw_data.ndim == 1:
                raw_data = raw_data[:, None]

            # Sample windows, avoiding overlap when possible
            t_size = self.cfg.t_size
            if self.cfg.dataset_size * t_size <= raw_data.shape[0]:
                # Sample without overlap by selecting non-overlapping start indices
                available_starts = torch.arange(0, raw_data.shape[0] - t_size + 1, t_size)
                selected_starts = available_starts[torch.randperm(len(available_starts))[:self.cfg.dataset_size]]
            else:
                # Use all possible non-overlapping starts first
                non_overlap_starts = torch.arange(0, raw_data.shape[0] - t_size + 1, t_size)
                remaining = self.cfg.dataset_size - len(non_overlap_starts)
                overlap_starts = torch.randint(0, raw_data.shape[0] - t_size + 1, (remaining,))
                selected_starts = torch.cat([non_overlap_starts, overlap_starts], dim=0)

            # Generate data_tensor from the sampled data
            sampled_data = np.array([raw_data[i:i + t_size] for i in selected_starts])
            data_tensor = torch.tensor(sampled_data, dtype=torch.float32, device=self.cfg.device)

            # Drop NaNs
            mask = ~torch.isnan(data_tensor)
            data_tensor[~mask] = 0.0

            # Transpose tensor to desired shape: [t_size, dataset_size, channels] (no time channel)
            ys = data_tensor.transpose(0, 1)

        # Normalize with respect to t=0 for each channel
        y0 = ys[0]  # shape: [dataset_size, channels]
        mask = ~torch.isnan(y0)
        y0[~mask] = 0.0

        # Calculate mean and std
        mean = (y0.sum(dim=0) / mask.sum(dim=0)).detach()
        var = ((y0 - mean)**2 * mask).sum(dim=0) / mask.sum(dim=0)
        std = torch.sqrt(var + 1e-6)  # small epsilon to avoid divide by zero

        # Normalize data
        ys = (ys - mean.view(1, 1, -1)) / std.view(1, 1, -1)

        # Save per-channel std and mean to denormalize later
        self.std = std.tolist()
        self.mean = mean.tolist()

        # Add time channel
        ys = torch.cat([self.ts.unsqueeze(0).unsqueeze(-1).expand(ys.size(1), ys.size(0), 1),
                        ys.transpose(0, 1)], dim=2)

        # Package
        data_size = ys.size(-1) - 1
        ys_coeffs = torchcde.linear_interpolation_coeffs(ys)
        dataset = torch.utils.data.TensorDataset(ys_coeffs)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True)

        # Return channel count and dataloader
        return data_size, dataloader

    def next(self):
        """Return next batch of samples from the dataloader."""
        real_data, = next(self.infinite_train_dataloader)
        return real_data

    def get_fake_samples(self, generator):
        """Return denormalized fake samples with time channel from the generator."""
        # Generate fake data
        generator.eval()
        with torch.no_grad():
            fake_data = generator(self.ts, self.cfg.batch_size)
        generator.train()

        fake_data = remove_time_channel(fake_data)
        fake_data = torchcde.LinearInterpolation(fake_data).evaluate(self.ts)

        # Convert std and mean to tensors
        std = torch.tensor(self.std, dtype=fake_data.dtype, device=fake_data.device).view(1, 1, -1)
        mean = torch.tensor(self.mean, dtype=fake_data.dtype, device=fake_data.device).view(1, 1, -1)

        # Apply per-channel denormalization
        fake_samples = fake_data * std + mean

        # Return samples with time channel
        return add_time_channel(self.ts, fake_samples)

    def get_real_samples(self):
        """Return denormalized real samples with time channel from the dataloader."""
        # Get real data
        next = self.next()
        real_data = remove_time_channel(next)
        real_data = torchcde.LinearInterpolation(real_data).evaluate(self.ts)

        # Convert std and mean to tensors
        std = torch.tensor(self.std, dtype=real_data.dtype, device=real_data.device).view(1, 1, -1)
        mean = torch.tensor(self.mean, dtype=real_data.dtype, device=real_data.device).view(1, 1, -1)

        # Apply per-channel denormalization
        real_samples = real_data * std + mean

        # Return samples with time channel
        return add_time_channel(self.ts, real_samples)
    
    def geolife_dataloader(self, t_size):
        """Custom dataloader for the Geolife dataset."""
        if t_size > 100:
            raise ValueError("t_size cannot be greater than 100 for the Geolife dataset.")

        # Read city-split data and identify trajectory start indices
        df = get_data_csv(self.cfg.data_source)

        if self.cfg.data_col != "None":
            # If config selects only 1 column for data
            self.cols = [self.cfg.data_col]
        else:
            self.cols = ["latitude", "longitude", "altitude"]
        
        traj_groups = df.groupby("trajectory_id").groups
        valid_starts = [indices[0] for indices in traj_groups.values() if len(indices) >= t_size]
        selected_starts = np.random.choice(valid_starts, size=min(self.cfg.dataset_size, len(valid_starts)), replace=False)

        # Get data samples from trajectory starts
        sampled_data = []
        for start in selected_starts:
            traj_data = df.iloc[start:start + t_size][self.cols].to_numpy()
            sampled_data.append(traj_data)
        sampled_data = np.array(sampled_data)

        # Create and return data tensor
        data_tensor = torch.tensor(sampled_data, dtype=torch.float32, device=self.cfg.device)
        ys = data_tensor.transpose(0, 1)
        return ys
    