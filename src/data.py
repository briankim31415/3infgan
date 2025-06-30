import torch
import torchcde
import torchsde
import numpy as np

from .utils import get_data_csv, add_time_channel, remove_time_channel

class OrnsteinUhlenbeckSDE(torch.nn.Module):
    def __init__(self, mu, theta, sigma, t_size, sde_type='ito', noise_type='scalar'):
        super().__init__()
        self.register_buffer('mu', torch.as_tensor(mu))
        self.register_buffer('theta', torch.as_tensor(theta))
        self.register_buffer('sigma', torch.as_tensor(sigma))
        self.t_size = t_size
        self.sde_type = sde_type
        self.noise_type = noise_type

    def f(self, t, y):
        return self.mu * t - self.theta * y

    def g(self, t, y):
        return self.sigma.expand(y.size(0), 1, 1) * (2 * t / self.t_size)

class Data():
    def __init__(self, cfg):
        self.cfg = cfg
        self.mean = []
        self.std = []
        self.cols = []
        self.ts = torch.linspace(0, cfg.t_size - 1, cfg.t_size, device=cfg.device)

        self.data_size, self.dataloader = self.create_dataloader()
        self.infinite_train_dataloader = (elem for it in iter(lambda: self.dataloader, None) for elem in it)

    def create_dataloader(self):
        if self.cfg.data_source == "ou_proc":
            # Generate Ornstein-Uhlenbeck process
            sde_gen = OrnsteinUhlenbeckSDE(mu=0.02, theta=0.1, sigma=0.4, t_size=self.cfg.t_size).to(self.cfg.device)
            y0 = torch.rand(self.cfg.dataset_size, device=self.cfg.device).unsqueeze(-1) * 2 - 1
            ts = torch.linspace(0, self.cfg.t_size - 1, self.cfg.t_size, device=self.cfg.device)
            ys = torchsde.sdeint(sde_gen, y0, ts, dt=1e-1) # 64, 8192, 1
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
        """Return next batch of samples."""
        real_data, = next(self.infinite_train_dataloader)
        return real_data

    def get_fake_samples(self, generator):
        """Return denormalized fake samples with time channel."""
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
        """Return denormalized real samples with time channel."""
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

    