import torch
import torchcde
import torchsde

from .utils import remove_ts, append_ts

'''
Load data
Generate and return dataloader
get_fake_samples >> denorm data
get_real_samples >> denorm data
'''

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
        self.ts = torch.linspace(0, cfg.t_size - 1, cfg.t_size, device=cfg.device)

        self.data_size, self.dataloader = self.create_dataloader()
        self.infinite_train_dataloader = (elem for it in iter(lambda: self.dataloader, None) for elem in it)

    def create_dataloader(self):
        # Return ts, num_features

        '''
        Get the data (OU, dataset)
        '''
        if self.cfg.data_source == "ou_proc":
            sde_gen = OrnsteinUhlenbeckSDE(mu=0.02, theta=0.1, sigma=0.4, t_size=self.cfg.t_size).to(self.cfg.device)
            y0 = torch.rand(self.cfg.dataset_size, device=self.cfg.device).unsqueeze(-1) * 2 - 1
            ts = torch.linspace(0, self.cfg.t_size - 1, self.cfg.t_size, device=self.cfg.device)
            ys = torchsde.sdeint(sde_gen, y0, ts, dt=1e-1)

        ###################
        # To demonstrate how to handle irregular data, then here we additionally drop some of the data (by setting it to
        # NaN.)
        ###################
        ys_num = ys.numel()
        to_drop = torch.randperm(ys_num)[:int(0.3 * ys_num)]
        ys.view(-1)[to_drop] = float('nan')

        ###################
        # Typically important to normalise data. Note that the data is normalised with respect to the statistics of the
        # initial data, _not_ the whole time series. This seems to help the learning process, presumably because if the
        # initial condition is wrong then it's pretty hard to learn the rest of the SDE correctly.
        ###################
        y0_flat = ys[0].view(-1)
        y0_not_nan = y0_flat.masked_select(~torch.isnan(y0_flat))
        mean = y0_not_nan.mean()
        std = y0_not_nan.std()
        ys = (ys - mean) / std
        self.mean.append(mean.item())
        self.std.append(std.item())

        ###################
        # As discussed, time must be included as a channel for the discriminator.
        ###################
        ys = torch.cat([self.ts.unsqueeze(0).unsqueeze(-1).expand(self.cfg.dataset_size, self.cfg.t_size, 1),
                        ys.transpose(0, 1)], dim=2)
        # shape (dataset_size=1000, t_size=100, 1 + data_size=3)

        ###################
        # Package up.
        ###################
        data_size = ys.size(-1) - 1  # How many channels the data has (not including time, hence the minus one).
        ys_coeffs = torchcde.linear_interpolation_coeffs(ys)  # as per neural CDEs.
        dataset = torch.utils.data.TensorDataset(ys_coeffs)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True)

        return data_size, dataloader

    def next(self):
        real_data, = next(self.infinite_train_dataloader)
        return real_data

    def get_fake_samples(self, generator):
        fake_samples = remove_ts(generator(self.ts, self.cfg.batch_size))
        # std = self.std.view(1, 1, -1)
        # mean = self.mean.view(1, 1, -1)
        std = self.std[0]
        mean = self.mean[0]
        return append_ts(self.ts, fake_samples * std + mean)

    # def get_real_samples(self):
    #     real_data = remove_ts(self.next())
    #     # std = self.std.view(1, 1, -1)
    #     # mean = self.mean.view(1, 1, -1)
    #     std = self.std[0]
    #     mean = self.mean[0]
    #     return append_ts(self.ts, real_data * std + mean)

    