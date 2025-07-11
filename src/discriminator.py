"""
DISCRIMINATOR.PY

This file contains the discriminator class for the Infinite GAN.
This code is taken from the "Neural SDEs as Infinite-Dimensional GANs" paper.

It contains the following classes:
- DiscriminatorFunc: A class for the discriminator function.
- Discriminator: A class for the discriminator.

It contains the following functions:
- forward: Forward pass of the discriminator.
"""


import torch
import torchcde
from .utils import MLP

class DiscriminatorFunc(torch.nn.Module):
    """
    Discriminator function class.

    Args:
        data_size: Size of the data.
        hidden_size: Size of the hidden layer.
        mlp_size: Size of the MLP.
        num_layers: Number of layers in the MLP.
    """
    def __init__(self, data_size, hidden_size, mlp_size, num_layers):
        super().__init__()
        self._data_size = data_size
        self._hidden_size = hidden_size

        # tanh is important for model performance
        self._module = MLP(1 + hidden_size, hidden_size * (1 + data_size), mlp_size, num_layers, tanh=True)

    def forward(self, t, h):
        # t has shape ()
        # h has shape (batch_size, hidden_size)
        t = t.expand(h.size(0), 1)
        th = torch.cat([t, h], dim=1)
        return self._module(th).view(h.size(0), self._hidden_size, 1 + self._data_size)


class Discriminator(torch.nn.Module):
    """
    Discriminator class.

    Args:
        data_size: Size of the data.
        hidden_size: Size of the hidden layer.
        mlp_size: Size of the MLP.
        num_layers: Number of layers in the MLP.
        d_dt: Time step.
    """
    def __init__(self, data_size, hidden_size, mlp_size, num_layers, d_dt):
        super().__init__()

        self._initial = MLP(1 + data_size, hidden_size, mlp_size, num_layers, tanh=False)
        self._func = DiscriminatorFunc(data_size, hidden_size, mlp_size, num_layers)
        self._readout = torch.nn.Linear(hidden_size, 1)

        self.h0 = None
        self.dt = d_dt

    def forward(self, ys_coeffs):
        # ys_coeffs has shape (batch_size, t_size, 1 + data_size)
        # The +1 corresponds to time. When solving CDEs, It turns out to be most natural to treat time as just another
        # channel: in particular this makes handling irregular data quite easy, when the times may be different between
        # different samples in the batch.

        Y = torchcde.LinearInterpolation(ys_coeffs)
        Y0 = Y.evaluate(Y.interval[0])
        self.h0 = self._initial(Y0)
        hs = torchcde.cdeint(Y, self._func, self.h0, Y.interval, method='reversible_heun', backend='torchsde', dt=self.dt,
                             adjoint_method='adjoint_reversible_heun',
                             adjoint_params=(ys_coeffs,) + tuple(self._func.parameters()))
        score = self._readout(hs[:, -1])
        # return score.mean()
        return score