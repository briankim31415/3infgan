"""
GENERATOR.PY

This file contains the generator class for the Infinite GAN.
This code is taken from the "Neural SDEs as Infinite-Dimensional GANs" paper.

It contains the following classes:
- GeneratorFunc: A class for the generator function.
- Generator: A class for the generator.

It contains the following functions:
- forward: Forward pass of the generator.
"""


import torch
import torchcde
import torchsde
from .utils import MLP

class GeneratorFunc(torch.nn.Module):
    """
    Generator function class.

    Args:
        noise_size: Size of the noise.
        hidden_size: Size of the hidden layer.
        mlp_size: Size of the MLP.
        num_layers: Number of layers in the MLP.
    """
    sde_type = 'stratonovich'
    noise_type = 'general'
    def __init__(self, noise_size, hidden_size, mlp_size, num_layers):
        super().__init__()
        self._noise_size = noise_size
        self._hidden_size = hidden_size

        ###################
        # Drift and diffusion are MLPs. They happen to be the same size.
        # Note the final tanh nonlinearity: this is typically important for good performance, to constrain the rate of
        # change of the hidden state.
        # If you have problems with very high drift/diffusions then consider scaling these so that they squash to e.g.
        # [-3, 3] rather than [-1, 1].
        ###################
        self._drift = MLP(1 + hidden_size, hidden_size, mlp_size, num_layers, tanh=True)
        self._diffusion = MLP(1 + hidden_size, hidden_size * noise_size, mlp_size, num_layers, tanh=True)

    def f_and_g(self, t, x):
        # t has shape ()
        # x has shape (batch_size, hidden_size)
        t = t.expand(x.size(0), 1)
        tx = torch.cat([t, x], dim=1)
        return self._drift(tx), self._diffusion(tx).view(x.size(0), self._hidden_size, self._noise_size)


###################
# Now we wrap it up into something that computes the SDE.
###################
class Generator(torch.nn.Module):
    """
    Generator class.

    Args:
        data_size: Size of the data.
        initial_noise_size: Size of the initial noise.
        noise_size: Size of the noise.
        hidden_size: Size of the hidden layer.
        mlp_size: Size of the MLP.
        num_layers: Number of layers in the MLP.
        g_dt: Time step.
    """
    def __init__(self, data_size, initial_noise_size, noise_size, hidden_size, mlp_size, num_layers, g_dt):
        super().__init__()
        self._initial_noise_size = initial_noise_size
        self._hidden_size = hidden_size

        self._initial = MLP(initial_noise_size, hidden_size, mlp_size, num_layers, tanh=False)
        self._func = GeneratorFunc(noise_size, hidden_size, mlp_size, num_layers)
        self._readout = torch.nn.Linear(hidden_size, data_size)

        self.x0 = None
        self.dt = g_dt

    def forward(self, ts, batch_size):
        # ts has shape (t_size,) and corresponds to the points we want to evaluate the SDE at.

        ###################
        # Actually solve the SDE.
        ###################
        init_noise = torch.randn(batch_size, self._initial_noise_size, device=ts.device)
        self.x0 = self._initial(init_noise)

        ###################
        # We use the reversible Heun method to get accurate gradients whilst using the adjoint method.
        ###################
        xs = torchsde.sdeint_adjoint(self._func, self.x0, ts, method='reversible_heun', dt=self.dt,
                                     adjoint_method='adjoint_reversible_heun',)
        xs = xs.transpose(0, 1)
        ys = self._readout(xs)

        ###################
        # Normalise the data to the form that the discriminator expects, in particular including time as a channel.
        ###################
        ts = ts.unsqueeze(0).unsqueeze(-1).expand(batch_size, ts.size(0), 1)
        return torchcde.linear_interpolation_coeffs(torch.cat([ts, ys], dim=2))