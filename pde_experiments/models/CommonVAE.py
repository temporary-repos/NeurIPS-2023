"""
@file CommonVAE.py

Holds the encoder/decoder architectures that are shared across the NSSM works
"""
import torch.nn as nn

from pde_experiments.utils.layers import Gaussian


class LinearDomainEncoder(nn.Module):
    def __init__(self, time_steps, num_hidden, num_channels, latent_dim, dim, stochastic=False):
        """
        Holds the rnn encoder that takes in a sequence of vectors and
        outputs the domain of the latent dynamics
        :param time_steps: how many GT steps are used in domain
        :param num_filters: base convolutional filters, upscaled by 2 every layer
        :param num_channels: how many image color channels there are
        :param latent_dim: dimension of the latent dynamics
        """
        super().__init__()
        self.time_steps = time_steps
        self.num_channels = num_channels
        self.stochastic = stochastic
        self.dim = dim

        # Encoder, q(z_0 | x_{0:time_steps})
        self.encoder = nn.RNN(
            input_size=self.dim, hidden_size=num_hidden, num_layers=1, batch_first=True
        )

        self.flatten = nn.Flatten()
        if stochastic:
            self.output = Gaussian(num_hidden * time_steps, latent_dim)
        else:
            self.output = nn.Linear(num_hidden * time_steps, latent_dim)

    def forward(self, x):
        """
        Handles getting the initial state given x and saving the distributional parameters
        :param x: input sequences [BatchSize, GenerationLen * NumChannels, H, W]
        :return: z0 over the batch [BatchSize, LatentDim]
        """
        z, _ = self.encoder(x)
        z = self.flatten(z)
        z = self.output(z)
        return z
        