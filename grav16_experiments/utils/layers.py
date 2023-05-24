"""
@file layers.py

Miscellaneous helper Torch layers
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Gaussian(nn.Module):
    def __init__(self, in_dim, out_dim, fix_variance=False):
        """
        Gaussian sample layer consisting of 2 simple linear layers.
        Can choose whether to fix the variance or let it be learned (training instability has been shown when learning).

        :param in_dim: input dimension (often a flattened latent embedding from a CNN)
        :param out_dim: output dimension
        :param fix_variance: whether to set the log-variance as a constant 0.1
        """
        super(Gaussian, self).__init__()
        self.fix_variance = fix_variance

        # Mean layer
        self.mu = nn.Sequential(
            nn.Linear(in_dim, out_dim)
        )

        # Log-var layer
        self.logvar = nn.Sequential(
            nn.Linear(in_dim, out_dim)
        )

        nn.init.zeros_(self.logvar[-1].weight)

    def reparameterize(self, mu, logvar):
        """ Reparameterization trick to get a sample from the output distribution """
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std)
        z = mu + (noise * std)
        return z

    def forward(self, x):
        """
        Forward function of the Gaussian layer. Handles getting the distributional parameters and sampling a vector
        :param x: input vector [BatchSize, InputDim]
        """
        # Get mu and logvar
        mu = self.mu(x)
        logvar = self.logvar(x)

        # Check on whether mu/logvar are getting out of normal ranges
        if (mu < -100).any() or (mu > 85).any() or (logvar < -100).any() or (logvar > 85).any():
            print("Explosion in mu/logvar. Mu {} Logvar {}".format(torch.mean(mu), torch.mean(logvar)))

        # Reparameterize and sample
        z = self.reparameterize(mu, logvar)
        return mu, logvar, z


class GroupSwish(nn.Module):
    def __init__(self, groups):
        super().__init__()
        # self.beta = nn.Parameter(torch.tensor([0.5 for _ in range(groups)]))
        self.silu = nn.SiLU()
        self.groups = groups

    def forward(self, x):
        n_ch_group = x.size(1) // self.groups
        t = x.shape[2:]
        x = x.reshape(-1, self.groups, n_ch_group, *t)
        # beta = self.beta.view(1, self.groups, 1, *[1 for _ in t])
        # return (x * torch.sigmoid_(x * F.softplus(beta))).div_(1.1).reshape(-1, self.groups * n_ch_group, *t)
        return self.silu(x).reshape(1, self.groups * n_ch_group, *t)


class GroupTanh(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.tanh = nn.Tanh()
        self.groups = groups

    def forward(self, x):
        n_ch_group = x.size(1) // self.groups
        t = x.shape[2:]

        # Reshape into ungrouped conv
        x = x.reshape(self.groups, n_ch_group, *t)

        # Perform batch normalization and reshape back into groups
        return self.tanh(x).reshape(1, self.groups * n_ch_group, *t)


class GroupBatchNorm(nn.Module):
    def __init__(self, groups, n_filters):
        super().__init__()
        self.bn = nn.BatchNorm1d(n_filters)
        self.groups = groups

    def forward(self, x):
        n_ch_group = x.size(1) // self.groups
        t = x.shape[2:]

        # Reshape into ungrouped conv
        x = x.reshape(self.groups, n_ch_group, *t)

        # Perform batch normalization and reshape back into groups
        return self.bn(x).reshape(1, self.groups * n_ch_group, *t)


class Flatten(nn.Module):
    def forward(self, input):
        """
        Handles flattening a Tensor within a nn.Sequential Block

        :param input: Torch object to flatten
        """
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, w):
        """
        Handles unflattening a vector into a 4D vector in a nn.Sequential Block

        :param w: width of the unflattened image vector
        """
        super().__init__()
        self.w = w

    def forward(self, input):
        nc = input[0].numel() // (self.w ** 2)
        return input.view(input.size(0), nc, self.w, self.w)
