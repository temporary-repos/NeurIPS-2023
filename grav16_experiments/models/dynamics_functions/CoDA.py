"""
@file

We perform parallel MLP networks by utilizing a 1D Group Convolution stack over the batch size.
With appropriate kernel sizes and in/out groups, it acts functionally the same as a MLP however can be
parallelized across multiple networks per input.

Additionally, we manage the computation graph back through the convolution network by utilizing the
re-parameterization trick (not the VAE one) as described here:
@url{https://discuss.pytorch.org/t/assign-parameters-to-nn-module-and-have-grad-fn-track-it/62677}
"""
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from torchdiffeq import odeint
from functools import reduce

from models.CommonMetaDynamics import LatentMetaDynamicsModel
from utils.layers import Flatten, GroupSwish, GroupTanh


class ODE(nn.Module):
    def __init__(self, args):
        """
        Represents the MetaPrior in the Global case where a single set of distributional parameters are optimized
        in the metaspace.
        :param args: script arguments to use for initialization
        """
        super(ODE, self).__init__()

        # Parameters
        self.args = args
        self.latent_dim = args.latent_dim
        self.conv_dim = args.num_filt * 4 ** 2
        self.n_groups = self.args.batch_size

        # Array that holds dimensions over hidden layers
        self.layers_dim = [self.latent_dim] + args.num_layers * [args.num_hidden] + [self.latent_dim]
        self.total_dynamics_params = reduce(lambda x, y: x*y, self.layers_dim)

        """ Hypernetwork for ODE """
        # Task embeddings for grav16
        self.codes = torch.nn.Parameter(0.01 * torch.randn([16, args.code_dim], requires_grad=True).float())

        # Hypernet to go from task embedding to adaptation weights
        self.hypernet = nn.Linear(args.code_dim, self.total_dynamics_params, bias=False)

        # Shared task parameters
        self.shared_weights = []
        for idx in range(len(self.layers_dim) - 1):
            # Get weights and biases out
            self.shared_weights.append(torch.nn.Parameter(0.01 * torch.randn([self.layers_dim[idx + 1], self.layers_dim[idx], 1]), requires_grad=True).float().to(args.gpus[0]))

        # Build the grouped convolution dynamics network
        self.dynamics_network = nn.Sequential(
            nn.Conv1d(self.args.latent_dim * self.n_groups, self.args.num_hidden * self.n_groups, 1, groups=self.n_groups, bias=False),
            GroupSwish(self.n_groups),
            nn.Conv1d(self.args.num_hidden * self.n_groups, self.args.num_hidden * self.n_groups, 1, groups=self.n_groups, bias=False),
            GroupSwish(self.n_groups),
            nn.Conv1d(self.args.num_hidden * self.n_groups, self.args.latent_dim * self.n_groups, 1, groups=self.n_groups, bias=False),
            GroupTanh(self.n_groups)
        )

    def set_initial_codes(self):
        indices = np.unique(np.where(self.codes.data.cpu().numpy() > 1e-10)[0])

        self.codes.data = torch.mean(self.codes.data[indices], dim=0).unsqueeze(0).repeat(16, 1)

    def sample_weights(self, labels):
        # Get weight outputs from codes
        self.ws = self.hypernet(self.codes[labels])

        # Split the output vector per layer
        next_idx = 0
        for i in range(len(self.layers_dim) - 1):
            cur_idx = next_idx  # 0, 8, 108, 208
            next_idx += self.layers_dim[i] * self.layers_dim[i + 1]  # 8, 108, 208, 216

            # Get weight split and reshape to conv filters
            weights = self.ws[:, cur_idx:next_idx].reshape(
                [self.args.batch_size * self.layers_dim[i + 1], self.layers_dim[i], 1]
            )

            # Copy over the generated weights into the parameters of the dynamics network
            del self.dynamics_network[i * 2].weight
            self.dynamics_network[i * 2].weight = (weights + self.shared_weights[i].repeat(self.args.batch_size, 1, 1))

    def forward(self, t, z):
        """ Wrapper function for the odeint calculation """
        return self.dynamics_network(z)


class CoDA(LatentMetaDynamicsModel):
    def __init__(self, args, top, exptop):
        super().__init__(args, top, exptop)

        # ODE-Net which holds mixture logic
        self.dynamics_func = ODE(args)

    def forward(self, x, domain, labels, generation_len):
        """
        Forward function of the network that handles locally embedding the given sample into the C codes,
        generating the z posterior that defines mixture weightings, and finding the winning components for each sample.
        :param x: data observation, which is a timeseries [BS, Timesteps, N Channels, Dim1, Dim2]
        :param generation_len: how many timesteps to generate over
        :return: reconstructions of the trajectory and generation
        """
        # Sample z_init
        z_init = self.encoder(x).reshape([1, -1, 1])

        # Evaluate model forward over T to get L latent reconstructions
        t = torch.linspace(0, generation_len - 1, generation_len).to(self.device)

        # Draw weights
        self.dynamics_func.sample_weights(labels.long().flatten())

        # Evaluate forward over timestep
        zt = odeint(self.dynamics_func, z_init, t, method='rk4', options={'step_size': 0.5})
        zt = zt.reshape([generation_len, self.args.batch_size, self.args.latent_dim])
        zt = zt.permute([1, 0, 2])

        # Stack zt and decode zts
        x_rec = self.decoder(zt)
        return x_rec, zt

    def model_specific_loss(self, x, domain, train=True):
        """ A standard KL prior is put over the weight codes of the hyper-prior to encourage good latent structure """
        # L1 Norm for Hypernetwork weights
        hypernet_reg = self.args.hypernet_reg_beta * (torch.norm(self.dynamics_func.hypernet.weight, dim=1)).sum()

        # L2 norm for codes
        code_reg = self.args.code_reg_beta * (torch.norm(self.dynamics_func.codes, dim=1) ** 2).mean()

        self.log("hyper_reg", hypernet_reg, prog_bar=True, on_step=False, on_epoch=True)
        self.log("code_reg", code_reg, prog_bar=True, on_step=False, on_epoch=True)
        return hypernet_reg + code_reg

    @staticmethod
    def get_model_specific_args():
        """ Add the hyperprior's arguments """
        return {
            'hypernet_reg_beta': 1e-4,      # Beta for the hypernet L2-norm regularizer
            'code_reg_beta': 1e-2,          # Beta for the code L21-norm regularizer
            'code_dim': 12,                 # Dimension of the weight codes
        }