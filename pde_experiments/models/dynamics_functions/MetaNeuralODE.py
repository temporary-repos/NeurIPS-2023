"""
@file MetaNeuralODE.py

Holds the model for the Neural ODE latent dynamics function
"""
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from openTSNE import TSNE
from torchdiffeq import odeint
from pde_experiments.utils.utils import get_act
from pde_experiments.utils.layers import Flatten
from pde_experiments.models.CommonMetaDynamics import LatentMetaDynamicsModel


class ODEFunction(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Array that holds dimensions over hidden layers
        self.layers_dim = [args.code_dim + args.latent_dim] + args.num_layers * [args.num_hidden] + [args.latent_dim]

        # Condition variable on the NODE
        self.z_c = None

        # Domain encoder for weight codes
        self.domain_encoder = nn.Sequential(
            Flatten(),
            nn.Linear(args.generation_training_len * args.dim, args.code_dim)
        )

        # Build network layers
        self.acts = nn.ModuleList([])
        self.layers = nn.ModuleList([])
        for i, (n_in, n_out) in enumerate(zip(self.layers_dim[:-1], self.layers_dim[1:])):
            self.acts.append(get_act(args.latent_act) if i < args.num_layers else get_act('linear'))
            self.layers.append(nn.Linear(n_in, n_out))

    def sample_domain(self, x, D):
        """ Given a batch of data points, embed them into their C representations """
        domain_size = D.shape[1]
        D = D.reshape([D.shape[0] * domain_size, -1, self.args.dim])

        # Get outputs over all domain samples over the batch and take mean over domain samples
        self.embeddings = self.domain_encoder(D)
        self.embeddings = self.embeddings.view([self.args.batch_size, domain_size, self.args.code_dim]).mean(dim=[1])

    def forward(self, t, x):
        x = torch.concatenate((x, self.embeddings), dim=1)
        for a, layer in zip(self.acts, self.layers):
            x = a(layer(x))
        return x


class MetaNeuralODE(LatentMetaDynamicsModel):
    def __init__(self, args, top, exptop):
        """ Latent dynamics as parameterized by a global deterministic neural ODE """
        super().__init__(args, top, exptop)

        # ODE-Net which holds mixture logic
        self.dynamics_func = ODEFunction(args)

    def forward(self, x, D, labels, generation_len):
        """
        Forward function of the ODE network
        :param x: data observation, which is a timeseries [BS, Timesteps, N Channels, Dim1, Dim2]
        :param generation_len: how many timesteps to generate over
        """
        # Evaluate model forward over T to get L latent reconstructions
        t = torch.linspace(0, generation_len - 1, generation_len).to(self.device)

        # Draw function for index
        self.dynamics_func.sample_domain(x, D[:, :, :generation_len])

        # Evaluate forward over timestep
        x_rec = odeint(self.dynamics_func, x[:, 0], t, method='rk4')
        x_rec = x_rec.reshape([generation_len, self.args.batch_size, self.args.latent_dim])
        x_rec = x_rec.permute([1, 0, 2])
        return x_rec
