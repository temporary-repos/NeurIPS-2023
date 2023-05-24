"""
@file MetaNeuralODE.py

Holds the model for the Neural ODE latent dynamics function
"""
import torch
import torch.nn as nn

from torchdiffeq import odeint
from switching_experiments.utils.layers import Flatten
from toy_experiment.models.CommonMetaDynamics import MetaDynamicsModel


class ODEFunction(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.dynamics_net = nn.Sequential(
            nn.Linear(args.latent_dim + (args.control_dim * 2), args.hidden_dim),
            nn.Tanh(),
            nn.Linear(args.hidden_dim, args.latent_dim),
            nn.Tanh()
        )

        # Domain encoder
        self.domain_encoder = nn.Sequential(
            Flatten(),
            nn.Linear(args.timesteps * args.in_dim, args.control_dim)
        )

    def set_control(self, domains, controls):
        # Get the domain encoding and concatenate with control
        embeddings = self.domain_encoder(domains.reshape([domains.shape[0] * self.args.k_shot, -1, self.args.in_dim]))
        embeddings = embeddings.view([self.args.batch_size, self.args.k_shot, self.args.control_dim]).mean(dim=[1])
        self.controls = torch.concatenate((embeddings, controls), dim=-1)

    def forward(self, t, x):
        x = torch.concatenate((x, self.controls), dim=-1)
        return self.dynamics_net(x)


class AdditiveModel(MetaDynamicsModel):
    def __init__(self, args, top, exptop):
        """ Latent dynamics as parameterized by a global deterministic neural ODE """
        super().__init__(args, top, exptop)
        self.encoder = nn.Linear(args.in_dim, args.latent_dim)
        self.decoder = nn.Linear(args.latent_dim, args.in_dim)

        # ODE-Net which holds mixture logic
        self.dynamics_func = ODEFunction(args)

    def forward(self, x, d, u, generation_len):
        """ Forward function of the ODE network """
        # Evaluate model forward over T to get L latent reconstructions
        t = torch.linspace(1, self.args.timesteps, self.args.timesteps, device='cuda')

        # Set controls for this batch
        self.dynamics_func.set_control(d, u)

        # Integrate and output
        pred = odeint(self.dynamics_func, self.encoder(x[:, 0]), t)
        pred = pred.permute([1, 0, 2])

        pred = self.decoder(pred.reshape([pred.shape[0] * pred.shape[1], -1]))
        pred = pred.reshape([x.shape[0], self.args.timesteps, self.args.in_dim])
        return pred
