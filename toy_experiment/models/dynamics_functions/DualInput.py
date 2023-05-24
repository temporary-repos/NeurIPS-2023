"""
@file MetaNeuralODE.py

Holds the model for the Neural ODE latent dynamics function
"""
import torch
import torch.nn as nn

from torchdiffeq import odeint
from toy_experiment.utils.utils import get_act
from toy_experiment.models.CommonDynamics import DynamicsModel


class StandardODE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dynamics_net = nn.Sequential(
            nn.Linear(args.latent_dim, args.hidden_dim),
            get_act(args.latent_act),
            nn.Linear(args.hidden_dim, args.latent_dim),
            nn.LeakyReLU()
        )

    def forward(self, t, x):
        return self.dynamics_net(x)


class DualInputModel(DynamicsModel):
    def __init__(self, args, top, exptop):
        """ Latent dynamics as parameterized by a global deterministic neural ODE """
        super().__init__(args, top, exptop)
        self.args = args

        # Overwritten encoder to include the control
        self.encoder = nn.Linear(args.in_dim + args.control_dim, args.latent_dim)

        # ODE-Net which holds mixture logic
        self.dynamics_func = StandardODE(args)

    def forward(self, x, u, generation_len):
        """ Forward function of the ODE network """
        # Evaluate model forward over T to get L latent reconstructions
        t = torch.linspace(1, self.args.time_end, self.args.timesteps, device='cuda')

        # Encode x0 and u0 into z0
        inputs = self.encoder(torch.concatenate((x[:, 0], u), dim=-1))

        # Integrate and output
        pred = odeint(self.dynamics_func, inputs, t)
        pred = pred.permute([1, 0, 2])

        pred = self.decoder(pred.reshape([-1, self.args.latent_dim]))
        pred = pred.reshape([x.shape[0], -1, self.args.out_dim])
        return pred
