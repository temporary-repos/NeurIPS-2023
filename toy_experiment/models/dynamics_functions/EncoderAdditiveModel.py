"""
@file MetaNeuralODE.py

Holds the model for the Neural ODE latent dynamics function
"""
import torch
import torch.nn as nn

from torchdiffeq import odeint
from toy_experiment.models.CommonDynamics import DynamicsModel


class ODEFunction(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.dynamics_net = nn.Sequential(
            nn.Linear(args.latent_dim + args.control_dim, args.hidden_dim),
            nn.Tanh(),
            nn.Linear(args.hidden_dim, args.latent_dim),
            nn.Tanh()
        )

        nn.init.normal_(self.dynamics_net[0].weight, 0, 0.01)
        nn.init.zeros_(self.dynamics_net[0].bias)

        nn.init.normal_(self.dynamics_net[2].weight, 0, 0.01)
        nn.init.zeros_(self.dynamics_net[2].bias)

    def set_control(self, control):
        self.controls = control

    def forward(self, t, x):
        x = torch.concatenate((x, self.controls), dim=-1)
        return self.dynamics_net(x)


class AdditiveModel(DynamicsModel):
    def __init__(self, args, top, exptop):
        """ Latent dynamics as parameterized by a global deterministic neural ODE """
        super().__init__(args, top, exptop)
        self.encoder = nn.Linear(args.in_dim, args.latent_dim)
        self.decoder = nn.Linear(args.latent_dim, args.in_dim)

        # ODE-Net which holds mixture logic
        self.dynamics_func = ODEFunction(args)

    def forward(self, x, u, generation_len):
        """ Forward function of the ODE network """
        # Evaluate model forward over T to get L latent reconstructions
        t = torch.linspace(1, self.args.timesteps, self.args.timesteps, device='cuda')

        # Set controls for this batch
        self.dynamics_func.set_control(u)

        # Integrate and output
        pred = odeint(self.dynamics_func, self.encoder(x[:, 0]), t)
        pred = pred.permute([1, 0, 2])

        pred = self.decoder(pred.reshape([pred.shape[0] * pred.shape[1], -1]))
        pred = pred.reshape([x.shape[0], self.args.timesteps, self.args.in_dim])
        return pred
