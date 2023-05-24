"""
@file MetaNeuralODE.py

Holds the model for the Neural ODE latent dynamics function
"""
import torch
import torch.nn as nn

from torchdiffeq import odeint
from switching_experiments.utils.utils import get_act
from switching_experiments.models.CommonMetaDynamics import LatentMetaDynamicsModel


class ODEFunction(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Array that holds dimensions over hidden layers
        self.layers_dim = [args.latent_dim] + args.num_layers * [args.num_hidden] + [args.latent_dim]

        # Vector gradients at each timestep
        self.vector_gradients = []

        # Build network layers
        self.acts = nn.ModuleList([])
        self.layers = nn.ModuleList([])
        for i, (n_in, n_out) in enumerate(zip(self.layers_dim[:-1], self.layers_dim[1:])):
            self.acts.append(get_act(args.latent_act) if i < args.num_layers else get_act('linear'))
            self.layers.append(nn.Linear(n_in, n_out))

    def forward(self, t, x):
        for a, layer in zip(self.acts, self.layers):
            x = a(layer(x))

        self.vector_gradients.append(x)
        return x


class NeuralODE(LatentMetaDynamicsModel):
    def __init__(self, args, top, exptop):
        """ Latent dynamics as parameterized by a global deterministic neural ODE """
        super().__init__(args, top, exptop)

        # Encoder/Decoder
        self.encoder = nn.Linear(args.dim, args.latent_dim)
        self.decoder = nn.Linear(args.latent_dim, args.dim)

        # ODE-Net which holds mixture logic
        self.dynamics_func = ODEFunction(args)

    def forward(self, x, D, labels, generation_len):
        """
        Forward function of the ODE network
        :param x: data observation, which is a timeseries [BS, Timesteps, N Channels, Dim1, Dim2]
        :param generation_len: how many timesteps to generate over
        """
        # Encode
        z_0 = self.encoder(x[:, 0])

        # Evaluate model forward over T to get L latent reconstructions
        t = torch.linspace(0, generation_len - 1, generation_len).to(self.device)

        # Evaluate forward over timestep
        self.dynamics_func.vector_gradients = []
        z = odeint(self.dynamics_func, z_0, t,
                       # method='rk4', options={'step_size': 0.5}
                       method='dopri5'
                       )
        z = z.reshape([generation_len, self.args.batch_size, self.args.latent_dim])
        z = z.permute([1, 0, 2])

        # Decode
        x_rec = self.decoder(z)
        return x_rec, z
