"""
@file LEADS.py

Holds the model for the Neural ODE latent dynamics function
"""
import torch
import torch.nn as nn

from utils.layers import GroupSwish, GroupTanh
from utils.utils import get_act
from torchdiffeq import odeint
from models.CommonMetaDynamics import LatentMetaDynamicsModel


class ODEFunction(nn.Module):
    def __init__(self, args):
        """ Standard Neural ODE dynamics function """
        super(ODEFunction, self).__init__()
        self.args = args
        self.n_groups = self.args.batch_size

        # Array that holds dimensions over hidden layers
        self.layers_dim = [args.latent_dim] + args.num_layers * [args.num_hidden] + [args.latent_dim]

        # Define per-environment variables
        self.ws = []
        self.bs = []
        for idx in range(len(self.layers_dim) - 1):
            # Get weights and biases out
            self.ws.append(torch.nn.Parameter(0.01 * torch.randn([16, self.layers_dim[idx + 1], self.layers_dim[idx], 1]), requires_grad=True).float().to(args.gpus[0]))
            self.bs.append(torch.nn.Parameter(0.01 * torch.randn([16, self.layers_dim[idx + 1], ]), requires_grad=True).float().to(args.gpus[0]))

        # Build the grouped convolution dynamics network
        self.dynamics_network = nn.Sequential(
            nn.Conv1d(args.latent_dim * self.n_groups, args.num_hidden * self.n_groups, 1, groups=self.n_groups, bias=True),
            GroupSwish(self.n_groups),
            nn.Conv1d(args.num_hidden * self.n_groups, args.num_hidden * self.n_groups, 1, groups=self.n_groups, bias=True),
            GroupSwish(self.n_groups),
            nn.Conv1d(args.num_hidden * self.n_groups, args.latent_dim * self.n_groups, 1, groups=self.n_groups, bias=True),
            GroupTanh(self.n_groups)
        )

        # Build shared network layers
        self.acts = nn.ModuleList([])
        self.layers = nn.ModuleList([])
        for i, (n_in, n_out) in enumerate(zip(self.layers_dim[:-1], self.layers_dim[1:])):
            self.acts.append(get_act(args.latent_act) if i < args.num_layers else get_act('tanh'))
            self.layers.append(nn.Linear(n_in, n_out, device=args.gpus[0]))

    def sample_weights(self, labels):
        # Generate weight codes
        for idx in range(len(self.layers_dim) - 1):
            # Get weights and biases out
            w = self.ws[idx][labels].view([self.args.batch_size * self.layers_dim[idx + 1], self.layers_dim[idx], 1])
            b = self.bs[idx][labels].view([self.args.batch_size * self.layers_dim[idx + 1], ])

            # Copy over the generated weights into the parameters of the dynamics network
            if hasattr(self.dynamics_network[idx * 2], 'weight'):
                del self.dynamics_network[idx * 2].weight
            self.dynamics_network[idx * 2].weight = w

            if hasattr(self.dynamics_network[idx * 2], 'bias'):
                del self.dynamics_network[idx * 2].bias
            self.dynamics_network[idx * 2].bias = b

    def forward(self, t, x):
        """ Wrapper function for the odeint calculation """
        env_outs = self.dynamics_network(x.reshape([1, -1, 1])).view(self.args.batch_size, self.args.latent_dim)
        for a, layer in zip(self.acts, self.layers):
            x = a(layer(x))
        return x + env_outs


class LEADS(LatentMetaDynamicsModel):
    def __init__(self, args, top, exptop):
        """ Latent dynamics as parameterized by a global deterministic neural ODE """
        super().__init__(args, top, exptop)

        # ODE-Net which holds mixture logic
        self.dynamics_func = ODEFunction(args)

    def forward(self, x, domain, labels, generation_len):
        """
        Forward function of the ODE network
        :param x: data observation, which is a timeseries [BS, Timesteps, N Channels, Dim1, Dim2]
        :param generation_len: how many timesteps to generate over
        """
        # Sample z_init
        z_init = self.encoder(x)

        # Evaluate model forward over T to get L latent reconstructions
        t = torch.linspace(0, generation_len - 1, generation_len).to(self.device)

        # Set batch labels for environment selection in forward
        self.dynamics_func.sample_weights(labels.long().flatten())

        # Evaluate forward over timestep
        zt = odeint(self.dynamics_func, z_init, t, method='rk4', options={'step_size': 0.5})  # [T,q]
        zt = zt.permute([1, 0, 2])

        # Stack zt and decode zts
        x_rec = self.decoder(zt)
        return x_rec, zt

    @staticmethod
    def get_model_specific_args():
        """ Add model arguments  """
        return {
            "code_dim": 12      # Dimension of the weight codes
        }
