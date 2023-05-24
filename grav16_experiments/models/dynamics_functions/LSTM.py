"""
@file LSTM.py

Holds the model for the autoregressive LSTM latent dynamics function
"""
import torch
import torch.nn as nn

from models.CommonMetaDynamics import LatentMetaDynamicsModel


class LSTM(LatentMetaDynamicsModel):
    def __init__(self, args, top, exptop):
        super().__init__(args, top, exptop)

        # Recurrent dynamics function
        self.dynamics_func = nn.LSTMCell(input_size=args.latent_dim, hidden_size=args.num_hidden)
        self.dynamics_out = nn.Linear(args.num_hidden, args.latent_dim)
        self.dynamics_act = nn.Tanh()

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

        # Evaluate forward over timesteps by recurrently passing in output states
        zt = [z_init]
        for tidx in t[1:]:
            if tidx == 1:
                z_hid, c_hid = self.dynamics_func(z_init)
            else:
                z_hid, c_hid = self.dynamics_func(z, (z_hid, c_hid))

            z = self.dynamics_act(self.dynamics_out(z_hid))
            zt.append(z)

        # Stack zt and decode zts
        zt = torch.stack(zt)
        x_rec = self.decoder(zt)
        return x_rec, zt
