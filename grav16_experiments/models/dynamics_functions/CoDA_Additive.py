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

from grav16_experiments.utils.utils import get_act


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

        # Build network layers
        self.acts = nn.ModuleList([])
        self.layers = nn.ModuleList([])
        for i, (n_in, n_out) in enumerate(zip(self.layers_dim[:-1], self.layers_dim[1:])):
            self.acts.append(get_act(args.latent_act) if i < args.num_layers else get_act('linear'))
            self.layers.append(nn.Linear(n_in, n_out))

        # Task embeddings, dynamically added during the runtime if it doesn't exist
        self.local_codes = nn.ParameterDict({})

    def get_embeddings(self, labels):
        codes = []
        for label in labels:
            codes.append(self.local_codes[str(label.item())])

        self.embeddings = torch.vstack(codes)

    def forward(self, t, x):
        x = torch.concatenate((x, self.embeddings), dim=1)
        for a, layer in zip(self.acts, self.layers):
            x = a(layer(x))
        return x


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

        params = []
        for label in labels:
            if str(label.item()) not in self.dynamics_func.local_codes.keys():
                self.dynamics_func.local_codes[str(label.item())] = torch.nn.Parameter(0.01 * torch.randn([1, self.args.code_dim], device=self.args.gpus[0]), requires_grad=True).float()

                if finetune is True:
                    params.append(self.dynamics_func.local_codes[str(label.item())])
                else:
                    self.trainer.optimizers[0].add_param_group(
                        {'amsgrad': False, 'betas': (0.9, 0.999), 'eps': 1e-08, 'lr': self.args.learning_rate * 0.1,
                         'params': [self.dynamics_func.local_codes[str(label.item())]], 'weight_decay': 0}
                    )

        # Draw weights
        self.dynamics_func.sample_weights(labels.long().flatten())

        # Evaluate forward over timestep
        zt = odeint(self.dynamics_func, z_init, t, method='rk4', options={'step_size': 0.5})
        zt = zt.reshape([generation_len, self.args.batch_size, self.args.latent_dim])
        zt = zt.permute([1, 0, 2])

        # Stack zt and decode zts
        x_rec = self.decoder(zt)
        if finetune is True:
            return x_rec, params
        else:
            return x_rec

    def model_specific_loss(self, x, domain, preds, train=True):
        """ A standard KL prior is put over the weight codes of the hyper-prior to encourage good latent structure """
        # Stack parameters
        embeds = torch.vstack([self.dynamics_func.local_codes[i] for i in self.dynamics_func.local_codes.keys()])

        # L1,2 Norm for Hypernetwork weights
        row_reg = self.args.hypernet_reg_beta * (torch.norm(embeds, dim=1)).sum()
        col_reg = self.args.code_reg_beta * (torch.norm(embeds, dim=0)).sum()

        self.log("row_reg", row_reg, prog_bar=True, on_step=False, on_epoch=True)
        self.log("col_reg", col_reg, prog_bar=True, on_step=False, on_epoch=True)
        return row_reg + col_reg

    @staticmethod
    def get_model_specific_args():
        """ Add the hyperprior's arguments """
        return {
            'hypernet_reg_beta': 1e-4,      # Beta for the hypernet L2-norm regularizer
            'code_reg_beta': 1e-2,          # Beta for the code L21-norm regularizer
            'code_dim': 12,                 # Dimension of the weight codes
        }