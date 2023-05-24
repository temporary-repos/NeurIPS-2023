"""
@file
"""
import torch
import functorch
import torch.nn as nn

from torchdiffeq import odeint
from pde_experiments.utils.utils import get_act
from hypernn.torch.utils import FunctionalParamVectorWrapper, count_params
from pde_experiments.models.CommonMetaDynamics import LatentMetaDynamicsModel


class ODE(nn.Module):
    def __init__(self, args):
        """
        Represents the MetaPrior in the Global case where a single set of distributional parameters are optimized
        in the metaspace.
        :param args: script arguments to use for initialization
        """
        super(ODE, self).__init__()
        self.args = args

        """ Main Network """
        dynamics_network = []
        dynamics_network.extend([
            nn.Linear(args.latent_dim, args.num_hidden),
            get_act(args.latent_act)
        ])

        for _ in range(args.num_layers - 1):
            dynamics_network.extend([
                nn.Linear(args.num_hidden, args.num_hidden),
                get_act(args.latent_act)
            ])

        dynamics_network.extend([nn.Linear(args.num_hidden, args.latent_dim)])
        dynamics_network = nn.Sequential(*dynamics_network)
        self.dynamics_network = FunctionalParamVectorWrapper(dynamics_network)

        """ Hypernetwork for ODE """
        # Global weights across tasks
        self.global_weights = torch.nn.Parameter(0.01 * torch.randn(count_params(dynamics_network)), requires_grad=True).float()

        # Task embeddings, dynamically added during the runtime if it doesn't exist
        self.codes = nn.ParameterDict({})

        # Hypernet to go from task embedding to adaptation weights
        self.hypernet = nn.Linear(args.code_dim, count_params(dynamics_network), bias=False)

    def sample_weights(self, labels):
        codes = []
        for label in labels:
            codes.append(self.codes[str(label.item())])

        # Get weight outputs from codes
        self.local_weights = self.hypernet(torch.vstack(codes))

        # Add params together in temp variable
        self.params = self.global_weights.unsqueeze(0).repeat(labels.shape[0], 1) + self.local_weights

    def forward(self, t, z):
        """ Wrapper function for the odeint calculation """
        return functorch.vmap(self.dynamics_network)(self.params, z)


class CoDA(LatentMetaDynamicsModel):
    def __init__(self, args, top, exptop):
        super().__init__(args, top, exptop)

        # ODE-Net which holds mixture logic
        self.dynamics_func = ODE(args)

    def forward(self, x, D, labels, generation_len, finetune=False):
        """ """
        # Evaluate model forward over T to get L latent reconstructions
        t = torch.linspace(0, generation_len - 1, generation_len).to(self.device)

        params = []
        for label in labels:
            if str(label.item()) not in self.dynamics_func.codes.keys():
                self.dynamics_func.codes[str(label.item())] = torch.nn.Parameter(0.01 * torch.randn([1, self.args.code_dim], device=self.args.gpus[0]), requires_grad=True).float()

                if finetune is True:
                    params.append(self.dynamics_func.codes[str(label.item())])
                else:
                    self.trainer.optimizers[0].add_param_group(
                        {'amsgrad': False, 'betas': (0.9, 0.999), 'eps': 1e-08, 'lr': self.args.learning_rate * 0.1,
                         'params': [self.dynamics_func.codes[str(label.item())]], 'weight_decay': 0}
                    )
        # Draw weights from codes
        self.dynamics_func.sample_weights(labels)

        # Evaluate forward over timestep
        x_rec = odeint(self.dynamics_func, x[:, 0], t, method='rk4')
        x_rec = x_rec.reshape([generation_len, x.shape[0], self.args.latent_dim])
        x_rec = x_rec.permute([1, 0, 2])

        if finetune is True:
            return x_rec, params
        else:
            return x_rec

    def model_specific_loss(self, x, domain, preds, train=True):
        """ A standard KL prior is put over the weight codes of the hyper-prior to encourage good latent structure """
        # L1,2 Norm for Hypernetwork weights
        row_reg = self.args.hypernet_reg_beta * (torch.norm(self.dynamics_func.hypernet.weight, dim=1)).sum()
        col_reg = self.args.code_reg_beta * (torch.norm(self.dynamics_func.hypernet.weight, dim=0)).sum()

        self.log("row_reg", row_reg, prog_bar=True, on_step=False, on_epoch=True)
        self.log("col_reg", col_reg, prog_bar=True, on_step=False, on_epoch=True)
        return row_reg + col_reg
