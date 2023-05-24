"""
@file

"""
import torch
import functorch
import torch.nn as nn

from torchdiffeq import odeint

from switching_experiments.utils.layers import Flatten
from toy_experiment.models.CommonMetaDynamics import MetaDynamicsModel
from hypernn.torch.utils import FunctionalParamVectorWrapper, count_params


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
        dynamics_network = nn.Sequential(
            nn.Linear(args.latent_dim, args.hidden_dim),
            nn.Tanh(),
            nn.Linear(args.hidden_dim, args.latent_dim),
            nn.Tanh()
        )
        self.dynamics_network = FunctionalParamVectorWrapper(dynamics_network)

        # Domain encoder
        self.domain_encoder = nn.Sequential(
            Flatten(),
            nn.Linear(args.timesteps * args.in_dim, args.control_dim)
        )

        # Hypernetwork going from the embeddings to the full main-network weights
        self.hypernet = nn.Linear(args.control_dim * 2, count_params(dynamics_network))
        nn.init.normal_(self.hypernet.weight, 0, 0.01)
        nn.init.zeros_(self.hypernet.bias)

    def sample_weights(self, domains, controls):
        # Get the domain encoding and concatenate with control
        embeddings = self.domain_encoder(domains.reshape([domains.shape[0] * self.args.k_shot, -1, self.args.in_dim]))
        embeddings = embeddings.view([self.args.batch_size, self.args.k_shot, self.args.control_dim]).mean(dim=[1])
        embeddings = torch.concatenate((embeddings, controls), dim=-1)

        # Get weight outputs from hypernetwork
        self.params = self.hypernet(embeddings)

    def forward(self, t, z):
        """ Wrapper function for the odeint calculation """
        z = functorch.vmap(self.dynamics_network)(self.params, z)
        return z


class MetaMultiplicativeModel(MetaDynamicsModel):
    def __init__(self, args, top, exptop):
        super().__init__(args, top, exptop)
        self.encoder = nn.Linear(args.in_dim, args.latent_dim)
        self.decoder = nn.Linear(args.latent_dim, args.in_dim)

        # ODE-Net which holds mixture logic
        self.dynamics_func = ODE(args)

    def forward(self, x, d, u, generation_len):
        """ Forward function of the ODE network """
        # Evaluate model forward over T to get L latent reconstructions
        t = torch.linspace(1, self.args.timesteps, self.args.timesteps, device='cuda')

        # Set controls for this batch
        self.dynamics_func.sample_weights(d, u)

        # Integrate and output
        pred = odeint(self.dynamics_func, self.encoder(x[:, 0]), t)
        pred = pred.permute([1, 0, 2])

        pred = self.decoder(pred.reshape([pred.shape[0] * pred.shape[1], -1]))
        pred = pred.reshape([x.shape[0], self.args.timesteps, self.args.in_dim])
        return pred
