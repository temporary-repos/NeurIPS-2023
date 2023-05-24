"""
@file

"""
import torch
import functorch
import torch.nn as nn

from torchdiffeq import odeint
from toy_experiment.models.CommonDynamics import DynamicsModel
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
            nn.Linear(args.in_dim, args.hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(args.hidden_dim, args.in_dim),
            nn.LeakyReLU(0.1)
        )
        self.dynamics_network = FunctionalParamVectorWrapper(dynamics_network)

        # Hypernetwork going from the embeddings to the full main-network weights
        # self.hypernet = nn.Linear(args.control_dim, count_params(dynamics_network))
        self.hypernet = nn.Sequential(
            nn.Linear(args.control_dim, args.control_dim * 100),
            nn.LeakyReLU(0.1),
            nn.Linear(args.control_dim * 100, count_params(dynamics_network)),
            nn.Tanh(),
        )
        nn.init.normal_(self.hypernet[2].weight, 0, 0.05)
        nn.init.zeros_(self.hypernet[2].bias)

    def sample_weights(self, controls):
        # Get weight outputs from hypernetwork
        self.params = self.hypernet(controls)

    def forward(self, t, z):
        """ Wrapper function for the odeint calculation """
        z = functorch.vmap(self.dynamics_network)(self.params, z)
        return z


class MultiplicativeModel(DynamicsModel):
    def __init__(self, args, top, exptop):
        super().__init__(args, top, exptop)

        # ODE-Net which holds mixture logic
        self.dynamics_func = ODE(args)

    def forward(self, x, u, generation_len):
        """ Forward function of the ODE network """
        # Evaluate model forward over T to get L latent reconstructions
        t = torch.linspace(1, self.args.timesteps, self.args.timesteps, device='cuda')

        # Set controls for this batch
        self.dynamics_func.sample_weights(u)

        # Integrate and output
        pred = odeint(self.dynamics_func, x[:, 0], t, method='dopri5')
        pred = pred.permute([1, 0, 2])
        return pred
