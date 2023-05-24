"""
@file

"""
import torch
import functorch
import torch.nn as nn
from torch import autograd

from torchdiffeq import odeint
from switching_experiments.utils.utils import get_act
from switching_experiments.utils.layers import Flatten
from switching_experiments.models.CommonMetaDynamics import LatentMetaDynamicsModel
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

        """ Hyper Network """
        # Domain encoder to take in the context samples and turn it into the context embeddings
        self.domain_encoder = nn.Sequential(
            Flatten(),
            nn.Linear(args.generation_training_len * args.dim, args.code_dim)
        )

        # Hypernetwork going from the embeddings to the full main-network weights
        self.hypernet = nn.Linear(args.code_dim, count_params(dynamics_network))
        nn.init.normal_(self.hypernet.weight, 0, 0.01)
        nn.init.zeros_(self.hypernet.bias)

        # Vector gradients at each timestep
        self.vector_gradients = []

    def sample_weights(self, D):
        # Clear out vector_gradients
        self.vector_gradients = []

        # Get outputs over all domain samples over the batch and take mean over domain samples
        self.embeddings = self.domain_encoder(D.reshape([D.shape[0] * self.args.domain_size, -1, self.args.dim]))
        self.embeddings = self.embeddings.view([self.args.batch_size, self.args.domain_size, self.args.code_dim]).mean(dim=[1])

        # Get weight outputs from hypernetwork
        self.params = self.hypernet(self.embeddings)

    def forward(self, t, z):
        """ Wrapper function for the odeint calculation """
        z = functorch.vmap(self.dynamics_network)(self.params, z)
        self.vector_gradients.append(z)
        return z


class MetaHyperNet(LatentMetaDynamicsModel):
    def __init__(self, args, top, exptop):
        super().__init__(args, top, exptop)

        # Encoder/Decoder
        if args.dim == args.latent_dim:
            self.encoder = nn.Identity()
            self.decoder = nn.Identity()
        else:
            self.encoder = nn.Linear(args.dim, args.latent_dim)
            self.decoder = nn.Linear(args.latent_dim, args.dim)

        # ODE-Net which holds mixture logic
        self.dynamics_func = ODE(args)

    def forward(self, x, D, labels, generation_len):
        """ TODO: Generate forward string """
        # Encode
        z_0 = self.encoder(x[:, 0])

        # Evaluate model forward over T to get L latent reconstructions
        t = torch.linspace(0, generation_len - 1, generation_len).to(self.device)

        # Draw function for index
        self.dynamics_func.sample_weights(D[:, :, :generation_len])

        # Evaluate forward over timestep
        z = odeint(self.dynamics_func, z_0, t,
                       # method='rk4', options={'step_size': 0.1}
                       method='dopri5'
        )
        z = z.reshape([generation_len, self.args.batch_size, self.args.latent_dim])
        z = z.permute([1, 0, 2])

        # Decode
        x_rec = self.decoder(z)
        return x_rec, z

    def model_specific_loss(self, x, domain, preds, is_train=True):
        """ A standard KL prior is put over the weight codes of the hyper-prior to encourage good latent structure """
        dynamics_loss = 0
        if is_train is False:
            return dynamics_loss

        # L2 norm for codes
        if self.args.code_reg_beta > 0:
            embeds = self.dynamics_func.embeddings.view([self.args.batch_size, -1])
            code_reg = (torch.norm(embeds, dim=1) ** 2)
            code_reg = self.args.code_reg_beta * code_reg.mean()
            self.log("code_reg", code_reg, prog_bar=True, on_step=True, on_epoch=False)
            dynamics_loss += code_reg

        # Gradient penalty on the hypernet
        if self.args.gradient_beta > 0:
            embeds = self.dynamics_func.embeddings.view([self.args.batch_size, -1])
            alpha = torch.rand_like(embeds, device=self.device)
            interpolates = (alpha * embeds + (1 - alpha) * embeds[torch.randperm(x.shape[0])]).requires_grad_(True)
            batch_interpolates = self.dynamics_func.hypernet(interpolates)
            fake = torch.full(batch_interpolates.shape, 1.0, device=self.device, requires_grad=False)

            gradients = autograd.grad(
                outputs=batch_interpolates,
                inputs=interpolates,
                grad_outputs=fake,
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )
            gradients = gradients[0].view(gradients[0].size(0), -1)
            gradient_penalty = self.args.gradient_beta * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
            self.log("gradient_reg", gradient_penalty, prog_bar=True, on_step=True, on_epoch=False)
            dynamics_loss += gradient_penalty

        return dynamics_loss

