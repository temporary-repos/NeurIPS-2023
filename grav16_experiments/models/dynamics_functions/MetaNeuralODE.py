"""
@file MetaNeuralODE.py

Holds the model for the Neural ODE latent dynamics function
"""
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from openTSNE import TSNE
from utils.utils import get_act
from torchdiffeq import odeint
from models.CommonVAE import LatentDomainEncoder
from models.CommonMetaDynamics import LatentMetaDynamicsModel


class ODEFunction(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Array that holds dimensions over hidden layers
        self.layers_dim = [args.code_dim + args.latent_dim] + args.num_layers * [args.num_hidden] + [args.latent_dim]

        # Condition variable on the NODE
        self.z_c = None

        # Domain encoder for z_c
        self.domain_encoder = LatentDomainEncoder(args.generation_training_len, args.num_filt, 1, args.code_dim, args.stochastic)

        # Build network layers
        self.acts = nn.ModuleList([])
        self.layers = nn.ModuleList([])
        for i, (n_in, n_out) in enumerate(zip(self.layers_dim[:-1], self.layers_dim[1:])):
            self.acts.append(get_act(args.latent_act) if i < args.num_layers else get_act('linear'))
            self.layers.append(nn.Linear(n_in, n_out, device=args.gpus[0]))

    def sample_domain(self, x, D):
        """ Given a batch of data points, embed them into their C representations """
        domain_size = D.shape[1]

        # Reshape to batch get the domain encodings
        D = D.reshape([D.shape[0] * domain_size, -1, self.args.dim, self.args.dim])

        # Get outputs over all domain samples over the batch
        if self.args.stochastic is True:
            self.embed_mus, self.embed_logvars, self.embeddings = self.domain_encoder(D)
        else:
            self.embeddings = self.domain_encoder(D)

        # Reshape to batch and take the average C over each sample
        self.embeddings = self.embeddings.view([self.args.batch_size, domain_size, self.args.code_dim])
        self.embeddings = self.embeddings.mean(dim=[1])

    def forward(self, t, x):
        x = torch.concatenate((x, self.embeddings), dim=1)
        for a, layer in zip(self.acts, self.layers):
            x = a(layer(x))
        return x


class MetaNeuralODE(LatentMetaDynamicsModel):
    def __init__(self, args, top, exptop):
        """ Latent dynamics as parameterized by a global deterministic neural ODE """
        super().__init__(args, top, exptop)

        # ODE-Net which holds mixture logic
        self.dynamics_func = ODEFunction(args)

    def forward(self, x, D, labels, generation_len):
        """
        Forward function of the ODE network
        :param x: data observation, which is a timeseries [BS, Timesteps, N Channels, Dim1, Dim2]
        :param generation_len: how many timesteps to generate over
        """
        # Sample z_init
        z_init = self.encoder(x)

        # Evaluate model forward over T to get L latent reconstructions
        t = torch.linspace(0, generation_len - 1, generation_len).to(self.device)

        # Draw function for index
        self.dynamics_func.sample_domain(x, D[:, :, :generation_len])

        # Evaluate forward over timestep
        zt = odeint(self.dynamics_func, z_init, t, method='rk4', options={'step_size': 0.5})
        zt = zt.permute([1, 0, 2])

        # Stack zt and decode zts
        x_rec = self.decoder(zt)
        return x_rec, zt

    def model_specific_plotting(self, version_path, outputs):
        # Get embeddings from outputs
        code_vectors, labels = [], []
        for output in outputs[:self.args.batches_to_save]:
            code_vectors.append(output['code_vectors'])
            labels.append(output['labels'])

        # Stack and convert to numpy arrays
        code_vectors, labels = torch.vstack(code_vectors), torch.vstack(labels)
        code_vectors = code_vectors.reshape([-1, self.args.code_dim])
        code_vectors = code_vectors.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        """ TSNE plot of C """
        # Generate TSNE embeddings of C
        tsne = TSNE(n_components=2, perplexity=30, initialization="pca", metric="cosine", n_jobs=8, random_state=3)
        tsne_embedding = tsne.fit(code_vectors)

        # Plot prototypes
        plt.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], c=labels)
        plt.title("t-SNE Plot of Code Embeddings")
        plt.legend(np.unique(labels))
        plt.savefig(f"lightning_logs/version_{self.top}/images/recon{self.current_epoch}tsne.png")
        plt.close()

    @staticmethod
    def get_model_specific_args():
        """ Add the hyperprior's arguments """
        return {
            'code_dim': 12,                 # Dimension of the weight codes
        }
