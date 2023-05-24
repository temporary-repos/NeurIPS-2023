"""
@file MetaNeuralODE.py

Holds the model for the Neural ODE latent dynamics function
"""
import torch
import numpy as np
from scipy import stats
import torch.nn as nn
import matplotlib.pyplot as plt

from openTSNE import TSNE
from torchdiffeq import odeint
from switching_experiments.utils.layers import Flatten
from switching_experiments.utils.utils import get_act
from switching_experiments.models.CommonMetaDynamics import LatentMetaDynamicsModel


class ODEFunction(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Array that holds dimensions over hidden layers
        self.layers_dim = [args.code_dim + args.latent_dim] + args.num_layers * [args.num_hidden] + [args.latent_dim]

        # Condition variable on the NODE
        self.z_c = None

        # Domain encoder for weight codes
        self.domain_encoder = nn.Sequential(
            Flatten(),
            nn.Linear(args.generation_training_len * args.dim, args.code_dim)
        )

        # Vector gradients at each timestep
        self.vector_gradients = []

        # Build network layers
        self.acts = nn.ModuleList([])
        self.layers = nn.ModuleList([])
        for i, (n_in, n_out) in enumerate(zip(self.layers_dim[:-1], self.layers_dim[1:])):
            self.acts.append(get_act(args.latent_act) if i < args.num_layers else get_act('linear'))
            self.layers.append(nn.Linear(n_in, n_out))

    def sample_domain(self, D):
        """ Given a batch of data points, embed them into their C representations """
        domain_size = D.shape[1]
        D = D.reshape([D.shape[0] * domain_size, -1, self.args.dim])

        # Clear out vector_gradients
        self.vector_gradients = []

        # Get outputs over all domain samples over the batch and take mean over domain samples
        self.embeddings = self.domain_encoder(D)
        self.embeddings = self.embeddings.view([self.args.batch_size, domain_size, self.args.code_dim]).mean(dim=[1])

    def forward(self, t, x):
        x = torch.concatenate((x, self.embeddings), dim=1)
        for a, layer in zip(self.acts, self.layers):
            x = a(layer(x))

        self.vector_gradients.append(x)
        return x


class MetaNeuralODE(LatentMetaDynamicsModel):
    def __init__(self, args, top, exptop):
        """ Latent dynamics as parameterized by a global deterministic neural ODE """
        super().__init__(args, top, exptop)

        # Encoder/Decoder
        if args.dim == args.latent_dim:
            self.encoder = nn.Identity()
            self.decoder = nn.Identity()
        else:
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

        # Draw function for index
        self.dynamics_func.sample_domain(D[:, :, :generation_len])

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
        labels = stats.mode(labels, axis=1)[0].ravel()

        # Plot prototypes
        if self.args.latent_dim == 2:
            plt.scatter(code_vectors[:, 0], code_vectors[:, 1], c=labels)
            plt.title("t-SNE Plot of Code Embeddings")
            plt.legend(np.unique(labels))
            plt.savefig(f"lightning_logs/version_{self.top}/signals/recon{self.current_epoch}tsne.png")
            plt.close()

        elif self.args.latent_dim == 3:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(code_vectors[:, 0], code_vectors[:, 1], code_vectors[:, 2], c=labels)
            plt.legend(np.unique(labels))
            plt.savefig(f"lightning_logs/version_{self.top}/signals/recon{self.current_epoch}tsne.png")
            plt.close()

        else:
            """ TSNE plot of C """
            # Generate TSNE embeddings of C
            tsne = TSNE(n_components=2, perplexity=30, initialization="pca", metric="cosine", n_jobs=8, random_state=3)
            tsne_embedding = tsne.fit(code_vectors)

            # Plot prototypes
            plt.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1])
            plt.title("t-SNE Plot of Code Embeddings")
            plt.legend(np.unique(labels))
            plt.savefig(f"lightning_logs/version_{self.top}/signals/recon{self.current_epoch}tsne.png")
            plt.close()
