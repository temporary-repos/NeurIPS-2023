"""
@file CommonVAE.py

Holds the encoder/decoder architectures that are shared across the NSSM works
"""
import torch
import torch.nn as nn

from utils.layers import Flatten, Gaussian, UnFlatten
from torch.distributions import Normal, kl_divergence as kl


class LatentStateEncoder(nn.Module):
    def __init__(self, z_amort, num_filters, num_channels, latent_dim, stochastic):
        """
        Holds the convolutional encoder that takes in a sequence of images and outputs the
        initial state of the latent dynamics
        :param z_amort: how many GT steps are used in initialization
        :param num_filters: base convolutional filters, upscaled by 2 every layer
        :param num_channels: how many image color channels there are
        :param latent_dim: dimension of the latent dynamics
        """
        super(LatentStateEncoder, self).__init__()
        self.z_amort = z_amort
        self.num_channels = num_channels
        self.stochastic = stochastic

        # Encoder, q(z_0 | x_{0:z_amort})
        self.initial_encoder = nn.Sequential(
            nn.Conv2d(z_amort, num_filters, kernel_size=5, stride=2, padding=(2, 2)),  # 14,14
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.Conv2d(num_filters, num_filters * 2, kernel_size=5, stride=2, padding=(2, 2)),  # 7,7
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU(),
            nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=5, stride=2, padding=(2, 2)),
            nn.BatchNorm2d(num_filters * 4),
            nn.ReLU(),
            nn.AvgPool2d(4),
            Flatten()
        )

        if self.stochastic is True:
            self.initial_encoder_out = Gaussian(num_filters * 4, latent_dim)
        else:
            self.initial_encoder_out = nn.Linear(num_filters * 4, latent_dim)

        self.out_act = nn.Tanh()

        # Holds generated z0 means and logvars for use in KL calculations
        self.z_means = None
        self.z_logvs = None

    def kl_z_term(self):
        """
        KL Z term, KL[q(z0|X) || N(0,1)]
        :return: mean klz across batch
        """
        if self.stochastic is True:
            batch_size = self.z_means.shape[0]
            mus, logvars = self.z_means.view([-1]), self.z_logvs.view([-1])  # N, 2

            q = Normal(mus, torch.exp(0.5 * logvars))
            N = Normal(torch.zeros(len(mus), device=mus.device),
                       torch.ones(len(mus), device=mus.device))

            klz = kl(q, N).view([batch_size, -1]).sum([1]).mean()
        else:
            klz = 0.0

        return klz

    def forward(self, x):
        """
        Handles getting the initial state given x and saving the distributional parameters
        :param x: input sequences [BatchSize, GenerationLen * NumChannels, H, W]
        :return: z0 over the batch [BatchSize, LatentDim]
        """
        if self.stochastic is True:
            self.z_means, self.z_logvs, z0 = self.initial_encoder_out(self.initial_encoder(x[:, :self.z_amort]))
        else:
            z0 = self.initial_encoder_out(self.initial_encoder(x[:, :self.z_amort]))
        return self.out_act(z0)


class LinearStateEncoder(nn.Module):
    def __init__(self, z_amort, num_hidden, num_channels, latent_dim):
        """
        Holds the MLP encoder that takes in a sequence of states and outputs the
        initial state of the latent dynamics
        :param z_amort: how many GT steps are used in initialization
        :param num_hidden: how many neural units to use
        :param num_channels: how many state dimensions there are
        :param latent_dim: dimension of the latent dynamics
        """
        super().__init__()
        self.z_amort = z_amort
        self.num_channels = num_channels

        # Encoder, q(z_0 | x_{0:z_amort})
        self.initial_encoder = nn.Sequential(
            nn.Linear(z_amort * num_channels, num_hidden),
            #nn.BatchNorm1d(num_hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(num_hidden, num_hidden),
            # nn.BatchNorm1d(num_hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(num_hidden, latent_dim),
            # nn.Tanh()
        )

        # Holds generated z0 means and logvars for use in KL calculations
        self.z_means = None
        self.z_logvs = None

    def kl_z_term(self):
        """
        KL Z term, KL[q(z0|X) || N(0,1)]
        :return: mean klz across batch
        """
        return 0.0

    def forward(self, x):
        """
        Handles getting the initial state given x and saving the distributional parameters
        :param x: input sequences [BatchSize, GenerationLen * NumChannels, H, W]
        :return: z0 over the batch [BatchSize, LatentDim]
        """
        x = x[:, :self.z_amort].reshape([x.shape[0], -1])
        z0 = self.initial_encoder(x)
        return z0


class EmissionDecoder(nn.Module):
    def __init__(self, batch_size, generation_len, dim, num_filters, num_channels, latent_dim):
        """
        Holds the convolutional decoder that takes in a batch of individual latent states and
        transforms them into their corresponding data space reconstructions
        """
        super(EmissionDecoder, self).__init__()
        self.batch_size = batch_size
        self.generation_len = generation_len
        self.dim = dim
        self.num_channels = num_channels

        # Variable that holds the estimated output for the flattened convolution vector
        self.conv_dim = num_filters * 4 ** 3

        # Emission model handling z_i -> x_i
        self.decoder = nn.Sequential(
            # Transform latent vector into 4D tensor for deconvolution
            nn.Linear(latent_dim, self.conv_dim),
            UnFlatten(4),

            # Perform de-conv to output space
            nn.ConvTranspose2d(self.conv_dim // 16, num_filters * 4, kernel_size=4, stride=1, padding=(0, 0)),
            nn.BatchNorm2d(num_filters * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(num_filters * 4, num_filters * 2, kernel_size=5, stride=2, padding=(1, 1)),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(num_filters * 2, num_filters, kernel_size=5, stride=2, padding=(1, 1), output_padding=(1, 1)),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.ConvTranspose2d(num_filters, 1, kernel_size=5, stride=1, padding=(2, 2)),
            nn.Sigmoid(),
        )

        # nn.init.normal_(self.decoder[-2].weight, -0.05, 0.01)
        # nn.init.constant_(self.decoder[-2].bias, 0)

    def forward(self, zts):
        """
        Handles decoding a batch of individual latent states into their corresponding data space reconstructions
        :param zts: latent states [BatchSize * GenerationLen, LatentDim]
        :return: data output [BatchSize, GenerationLen, NumChannels, H, W]
        """
        # Flatten to [BS * SeqLen, -1]
        zts = zts.contiguous().view([zts.shape[0] * zts.shape[1], -1])

        # Decode back to image space
        x_rec = self.decoder(zts)

        # Reshape to image output
        x_rec = x_rec.view([self.batch_size, x_rec.shape[0] // self.batch_size, self.dim, self.dim])
        return x_rec


class LinearDecoder(nn.Module):
    def __init__(self, batch_size, generation_len, dim, num_hidden, latent_dim):
        """
        Holds the convolutional decoder that takes in a batch of individual latent states and
        transforms them into their corresponding data space reconstructions
        """
        super(LinearDecoder, self).__init__()
        self.batch_size = batch_size
        self.generation_len = generation_len
        self.dim = dim

        # Emission model handling z_i -> x_i
        self.decoder = nn.Sequential(
            # Transform latent vector into 4D tensor for deconvolution
            nn.Linear(latent_dim, num_hidden),
            #nn.BatchNorm1d(num_hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(num_hidden, num_hidden * 2),
            #nn.BatchNorm1d(num_hidden * 2),
            nn.LeakyReLU(0.1),
            nn.Linear(num_hidden * 2, self.dim)
        )

    def forward(self, zts):
        """
        Handles decoding a batch of individual latent states into their corresponding data space reconstructions
        :param zts: latent states [BatchSize * GenerationLen, LatentDim]
        :return: data output [BatchSize, GenerationLen, NumChannels, H, W]
        """
        # Flatten to [BS * SeqLen, -1]
        zts = zts.contiguous().view([zts.shape[0] * zts.shape[1], -1])

        # Decode back to image space
        x_rec = self.decoder(zts)

        # Reshape to image output
        x_rec = x_rec.view([self.batch_size, x_rec.shape[0] // self.batch_size, self.dim])
        return x_rec


class SpatialTemporalBlock(nn.Module):
    def __init__(self, t_in, t_out, n_in, n_out, num_channels, last):
        super().__init__()
        self.t_in = t_in
        self.t_out = t_out
        self.n_in = n_in
        self.n_out = n_out
        self.num_channels = num_channels
        self.last = last

        self.conv = nn.Conv2d(n_in, n_out, kernel_size=5, stride=2, padding=(2, 2))
        self.bn = nn.BatchNorm2d(n_out)
        self.act = nn.ReLU()
        self.lin_t = nn.Linear(t_in, t_out)

        if last:
            self.act_last = nn.Tanh()

    def forward(self, x):
        B, _, _, H_in, W_in = x.shape
        x = x.contiguous()
        x = x.view(B * self.t_in, self.n_in, H_in, W_in)
        x = self.act(self.bn(self.conv(x)))
        # x = self.act(self.conv(x))
        H_out, W_out = x.shape[2], x.shape[3]
        x = x.view(B, self.t_in, self.n_out, H_out, W_out)

        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.lin_t(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        if self.last:
            x = self.act_last(x)
            x = x.view(B, -1, H_out, W_out)
        else:
            x = self.act(x)
        return x


class LatentDomainEncoder(nn.Module):
    def __init__(self, time_steps, num_filters, num_channels, latent_dim, stochastic=False):
        """
        Holds the convolutional encoder that takes in a sequence of images and
        outputs the domain of the latent dynamics
        :param time_steps: how many GT steps are used in domain
        :param num_filters: base convolutional filters, upscaled by 2 every layer
        :param num_channels: how many image color channels there are
        :param latent_dim: dimension of the latent dynamics
        """
        super().__init__()
        self.time_steps = time_steps
        self.num_channels = num_channels
        self.stochastic = stochastic

        # Encoder, q(z_0 | x_{0:time_steps})
        self.encoder = nn.Sequential(
            SpatialTemporalBlock(time_steps, time_steps // 2, 1, num_filters, num_channels, False),
            SpatialTemporalBlock(time_steps // 2, time_steps // 4, num_filters, num_filters * 2, num_channels, False),
            SpatialTemporalBlock(time_steps // 4, 1, num_filters * 2, num_filters, num_channels, True),
            Flatten()
        )
        if stochastic:
            self.output = Gaussian(num_filters * 4 ** 2, latent_dim)
        else:
            self.output = nn.Linear(num_filters * 4 ** 2, latent_dim)

    def forward(self, x):
        """
        Handles getting the initial state given x and saving the distributional parameters
        :param x: input sequences [BatchSize, GenerationLen * NumChannels, H, W]
        :return: z0 over the batch [BatchSize, LatentDim]
        """
        B, _, H, W = x.shape
        x = x.view(B, self.time_steps, self.num_channels, H, W)
        z = self.encoder(x)
        if self.stochastic:
            mu, var, z = self.output(z)
            return mu, var, z
        else:
            z = self.output(z)
            return z


""" Shallow-Water Encoder/Decoders """
class LatentStateEncoderSWE(nn.Module):
    def __init__(self, z_amort, num_filters, num_channels, latent_dim, stochastic):
        """
        Holds the convolutional encoder that takes in a sequence of images and outputs the
        initial state of the latent dynamics
        :param z_amort: how many GT steps are used in initialization
        :param num_filters: base convolutional filters, upscaled by 2 every layer
        :param num_channels: how many image color channels there are
        :param latent_dim: dimension of the latent dynamics
        """
        super(LatentStateEncoderSWE, self).__init__()
        self.z_amort = z_amort
        self.num_channels = num_channels
        self.stochastic = stochastic

        # Encoder, q(z_0 | x_{0:time_steps})
        self.initial_encoder = nn.Sequential(
            nn.Conv2d(z_amort * num_channels, num_filters, kernel_size=3, stride=2, padding=(1, 1)),  # 14,14
            nn.BatchNorm2d(num_filters),
            nn.LeakyReLU(0.1),
            nn.Conv2d(num_filters, num_filters * 2, kernel_size=3, stride=2, padding=(1, 1)),  # 7,7
            nn.BatchNorm2d(num_filters * 2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=3, stride=2, padding=(1, 1)),  # 7,7
            nn.BatchNorm2d(num_filters * 4),
            nn.LeakyReLU(0.1),
            nn.Conv2d(num_filters * 4, num_filters, kernel_size=3, stride=2, padding=(1, 1)),
            Flatten(),
        )

        self.initial_encoder_out = nn.Linear(num_filters * 4 ** 2, latent_dim)
        self.out_act = nn.Tanh()

        # Holds generated z0 means and logvars for use in KL calculations
        self.z_means = None
        self.z_logvs = None

    def kl_z_term(self):
        """
        KL Z term, KL[q(z0|X) || N(0,1)]
        :return: mean klz across batch
        """
        klz = 0.0
        return klz

    def forward(self, x):
        """
        Handles getting the initial state given x and saving the distributional parameters
        :param x: input sequences [BatchSize, GenerationLen * NumChannels, H, W]
        :return: z0 over the batch [BatchSize, LatentDim]
        """
        z0 = self.initial_encoder_out(self.initial_encoder(x[:, :self.z_amort]))
        return self.out_act(z0)


class EmissionDecoderSWE(nn.Module):
    def __init__(self, batch_size, generation_len, dim, num_filters, num_channels, latent_dim):
        """
        Holds the convolutional decoder that takes in a batch of individual latent states and
        transforms them into their corresponding data space reconstructions
        """
        super(EmissionDecoderSWE, self).__init__()
        self.batch_size = batch_size
        self.generation_len = generation_len
        self.dim = dim
        self.num_channels = num_channels

        # Variable that holds the estimated output for the flattened convolution vector
        self.conv_dim = num_filters * 4 ** 3

        # Emission model handling z_i -> x_i
        self.decoder = nn.Sequential(
            # Transform latent vector into 4D tensor for deconvolution
            nn.Linear(latent_dim, self.conv_dim),
            nn.LeakyReLU(0.1),
            UnFlatten(4),

            # Perform de-conv to output space
            nn.ConvTranspose2d(self.conv_dim // 16, num_filters * 8, kernel_size=4, stride=2, padding=(1, 1)),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(num_filters * 8, num_filters * 4, kernel_size=4, stride=2, padding=(1, 1)),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(num_filters * 4, num_filters * 2, kernel_size=4, stride=2, padding=(1, 1)),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(num_filters * 2, num_channels, kernel_size=4, stride=2, padding=(1, 1)),
            nn.LeakyReLU(0.1),
        )

    def forward(self, zts):
        """
        Handles decoding a batch of individual latent states into their corresponding data space reconstructions
        :param zts: latent states [BatchSize * GenerationLen, LatentDim]
        :return: data output [BatchSize, GenerationLen, NumChannels, H, W]
        """
        # Flatten to [BS * SeqLen, -1]
        zts = zts.contiguous().view([zts.shape[0] * zts.shape[1], -1])

        # Decode back to image space
        x_rec = self.decoder(zts)

        # Reshape to image output
        x_rec = x_rec.view([self.batch_size, x_rec.shape[0] // self.batch_size, self.dim, self.dim])
        return x_rec


class LatentDomainEncoderSWE(nn.Module):
    def __init__(self, time_steps, num_filters, num_channels, latent_dim, stochastic=False):
        """
        Holds the convolutional encoder that takes in a sequence of images and
        outputs the domain of the latent dynamics
        :param time_steps: how many GT steps are used in domain
        :param num_filters: base convolutional filters, upscaled by 2 every layer
        :param num_channels: how many image color channels there are
        :param latent_dim: dimension of the latent dynamics
        """
        super().__init__()
        self.time_steps = time_steps
        self.num_channels = num_channels
        self.stochastic = stochastic

        # Encoder, q(z_0 | x_{0:time_steps})
        self.encoder = nn.Sequential(
            SpatialTemporalBlock(time_steps, time_steps // 2, num_channels, num_filters, num_channels, False),
            SpatialTemporalBlock(time_steps // 2, time_steps // 4, num_filters, num_filters * 2, num_channels, False),
            SpatialTemporalBlock(time_steps // 4, time_steps // 8, num_filters * 2, num_filters * 4, num_channels, False),
            SpatialTemporalBlock(time_steps // 8, 1, num_filters * 4, num_filters, num_channels, True),
            Flatten()
        )
        self.output = nn.Linear(num_filters * 4 ** 2, latent_dim)

    def forward(self, x):
        """
        Handles getting the initial state given x and saving the distributional parameters
        :param x: input sequences [BatchSize, GenerationLen * NumChannels, H, W]
        :return: z0 over the batch [BatchSize, LatentDim]
        """
        B, _, H, W = x.shape
        x = x.view(B, self.time_steps, self.num_channels, H, W)
        z = self.encoder(x)
        z = self.output(z)
        return z


class LinearDomainEncoder(nn.Module):
    def __init__(self, time_steps, num_hidden, num_channels, latent_dim, dim, stochastic=False):
        """
        Holds the rnn encoder that takes in a sequence of vectors and
        outputs the domain of the latent dynamics
        :param time_steps: how many GT steps are used in domain
        :param num_filters: base convolutional filters, upscaled by 2 every layer
        :param num_channels: how many image color channels there are
        :param latent_dim: dimension of the latent dynamics
        """
        super().__init__()
        self.time_steps = time_steps
        self.num_channels = num_channels
        self.stochastic = stochastic
        self.dim = dim

        # Encoder, q(z_0 | x_{0:time_steps})
        self.encoder = nn.RNN(
            input_size=self.dim, hidden_size=num_hidden, num_layers=1, batch_first=True
        )

        self.flatten = nn.Flatten()
        if stochastic:
            self.output = Gaussian(num_hidden * time_steps, latent_dim)
        else:
            self.output = nn.Linear(num_hidden * time_steps, latent_dim)

    def forward(self, x):
        """
        Handles getting the initial state given x and saving the distributional parameters
        :param x: input sequences [BatchSize, GenerationLen * NumChannels, H, W]
        :return: z0 over the batch [BatchSize, LatentDim]
        """
        z, _ = self.encoder(x)
        z = self.flatten(z)
        z = self.output(z)
        return z
        