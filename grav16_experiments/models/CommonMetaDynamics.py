"""
@file CommonMetaDynamics.py

A common class that each meta latent dynamics function inherits.
Holds the training + validation step logic and the VAE components for reconstructions.
Has a testing step for holdout steps that handles all metric calculations and visualizations.
"""
import os
import json
import torch
import shutil
import itertools
import numpy as np
import torch.nn as nn
import pytorch_lightning
import matplotlib.pyplot as plt

from openTSNE import TSNE
from scipy import interpolate
from scipy.spatial import ConvexHull

from utils.plotting import show_images, get_embedding_trajectories, show_sequences
from utils import metrics
from utils.utils import determine_annealing_factor, CosineAnnealingWarmRestartsWithDecayAndLinearWarmup
from models.CommonVAE import LatentStateEncoder, EmissionDecoder, LinearStateEncoder, LinearDecoder, \
    LatentStateEncoderSWE, EmissionDecoderSWE


class LatentMetaDynamicsModel(pytorch_lightning.LightningModule):
    def __init__(self, args, top, exptop):
        """
        Generic implementation of a Latent Dynamics Model
        Holds the training and testing boilerplate, as well as experiment tracking
        :param args: passed in user arguments
        :param top: top lightning log version
        :param exptop: top experiment folder version
        """
        super().__init__()
        self.save_hyperparameters(args)

        # Args
        self.args = args
        self.top = top
        self.exptop = exptop

        # Encoder + Decoder
        if args.linear is True:
            self.encoder = LinearStateEncoder(args.z_amort, args.num_hidden, 50, args.latent_dim)
            self.decoder = LinearDecoder(args.batch_size, args.generation_training_len, 50, args.num_hidden, args.latent_dim)
        elif args.dim == 64:
            self.encoder = LatentStateEncoderSWE(args.z_amort, args.num_filt, 1, args.latent_dim, args.fix_variance)
            self.decoder = EmissionDecoderSWE(args.batch_size, args.generation_training_len, args.dim, args.num_filt, 1, args.latent_dim)
        else:
            self.encoder = LatentStateEncoder(args.z_amort, args.num_filt, 1, args.latent_dim, args.stochastic)
            self.decoder = EmissionDecoder(args.batch_size, args.generation_training_len, args.dim, args.num_filt, 1, args.latent_dim)

        # Recurrent dynamics function
        self.dynamics_func = None
        self.dynamics_out = None

        # Number of steps for training
        self.n_updates = 0

        # Losses
        self.reconstruction_loss = nn.MSELoss(reduction='none')

    def forward(self, x, D, labels, generation_training_len):
        """ Placeholder function for the dynamics forward pass """
        raise NotImplementedError("In forward: Latent Dynamics function not specified.")

    def model_specific_loss(self, x, domain, train=True):
        """ Placeholder function for any additional loss terms a dynamics function may have """
        return 0.0

    def model_specific_plotting(self, version_path, outputs):
        """ Placeholder function for any additional plots a dynamics function may have """
        return None

    @staticmethod
    def get_model_specific_args():
        """ Placeholder function for model-specific arguments """
        return {}

    def configure_optimizers(self):
        """
        By default, we assume a joint optim with the Adam Optimizer. We additionally include LR Warmup and
        CosineAnnealing with decay for standard learning rate care during training.

        For CosineAnnealing, we set the LR bounds to be [LR * 1e-2, LR]
        """
        optim = torch.optim.AdamW(self.parameters(), lr=self.args.learning_rate)
        scheduler = CosineAnnealingWarmRestartsWithDecayAndLinearWarmup(
            optim,
            T_0=self.args.scheduler['restart_interval'], T_mult=1,
            eta_min=self.args.learning_rate * 1e-2,
            warmup_steps=self.args.scheduler['warmup_steps'],
            decay=self.args.scheduler['decay']
        )

        # Explicit dictionary to state how often to ping the scheduler
        scheduler = {
            'scheduler': scheduler,
            'frequency': 1,
            'interval': 'step'
        }
        return [optim], [scheduler]

    def on_train_start(self):
        """
        Before a training session starts, we set some model variables and save a JSON configuration of the
        used hyper-parameters to allow for easy load-in at test-time.
        """
        if self.args.model == "coda":
            self.dynamics_func.set_initial_codes()

        # Get local version path from absolute directory
        self.version_path = f"{os.path.abspath('')}/lightning_logs/version_{self.top}/"

        # Get total number of parameters for the model and save
        self.log("total_num_parameters", float(sum(p.numel() for p in self.parameters() if p.requires_grad)), prog_bar=False)

        # Save config file to the version path
        shutil.copy(f"{self.args.config_path}", f"{self.version_path}/")

        # Make image dir in lightning experiment folder if it doesn't exist
        if not os.path.exists(f"{self.version_path}/images/"):
            os.mkdir(f"{self.version_path}/images/")

    def get_step_outputs(self, batch, generation_len):
        """
        Handles the process of pre-processing and subsequence sampling a batch,
        as well as getting the outputs from the models regardless of step
        :param batch: list of dictionary objects representing a single image
        :param generation_training_len: how far out to generate for, dependent on the step (train/val)
        :return: processed model outputs
        """
        # Stack batch and restrict to generation length
        images, domains, states, domain_state, labels = batch

        # Same random portion of the sequence over generation_training_len, saving room for backwards solving
        random_start = np.random.randint(generation_len, images.shape[1] - generation_len)

        # If a varying domain is selected, sample a random number from batch's domain
        if self.args.domain_varying is True:
            random_domain = np.random.randint(1, domains.shape[1])
            domains = domains[:, :random_domain]

        # Get forward sequences
        images = images[:, random_start:random_start + generation_len]
        if domains is not None:
            domains = domains[:, :, random_start:random_start + generation_len]
        states = states[:, random_start:random_start + generation_len]

        # Get predictions
        preds, embeddings = self(images, domains, labels, generation_len)
        return images, domains, states, labels, preds, embeddings

    def get_step_losses(self, images, domains, preds):
        """
        Handles getting the ELBO terms for the given step
        :param images: ground truth images
        :param preds: forward predictions from the model
        :return: likelihood, kl on z0, model-specific dynamics loss
        """
        # Reconstruction loss for the sequence and z0
        likelihood = self.reconstruction_loss(preds, images)
        likelihood = likelihood.reshape([likelihood.shape[0] * likelihood.shape[1], -1]).sum([-1]).mean()

        # Initial encoder loss, KL[q(z_K|x_0:K) || p(z_K)]
        klz = self.encoder.kl_z_term()

        # Get the loss terms from the specific latent dynamics loss
        dynamics_loss = self.model_specific_loss(images, domains, preds)
        return likelihood, klz, dynamics_loss

    def get_epoch_metrics(self, outputs):
        """
        Takes the dictionary of saved batch metrics, stacks them, and gets outputs to log in the Tensorboard.
        :param outputs: list of dictionaries with outputs from each back
        :return: dictionary of metrics aggregated over the epoch
        """
        # Convert outputs to Tensors and then Numpy arrays
        images = torch.vstack([out["images"] for out in outputs]).cpu().numpy()
        preds = torch.vstack([out["preds"] for out in outputs]).cpu().numpy()

        # Iterate through each metric function and add to a dictionary
        out_metrics = {}
        for met in self.args.metrics:
            metric_function = getattr(metrics, met)
            out_metrics[met] = metric_function(images, preds, args=self.args)[0]

        # Return a dictionary of metrics
        return out_metrics

    def training_step(self, batch, batch_idx):
        """
        PyTorch-Lightning training step where the network is propagated and returns a single loss value,
        which is automatically handled for the backward update
        :param batch: list of dictionary objects representing a single image
        :param batch_idx: how far in the epoch this batch is
        """
        if self.args.meta is True:
            self.trainer.train_dataloader.dataset.datasets.split()

        # Get model outputs from batch
        images, domains, states, labels, preds, embeddings = self.get_step_outputs(batch, self.args.generation_training_len)

        # Get model loss terms for the step
        likelihood, klz, dynamics_loss = self.get_step_losses(images, domains, preds)

        # Determine KL annealing factor for the current step
        kl_factor = determine_annealing_factor(self.n_updates, anneal_update=1000)

        # Build the full loss
        loss = likelihood + kl_factor * ((self.args.z0_beta * klz) + dynamics_loss)

        # Log loss terms
        self.log_dict({
            "likelihood": likelihood,
            "klz_loss": klz,
            "dynamics_loss": dynamics_loss,
            'kl_factor': kl_factor
        }, prog_bar=True, on_epoch=False, on_step=True)

        # Return outputs as dict
        self.n_updates += 1
        out = {"loss": loss, "labels": labels.detach()}
        if batch_idx < self.args.batches_to_save:
            out["preds"] = preds.detach()
            out["images"] = images.detach()

            # For code vector based models (i.e. the proposed_nonstationary models) also add their local codes
            if hasattr(self.dynamics_func, 'embeddings'):
                out['code_vectors'] = self.dynamics_func.embeddings.detach()
            # if hasattr(self.dynamics_func, 'ws'):
            #     out['ws'] = torch.concatenate([w.reshape([images.shape[0], -1]) for w in self.dynamics_func.ws], dim=1).detach()
            # if hasattr(self.dynamics_func, 'bs'):
            #     out['bs'] = torch.concatenate([b.reshape([images.shape[0], -1]) for b in self.dynamics_func.bs], dim=1).detach()
        return out

    def training_epoch_end(self, outputs):
        """
        Every 4 epochs, get a reconstruction example, model-specific plots, and copy over to the experiments folder
        :param outputs: list of outputs from the training steps, with the last 25 steps having reconstructions
        """
        # Log epoch metrics on saved batches
        metrics = self.get_epoch_metrics(outputs[:self.args.batches_to_save])
        for metric in metrics.keys():
            self.log(f"train_{metric}", metrics[metric], prog_bar=True)

        # Only log images every 10 epochs
        if self.current_epoch % 10 != 0:
            return

        # Show side-by-side reconstructions
        show_images(outputs[0]["images"], outputs[0]["preds"],
                    f'{self.version_path}/images/recon{self.current_epoch}train.png', num_out=5)

        # Get per-dynamics plots
        self.model_specific_plotting(self.version_path, outputs)

        # Copy experiment to relevant folder
        if self.args.exptype is not None:
            shutil.copytree(
                self.version_path, f"experiments/{self.args.exptype}/{self.args.model}/version_{self.exptop}",
                dirs_exist_ok=True
            )

    def validation_step(self, batch, batch_idx):
        """
        PyTorch-Lightning validation step. Similar to the training step but on the given val set under torch.no_grad()
        :param batch: list of dictionary objects representing a single image
        :param batch_idx: how far in the epoch this batch is
        """
        # Get model outputs from batch
        images, domains, states, labels, preds, embeddings = self.get_step_outputs(batch, self.args.generation_validation_len)

        # Get model loss terms for the step
        likelihood, klz, dynamics_loss = self.get_step_losses(images, domains, preds)

        # Log validation likelihood and metrics
        self.log("val_likelihood", likelihood, prog_bar=True)

        # Build the full loss
        loss = likelihood + dynamics_loss

        # Return outputs as dict
        out = {"loss": loss}
        if batch_idx < self.args.batches_to_save:
            out["preds"] = preds.detach()
            out["images"] = images.detach()
        return out

    def validation_epoch_end(self, outputs):
        """
        Every 4 epochs get a validation reconstruction sample
        :param outputs: list of outputs from the validation steps at batch 0
        """
        # Log epoch metrics on saved batches
        metrics = self.get_epoch_metrics(outputs[:self.args.batches_to_save])
        for metric in metrics.keys():
            self.log(f"val_{metric}", metrics[metric], prog_bar=True)

        # Show side-by-side reconstructions
        show_images(outputs[0]["images"], outputs[0]["preds"],
                    f'{self.version_path}/images/recon{self.current_epoch}val.png', num_out=5)

    def test_step(self, batch, batch_idx):
        """
        PyTorch-Lightning testing step.
        :param batch: list of dictionary objects representing a single image
        :param batch_idx: how far in the epoch this batch is
        """
        if self.args.meta is True:
            self.trainer.test_dataloaders[0].dataset.split()

        # Get model outputs from batch
        images, domains, states, labels, preds, embeddings = \
            self.get_step_outputs(batch, self.args.testing['generation_testing_len'])

        # Build output dictionary
        out = {"labels": labels.detach()}
        if batch_idx < self.args.batches_to_save:
            out.update({"states": states.detach(), "embeddings": embeddings.detach(),
               "preds": preds.detach(), "images": images.detach()})

            # For code vector based models (i.e. the proposed_nonstationary models) also add their local codes
            if hasattr(self.dynamics_func, 'embeddings'):
                out['code_vectors'] = self.dynamics_func.embeddings.detach()
            # if hasattr(self.dynamics_func, 'ws'):
            #     out['ws'] = torch.concatenate([w.reshape([self.args.batch_size, -1]) for w in self.dynamics_func.ws], dim=1).detach()
            # if hasattr(self.dynamics_func, 'bs'):
            #     out['bs'] = torch.concatenate([b.reshape([self.args.batch_size, -1]) for b in self.dynamics_func.bs], dim=1).detach()
        return out

    def test_epoch_end(self, outputs):
        """
        For testing end, save the predictions, gt, and MSE to NPY files in the respective experiment folder
        :param outputs: list of outputs from the validation steps at batch 0
        """
        # Consolidate arrays into lists from output dictionaries
        preds, images, states, embeddings, labels = [], [], [], [], []
        code_vectors, ws, bs = [], [], []
        for output in outputs[:self.args.batches_to_save]:
            preds.append(output["preds"])
            images.append(output["images"])
            states.append(output["states"])
            labels.append(output["labels"])
            embeddings.append(output["embeddings"])

            if "code_vectors" in output.keys():
                code_vectors.append(output["code_vectors"])
            if "ws" in output.keys():
                ws.append(output["ws"])
            if "bs" in output.keys():
                bs.append(output["bs"])

        # Stack all tensors at once and do one cpu transfer each
        preds = torch.vstack(preds).cpu().numpy()
        images = torch.vstack(images).cpu().numpy()
        states = torch.vstack(states).cpu().numpy()
        embeddings = torch.vstack(embeddings).cpu().numpy()
        labels = torch.vstack(labels).cpu().numpy()
        if "code_vectors" in output.keys():
            code_vectors = torch.vstack(code_vectors).cpu().numpy()
        if "ws" in output.keys():
            ws = torch.vstack(ws).cpu().numpy()
        if "bs" in output.keys():
            bs = torch.vstack(bs).cpu().numpy()
        del outputs

        # Iterate through each metric function and add to a dictionary
        out_metrics = {}
        for met in self.args.metrics:
            metric_function = getattr(metrics, met)
            metric_mean, metric_std = metric_function(images, preds, args=self.args)
            out_metrics[f"{met}_mean"], out_metrics[f"{met}_std"] = float(metric_mean), float(metric_std)
            print(f"=> {met}: {metric_mean:4.5f}+-{metric_std:4.5f}")

        # Set up output path and create dir
        output_path = f"{self.args.ckpt_path}/test_{self.args.split}"
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        # Save metrics to JSON in checkpoint folder
        with open(f"{output_path}/test_{self.args.split}_metrics.json", 'w') as f:
            json.dump(out_metrics, f)

        # Save metrics to an easy excel conversion style
        with open(f"{output_path}/test_{self.args.dataset}_excel.txt", 'w') as f:
            for metric in self.args.metrics:
                f.write(f"{out_metrics[f'{metric}_mean']:0.3f}({out_metrics[f'{metric}_std']:0.3f}),")

        # Save files
        if self.args.testing['save_files'] is True:
            np.save(f"{output_path}/test_{self.args.split}_recons.npy", preds)
            np.save(f"{output_path}/test_{self.args.split}_images.npy", images)
            np.save(f"{output_path}/test_{self.args.split}_labels.npy", labels)

        # Show side-by-side reconstructions
        if self.args.linear is True:
            show_sequences(images[:10], preds[:10], f"{output_path}/test_{self.args.split}_examples.png", num_out=5)
        else:
            show_images(images[:10], preds[:10], f"{output_path}/test_{self.args.split}_examples.png", num_out=5)

        # Save trajectory examples
        get_embedding_trajectories(embeddings[0], states[0], f"{output_path}/")

        # Get Z0 TSNE
        tsne = TSNE(n_components=2, perplexity=30, initialization="pca", metric="cosine", n_jobs=8, random_state=3)
        tsne_embedding = tsne.fit(embeddings[:, 0])

        # Plot prototypes
        for i in np.unique(labels):
            subset = tsne_embedding[np.where(labels == i)[0], :]
            plt.scatter(subset[:, 0], subset[:, 1],alpha=0.5, label=f"{i}")

        plt.title("t-SNE Plot of Z0 Embeddings")
        plt.legend(np.unique(labels), loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(f"{output_path}/test_{self.args.split}_Z0tsne.png", bbox_inches='tight')
        plt.close()

        # If it's a code model, get embedding TSNE
        if len(code_vectors) > 0:
            # Stack and reshape
            code_vectors = np.vstack(code_vectors)
            code_vectors = code_vectors.reshape([code_vectors.shape[0], -1])

            # Save code_vectors to local file
            if self.args.testing['save_files'] is True:
                np.save(f"{output_path}/test_{self.args.split}_codevectors.npy", code_vectors)

            # Generate TSNE embeddings of C
            tsne = TSNE(n_components=2, perplexity=30, initialization="pca", metric="cosine", n_jobs=8, random_state=3)
            tsne_embedding = tsne.fit(code_vectors)

            # Plot codes in TSNE
            marker = itertools.cycle(('o', 'v', '^', '<', '>', 's', '8', 'p'))
            for i in np.unique(labels):
                subset = tsne_embedding[np.where(labels == i)[0], :]
                color = next(plt.gca()._get_lines.prop_cycler)['color']

                # Get convex hull
                hull = ConvexHull(subset)
                x_hull = np.append(subset[hull.vertices, 0], subset[hull.vertices, 0][0])
                y_hull = np.append(subset[hull.vertices, 1], subset[hull.vertices, 1][0])

                # Interpolate
                dist = np.sqrt((x_hull[:-1] - x_hull[1:]) ** 2 + (y_hull[:-1] - y_hull[1:]) ** 2)
                dist_along = np.concatenate(([0], dist.cumsum()))
                spline, u = interpolate.splprep([x_hull, y_hull], u=dist_along, s=0, per=1)
                interp_d = np.linspace(dist_along[0], dist_along[-1], 50)
                interp_x, interp_y = interpolate.splev(interp_d, spline)

                # Plot points in cluster
                plt.scatter(subset[:, 0], subset[:, 1], alpha=0.5, c=color, marker=next(marker), label=f"{i}")

                # Plot boundaries
                plt.fill(interp_x, interp_y, '--', alpha=0.2, c=color)

            plt.title("t-SNE Plot of Generated Codes")
            plt.legend(loc='upper right')
            plt.savefig(f"{output_path}/test_{self.args.split}_codetsne.png", bbox_inches='tight')
            plt.close()

            # Plot codes in histogram
            for i in np.unique(labels):
                subset = np.reshape(code_vectors[np.where(labels == i)[0], :], [-1])
                plt.hist(subset, bins=100, alpha=0.5, label=f"{i}")

            plt.legend()
            plt.title("Distribution of Generated Codes")
            plt.savefig(f"{output_path}/test_{self.args.split}_codehist.png", bbox_inches='tight')
            plt.close()

        # If it is a code model
        if len(ws) > 0:
            # Stack and reshape
            ws = np.vstack(ws)

            # Get weight histogram
            for i in np.unique(labels):
                subset = np.mean(ws[np.where(labels == i)[0], :], axis=0)
                plt.hist(subset, bins=100, alpha=0.5, label=f"{i}")

            plt.legend()
            plt.title("Distribution of Generated Weights from Codes")
            plt.savefig(f"{output_path}/test_{self.args.split}_codeweights.png", bbox_inches='tight')
            plt.close()

        if len(bs) > 0:
            # Stack biases
            bs = np.vstack(bs)

            # Get bias histogram
            for i in np.unique(labels):
                subset = np.mean(bs[np.where(labels == i)[0], :], axis=0)
                plt.hist(subset, bins=50, alpha=0.5, label=f"{i}")

            plt.ylim([0, 15])
            plt.title("Distribution of Generated Biases from Codes")
            plt.legend()
            plt.savefig(f"{output_path}/test_{self.args.split}_codebiases.png", bbox_inches='tight')
            plt.close()
