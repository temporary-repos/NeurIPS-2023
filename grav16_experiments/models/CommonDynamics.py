"""
@file CommonDynamics.py

A common class that each latent dynamics function inherits.
Holds the training + validation step logic and the VAE components for reconstructions.
"""
import os
import json
import torch
import shutil
import numpy as np
import torch.nn as nn
import pytorch_lightning
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from utils.plotting import show_images, get_embedding_trajectories, show_sequences
from utils.metrics import vpt, dst, r2fit, vpd
from utils.utils import determine_annealing_factor, CosineAnnealingWarmRestartsWithDecayAndLinearWarmup
from models.CommonVAE import LatentStateEncoder, EmissionDecoder, LinearStateEncoder, LinearDecoder


class LatentDynamicsModel(pytorch_lightning.LightningModule):
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
        if self.args.linear is True:
            self.encoder = LinearStateEncoder(self.args.z_amort, self.args.num_hidden, self.args.dim, self.args.latent_dim, self.args.fix_variance)
            self.decoder = LinearDecoder(self.args.batch_size, self.args.generation_len, self.args.dim, self.args.num_hidden, self.args.latent_dim)
        else:
            self.encoder = LatentStateEncoder(self.args.z_amort, self.args.num_filt, 1, self.args.latent_dim, self.args.stochastic)
            self.decoder = EmissionDecoder(self.args.batch_size, self.args.generation_len, self.args.dim, self.args.num_filt, 1, self.args.latent_dim)

        # Recurrent dynamics function
        self.dynamics_func = None
        self.dynamics_out = None

        # Number of steps for training
        self.n_updates = 0

        # Losses
        self.reconstruction_loss = nn.MSELoss(reduction='none')

    def forward(self, x, generation_len):
        """ Placeholder function for the dynamics forward pass """
        raise NotImplementedError("In forward: Latent Dynamics function not specified.")

    def model_specific_loss(self, images, preds, labels, train=True):
        """ Placeholder function for any additional loss terms a dynamics function may have """
        return 0.0

    def model_specific_plotting(self, version_path, outputs):
        """ Placeholder function for any additional plots a dynamics function may have """
        return None

    @staticmethod
    def add_model_specific_args(parent_parser):
        """ Placeholder function for model-specific arguments """
        return parent_parser

    def configure_optimizers(self):
        """
        Most standard NSSM models have a joint optimization step under one ELBO, however there is room
        for EM-optimization procedures based on the PGM.

        By default, we assume a joint optim with the Adam Optimizer. We additionally include LR Warmup and
        StepLR decay for standard learning rate care during training.
        """
        optim = torch.optim.AdamW(self.parameters(), lr=self.args.learning_rate)
        scheduler = CosineAnnealingWarmRestartsWithDecayAndLinearWarmup(optim, T_0=5000, T_mult=1,
                                                                        eta_min=self.args.learning_rate * 1e-2,
                                                                        warmup_steps=2, decay=0.90)

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
        # Get local version path from absolute directory
        self.version_path = f"{os.path.abspath('')}/lightning_logs/version_{self.top}/"

        # Get total number of parameters for the model and save
        pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Save hyperparameters to the log
        params = {
            'latent_dim': self.args.latent_dim,
            'num_layers': self.args.num_layers,
            'num_hidden': self.args.num_hidden,
            'num_filt': self.args.num_filt,
            'latent_act': self.args.latent_act,
            'z_amort': self.args.z_amort,
            'fix_variance': self.args.fix_variance,
            'number_params': pytorch_total_params
        }
        with open(f"{self.version_path}/params.json", 'w') as f:
            json.dump(params, f)

        # Make image dir in lightning experiment folder if it doesn't exist
        if not os.path.exists(f"{self.version_path}/images/"):
            os.mkdir(f"{self.version_path}/images/")

    def on_validation_start(self):
        """
        Extra check to make sure the version path variable is set when restarting from a given checkpoint.
        Sometimes it can start on a reconstruction-based epoch and cause errors without this check.
        """
        # Get local version path from absolute directory
        self.version_path = f"{os.path.abspath('')}/lightning_logs/version_{self.top}/"

        # Make image dir in lightning experiment folder if it doesn't exist
        if not os.path.exists(f"{self.version_path}/images/"):
            os.mkdir(f"{self.version_path}/images/")

    def get_step_outputs(self, batch, generation_len):
        """
        Handles the process of pre-processing and subsequence sampling a batch,
        as well as getting the outputs from the models regardless of step
        :param batch: list of dictionary objects representing a single image
        :param generation_len: how far out to generate for, dependent on the step (train/val)
        :return: processed model outputs
        """
        # Stack batch and restrict to generation length
        images = torch.stack([b['image'] for b in batch[0]])
        states = torch.stack([b['x'] for b in batch[0]]).squeeze(1)
        labels = torch.stack([b['class_id'] for b in batch[0]])

        # One-hot encode the data points
        map_dict = {2: 0, 5: 1, 10: 2}
        labels = torch.tensor([map_dict[x.item()] for x in labels]).to(self.device)

        # Same random portion of the sequence over generation_len, saving room for backwards solving
        random_start = np.random.randint(generation_len, images.shape[1] - generation_len)

        # Get forward sequences
        images = images[:, random_start:random_start + generation_len]
        states = states[:, random_start:random_start + generation_len]

        # Get predictions
        preds, embeddings = self(images, generation_len)
        return images, states, labels, preds, embeddings

    def get_step_losses(self, images, preds, labels):
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
        dynamics_loss = self.model_specific_loss(images, preds, labels)
        return likelihood, klz, dynamics_loss

    def training_step(self, batch, batch_idx):
        """
        PyTorch-Lightning training step where the network is propagated and returns a single loss value,
        which is automatically handled for the backward update
        :param batch: list of dictionary objects representing a single image
        :param batch_idx: how far in the epoch this batch is
        """
        # Get model outputs from batch
        images, states, labels, preds, embeddings = self.get_step_outputs(batch, self.args.generation_len)

        # Get model loss terms for the step
        likelihood, klz, dynamics_loss = self.get_step_losses(images, preds, labels)

        # Log ELBO loss terms
        self.log("likelihood", likelihood, prog_bar=True, on_epoch=True, on_step=False)
        self.log("klz_loss", klz, prog_bar=True, on_epoch=True, on_step=False)
        self.log("dynamics_loss", dynamics_loss, on_epoch=True, on_step=False)

        # Log various metrics
        if self.args.linear is False:
            self.log("train_vpt", vpt(images, preds.detach())[0], prog_bar=True, on_epoch=True, on_step=False)
            self.log("train_pixel_mse", self.reconstruction_loss(preds, images).mean([1, 2, 3]).mean(), prog_bar=True, on_epoch=True, on_step=False)
            self.log("train_dst", dst(images, preds.detach())[1], prog_bar=True, on_epoch=True, on_step=False)
            self.log("train_vpd", vpd(images, preds.detach())[1], prog_bar=True, on_epoch=True, on_step=False)

        # Determine KL annealing factor for the current step
        kl_factor = determine_annealing_factor(self.n_updates, anneal_update=1000)
        self.log('kl_factor', kl_factor, prog_bar=False, on_epoch=True, on_step=False)
        self.log("learning_rate", self.optimizers().param_groups[0]['lr'], prog_bar=False, on_step=True)

        # Build the full loss
        loss = likelihood + kl_factor * ((self.args.z0_beta * klz) + dynamics_loss)

        # Return outputs as dict
        self.n_updates += 1
        out = {"loss": loss, "labels": labels.detach()}
        if batch_idx < self.args.batches_to_save:
            out["preds"] = preds.detach()
            out["images"] = images.detach()

            # For code vector based models (i.e. the proposed_nonstationary models) also add their local codes
            if hasattr(self.dynamics_func, 'embeddings'):
                out['code_vectors'] = self.dynamics_func.embeddings.detach()
        return out

    def training_epoch_end(self, outputs):
        """
        # Every 4 epochs, get a reconstruction example, model-specific plots, and copy over to the experiments folder
        :param outputs: list of outputs from the training steps, with the last 25 steps having reconstructions
        """
        if self.current_epoch % 10 != 0:
            return

        # Show side-by-side reconstructions
        if self.args.linear is True:
            show_sequences(outputs[0]["images"], outputs[0]["preds"],
                    f'{self.version_path}/images/recon{self.current_epoch}train.png', num_out=5)
        else:
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

    # def backward(self, loss, optimizer, optimizer_idx):
    #     # Zero out gradients
    #     optimizer.zero_grad()
    #
    #     # Call loss backwards on the graph
    #     loss.backward()
    #
    #     # Get average grad and norm of grad
    #     grads = []
    #     for m in self.named_modules():
    #         if 'dynamics_network' in m[0]:
    #             continue
    #
    #         m = m[1]
    #         if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
    #             grads.append(m.weight.grad.detach())
    #             grads.append(m.bias.grad.detach())
    #     self.log("avg_grad", torch.concatenate([g.flatten() for g in grads]).mean(), on_step=True, prog_bar=True)
    #     self.log("avg_gradnorm", torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in grads]), 2), on_step=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        """
        PyTorch-Lightning validation step. Similar to the training step but on the given val set under torch.no_grad()
        :param batch: list of dictionary objects representing a single image
        :param batch_idx: how far in the epoch this batch is
        """
        # Get model outputs from batch
        images, states, labels, preds, embeddings = self.get_step_outputs(batch, self.args.generation_len)

        # Get model loss terms for the step
        likelihood, klz, dynamics_loss = self.get_step_losses(images, preds, labels)
        del states, labels, embeddings

        # Log validation likelihood and metrics
        self.log("val_likelihood", likelihood, prog_bar=True)

        if self.args.linear is False:
            self.log("val_vpt", vpt(images, preds.detach())[0], prog_bar=True, on_epoch=True, on_step=False)
            self.log("val_pixel_mse", self.reconstruction_loss(preds, images).mean([1, 2, 3]).mean(), prog_bar=True, on_epoch=True, on_step=False)
            self.log("val_dst", dst(images, preds.detach())[1], prog_bar=True, on_epoch=True, on_step=False)
            self.log("val_vpd", vpd(images, preds.detach())[1], prog_bar=True, on_epoch=True, on_step=False)

        # Build the full loss
        loss = likelihood + dynamics_loss

        # Return outputs as dict
        out = {"loss": loss}
        if batch_idx == 0:
            out["preds"] = preds.detach()
            out["images"] = images.detach()
        return out

    def validation_epoch_end(self, outputs):
        """
        Every 4 epochs get a validation reconstruction sample
        :param outputs: list of outputs from the validation steps at batch 0
        """
        if self.current_epoch == 0:
            return

        # Show side-by-side reconstructions
        if self.args.linear is True:
            show_sequences(outputs[0]["images"], outputs[0]["preds"],
                    f'{self.version_path}/images/recon{self.current_epoch}val.png', num_out=5)
        else:
            show_images(outputs[0]["images"], outputs[0]["preds"],
                    f'{self.version_path}/images/recon{self.current_epoch}val.png', num_out=5)

    def test_step(self, batch, batch_idx):
        """
        PyTorch-Lightning testing step.
        :param batch: list of dictionary objects representing a single image
        :param batch_idx: how far in the epoch this batch is
        """
        # Get model outputs from batch
        # TODO - Output 50 runs per batch to get averaged metrics rather than one run
        images, states, labels, preds, embeddings = self.get_step_outputs(batch, self.args.generation_len)

        pixel_mse_recon = self.reconstruction_loss(preds[:, :self.args.generation_len], images[:, :self.args.generation_len]).mean([1, 2, 3]).detach().cpu().numpy()
        pixel_mse_extrapolation = self.reconstruction_loss(preds[:, self.args.generation_len:], images[:, self.args.generation_len:]).mean([1, 2, 3]).detach().cpu().numpy()
        test_vpt = vpt(images, preds.detach())[0]
        test_dst = dst(images, preds.detach())[1]
        test_vpd = vpd(images, preds.detach())[1]

        # Define output dict
        out = {"states": states.detach().cpu().numpy(), "embeddings": embeddings.detach().cpu().numpy(),
         "preds": preds.detach().cpu().numpy(), "images": images.detach().cpu().numpy(),
         "labels": labels.detach().cpu().numpy(),
         "pixel_mse_recon": pixel_mse_recon, "pixel_mse_extrapolation": pixel_mse_extrapolation,
         "vpt": test_vpt, "dst": test_dst, "vpd": test_vpd}

        # For code vector based models (i.e. the proposed_nonstationary models) also add their local codes
        if hasattr(self.dynamics_func, 'embeddings'):
            out['code_vectors'] = self.dynamics_func.embeddings.detach().cpu().numpy()
        return out

    def test_epoch_end(self, outputs):
        """
        For testing end, save the predictions, gt, and MSE to NPY files in the respective experiment folder
        :param outputs: list of outputs from the validation steps at batch 0
        """
        # Stack all outputs
        preds, images, states, embeddings, labels = [], [], [], [], []
        mse_recons, mse_extrapolations, vpts, dsts, vpds = [], [], [], [], []
        code_vectors = []
        for output in outputs:
            preds.append(output["preds"])
            images.append(output["images"])
            states.append(output["states"])
            embeddings.append(output["embeddings"])
            labels.append(output["labels"])

            mse_recons.append(output["pixel_mse_recon"])
            mse_extrapolations.append(output["pixel_mse_extrapolation"])
            vpts.append(output["vpt"])
            dsts.append(output["dst"])
            vpds.append(output["vpd"])

            if "code_vectors" in output.keys():
                code_vectors.append(output["code_vectors"])

        preds = np.vstack(preds)
        images = np.vstack(images)
        states = np.vstack(states)
        embeddings = np.vstack(embeddings)
        labels = np.vstack(labels)

        pixel_mse_recons = np.vstack(mse_recons)
        pixel_mse_extrapolations = np.vstack(mse_extrapolations)
        vpts = np.vstack(vpts)
        dsts = np.vstack(dsts)
        vpds = np.vstack(vpds)
        del outputs

        # Print statistics over the full set
        print("")
        print(f"=> Pixel Recon MSE: {np.mean(pixel_mse_recons):4.5f}+-{np.std(pixel_mse_recons):4.5f}")
        print(f"=> Pixel Extrapolation MSE: {np.mean(pixel_mse_extrapolations):4.5f}+-{np.std(pixel_mse_extrapolations):4.5f}")
        print(f"=> VPT:       {np.mean(vpts):4.5f}+-{np.std(vpts):4.5f}")
        print(f"=> DST:       {np.mean(dsts):4.5f}+-{np.std(dsts):4.5f}")
        print(f"=> VPD:       {np.mean(vpds):4.5f}+-{np.std(vpds):4.5f}")

        metrics = {
            "mse_recon_mean": float(np.mean(pixel_mse_recons)),
            "mse_recon_std": float(np.std(pixel_mse_recons)),
            "mse_extrapolation_mean": float(np.mean(pixel_mse_extrapolations)),
            "mse_extrapolation_std": float(np.std(pixel_mse_extrapolations)),
            "vpt_mean": float(np.mean(vpts)),
            "vpt_std": float(np.std(vpts)),
            "dst_mean": float(np.mean(dsts)),
            "dst_std": float(np.std(dsts)),
            "vpd_mean": float(np.mean(vpds)),
            "vpd_std": float(np.std(vpds))
        }

        # # Get polar coordinates (sin and cos) of the angle for evaluation
        # sins = torch.sin(states[:, :, 0])
        # coss = torch.cos(states[:, :, 0])
        # states = torch.stack((sins, coss, states[:, :, 1]), dim=2)
        #
        # # Get r2 score
        # r2s = r2fit(embeddings, states, mlp=True)
        #
        # # Log each dimension's R2 individually
        # for idx, r in enumerate(r2s):
        #     metrics[f"r2_{idx}"] = r

        # Set up output path and create dir
        output_path = f"{self.args.ckpt_path}/test_{self.args.dataset}"
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        # Save files
        np.save(f"{output_path}/test_{self.args.dataset}_pixelmse_recon.npy", pixel_mse_recons)
        np.save(f"{output_path}/test_{self.args.dataset}_pixelmse_extrapolation.npy", pixel_mse_extrapolations)
        np.save(f"{output_path}/test_{self.args.dataset}_recons.npy", preds)
        np.save(f"{output_path}/test_{self.args.dataset}_images.npy", images)

        # Show side-by-side reconstructions
        if self.args.linear is True:
            show_sequences(images[:10], preds[:10], f"{output_path}/test_{self.args.dataset}_examples.png", num_out=5)
        else:
            show_images(images[:10], preds[:10], f"{output_path}/test_{self.args.dataset}_examples.png", num_out=5)

        # Save trajectory examples
        get_embedding_trajectories(embeddings[0], states[0], f"{output_path}/")

        # Get a TSNE plot of the z0 embeddings
        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, early_exaggeration=12)
        fitted = tsne.fit(embeddings[:, 0])
        print("Finished after {} iterations".format(fitted.n_iter))
        tsne_embedding = fitted.embedding_

        plt.figure()
        plt.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], c=labels)
        plt.title("t-SNE Plot of Z0 Latent Embeddings")
        plt.savefig(f"{output_path}/test_z0_tsne.png")
        plt.close()

        # Get a TSNE plot of the code embeddings embeddings
        if len(code_vectors) > 0:
            code_vectors = np.vstack(code_vectors)
            code_vectors = np.reshape(code_vectors, [code_vectors.shape[0], -1])

            tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, early_exaggeration=12)
            fitted = tsne.fit(code_vectors)
            print("Finished after {} iterations".format(fitted.n_iter))
            tsne_embedding = fitted.embedding_

            plt.figure()
            plt.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], c=labels)
            plt.title("t-SNE Plot of Code/Weight Embeddings")
            plt.savefig(f"{output_path}/test_weightembedding_tsne.png")
            plt.close()

        # Save metrics to JSON in checkpoint folder
        with open(f"{output_path}/test_{self.args.dataset}_metrics.json", 'w') as f:
            json.dump(metrics, f)

        # Save metrics to an easy excel conversion style
        with open(f"{output_path}/test_{self.args.dataset}_excel.txt", 'w') as f:
            f.write(f"{metrics['mse_recon_mean']},{metrics['mse_recon_std']},"
                    f"{metrics['mse_extrapolation_mean']},{metrics['mse_extrapolation_std']}"
                    f"{metrics['vpt_mean']},{metrics['vpt_std']},"
                    f"{metrics['dst_mean']},{metrics['dst_std']},{metrics['vpd_mean']},{metrics['vpd_std']}")
