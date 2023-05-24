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
import numpy as np
import pytorch_lightning
import matplotlib.pyplot as plt

from toy_experiment.utils import metrics
from toy_experiment.utils.plotting import show_sequences
from toy_experiment.utils.utils import MAELoss


class DynamicsModel(pytorch_lightning.LightningModule):
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

        # Encoder/decoder architecture
        # self.encoder = nn.Linear(args.in_dim, args.latent_dim)
        # self.decoder = nn.Linear(args.latent_dim, args.out_dim)

        # Recurrent dynamics function
        self.dynamics_func = None

        # Losses
        self.reconstruction_loss = MAELoss()

    def forward(self, x, controls, generation_training_len):
        """ Placeholder function for the dynamics forward pass """
        raise NotImplementedError("In forward: Latent Dynamics function not specified.")

    def model_specific_loss(self, x, domain, preds, train=True):
        """ Placeholder function for any additional loss terms a dynamics function may have """
        return 0.0

    def model_specific_plotting(self, version_path, outputs):
        """ Placeholder function for any additional plots a dynamics function may have """
        return None

    def configure_optimizers(self):
        """ By default, we assume a optim with the Adam Optimizer """
        optim = torch.optim.AdamW(self.parameters(), lr=self.args.learning_rate)
        return optim

    def on_train_start(self):
        """
        Before a training session starts, we set some model variables and save a JSON configuration of the
        used hyper-parameters to allow for easy load-in at test-time.
        """
        # Get local version path from absolute directory
        self.version_path = f"{os.path.abspath('')}/lightning_logs/version_{self.top}/"

        # Get total number of parameters for the model and save
        params = float(sum(p.numel() for p in self.parameters() if p.requires_grad))
        if hasattr(self.dynamics_func, "codes"):
            params -= sum(p.numel() for p in self.dynamics_func.dynamics_network.parameters())

        self.log("total_num_parameters", params, prog_bar=False)
        print(f"=> Parameters w/o GroupConv: {params}")

        # Save config file to the version path
        shutil.copy(f"{self.args.config_path}", f"{self.version_path}/")

        # Make image dir in lightning experiment folder if it doesn't exist
        if not os.path.exists(f"{self.version_path}/signals/"):
            os.mkdir(f"{self.version_path}/signals/")

    def get_step_outputs(self, batch, generation_len):
        """
        Handles the process of pre-processing and subsequence sampling a batch,
        as well as getting the outputs from the models regardless of step
        :param batch: list of dictionary objects representing a single image
        :param generation_training_len: how far out to generate for, dependent on the step (train/val)
        :return: processed model outputs
        """
        # Stack batch and restrict to generation length
        indices, signals, controls = batch

        # Get predictions
        preds = self(signals, controls, generation_len)
        return signals, controls, preds

    def get_step_losses(self, signals, domains, preds, is_train=True):
        """
        Handles getting the ELBO terms for the given step
        :param signals: ground truth signals
        :param preds: forward predictions from the model
        :return: likelihood, kl on z0, model-specific dynamics loss
        """
        # Reconstruction loss for the sequence and z0
        likelihood = self.reconstruction_loss(preds, signals).mean([1, 2]).mean()
        init_likelihood = 0

        # Get the loss terms from the specific latent dynamics loss
        dynamics_loss = self.model_specific_loss(signals, domains, preds, is_train)
        return likelihood, init_likelihood, dynamics_loss

    def get_epoch_metrics(self, outputs):
        """
        Takes the dictionary of saved batch metrics, stacks them, and gets outputs to log in the Tensorboard.
        :param outputs: list of dictionaries with outputs from each back
        :return: dictionary of metrics aggregated over the epoch
        """
        # Convert outputs to Tensors and then Numpy arrays
        signals = torch.vstack([out["signals"] for out in outputs])
        preds = torch.vstack([out["preds"] for out in outputs])

        # Iterate through each metric function and add to a dictionary
        out_metrics = {}
        for met in self.args.metrics:
            metric_function = getattr(metrics, met)
            out_metrics[met] = metric_function(signals, preds, args=self.args)[0]

        # Return a dictionary of metrics
        return out_metrics

    def training_step(self, batch, batch_idx):
        """
        PyTorch-Lightning training step where the network is propagated and returns a single loss value,
        which is automatically handled for the backward update
        :param batch: list of dictionary objects representing a single image
        :param batch_idx: how far in the epoch this batch is
        """
        # Get model outputs from batch
        signals, controls, preds = self.get_step_outputs(batch, self.args.timesteps)

        # Get model loss terms for the step
        likelihood, init_likelihood, dynamics_loss = self.get_step_losses(signals, controls, preds, is_train=True)

        # Build the full loss
        loss = likelihood + 5 * init_likelihood + dynamics_loss

        # Log loss terms
        self.log_dict({
            "likelihood": likelihood,
            "init_likelihood": 5 * init_likelihood,
            "dynamics_loss": dynamics_loss
        }, prog_bar=True, on_epoch=False, on_step=True)

        # Return outputs as dict
        out = {"loss": loss, "preds": preds.detach(), "signals": signals.detach(), "controls": controls.detach()}
        return out

    def training_epoch_end(self, outputs):
        """
        Every 4 epochs, get a reconstruction example, model-specific plots, and copy over to the experiments folder
        :param outputs: list of outputs from the training steps, with the last 25 steps having reconstructions
        """
        # Log epoch metrics on saved batches
        metrics = self.get_epoch_metrics(outputs)
        for metric in metrics.keys():
            self.log(f"train_{metric}", metrics[metric], prog_bar=True)

        # Only log signals every 10 epochs
        if self.current_epoch % self.args.check_every_n_steps != 0:
            return

        # Show side-by-side reconstructions
        show_sequences(outputs[0]["signals"], outputs[0]["preds"],
                       f'{self.version_path}/signals/recon{self.current_epoch}train.png', num_out=4)

        # Get per-dynamics plots
        self.model_specific_plotting(self.version_path, outputs)

        # Copy experiment to relevant folder
        if self.args.exptype is not None:
            shutil.copytree(
                self.version_path, f"experiments/{self.args.exptype}/{self.args.model}/version_{self.exptop}",
                dirs_exist_ok=True
            )

    def test_step(self, batch, batch_idx):
        """
        PyTorch-Lightning testing step.
        :param batch: list of dictionary objects representing a single image
        :param batch_idx: how far in the epoch this batch is
        """
        # Get model outputs from batch
        signals, controls, preds = self.get_step_outputs(batch, self.args.timesteps)

        # Return outputs as dict
        out = {"preds": preds.detach(), "signals": signals.detach(), "controls": controls.detach()}
        return out

    def test_epoch_end(self, outputs):
        """
        For testing end, save the predictions, gt, and MSE to NPY files in the respective experiment folder
        :param outputs: list of outputs from the validation steps at batch 0
        """
        # Consolidate arrays into lists from output dictionaries
        preds, signals, controls = [], [], []
        for output in outputs:
            preds.append(output["preds"])
            signals.append(output["signals"])
            controls.append(output["controls"])

        # Stack all tensors at once and do one cpu transfer each
        preds = torch.vstack(preds)
        signals = torch.vstack(signals)
        controls = torch.vstack(controls)
        del outputs

        # Iterate through each metric function and add to a dictionary
        out_metrics = {}
        for met in self.args.metrics:
            metric_function = getattr(metrics, met)
            metric_mean, metric_std = metric_function(signals, preds, args=self.args)
            out_metrics[f"{met}_mean"], out_metrics[f"{met}_std"] = float(metric_mean), float(metric_std)
            print(f"=> {met}: {metric_mean:4.10f}+-{metric_std:4.10f}")

        if not os.path.exists(f"{self.args.ckpt_path}/test_files/"):
            os.mkdir(f"{self.args.ckpt_path}/test_files/")

        # Save metrics to JSON in checkpoint folder
        if self.args.interpolation is True:
            with open(f"{self.args.ckpt_path}/test_files/test_metrics_interpolation.json", 'w+') as f:
                json.dump(out_metrics, f)
        else:
            with open(f"{self.args.ckpt_path}/test_files/test_metrics_extrapolation.json", 'w+') as f:
                json.dump(out_metrics, f)

        # Save metrics to an easy excel conversion style
        with open(f"{self.args.ckpt_path}/test_files/test_excel.txt", 'w+') as f:
            for metric in self.args.metrics:
                f.write(f"{out_metrics[f'{metric}_mean']:0.3f}({out_metrics[f'{metric}_std']:0.3f}),")

        # Save files
        np.save(f"{self.args.ckpt_path}/test_files/test_recons.npy", preds.cpu().numpy())
        np.save(f"{self.args.ckpt_path}/test_files/test_signals.npy", signals.cpu().numpy())
        np.save(f"{self.args.ckpt_path}/test_files/test_controls.npy", controls.cpu().numpy())

        # Show side-by-side reconstructions
        show_sequences(signals[:10], preds[:10], f"{self.args.ckpt_path}/test_files/test_examples.png", num_out=5)

        # Plot the reconstructed full mesh
        plt.plot(range(20), signals[:, :, 0].detach().cpu().numpy().T, c='b')
        plt.plot(range(20), preds[:, :, 0].detach().cpu().numpy().T, c='k', linestyle='--')
        plt.title(f"Plot of U[-2, 2]")
        plt.xlabel("Blue: GT | Black: Preds")

        if self.args.interpolation is True:
            plt.savefig(f"{self.args.ckpt_path}/reconstructedControls_interpolation.png")
        else:
            plt.savefig(f"{self.args.ckpt_path}/reconstructedControls_extrapolation.png")
        plt.close()
