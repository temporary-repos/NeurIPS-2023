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
import torch.nn as nn
import pytorch_lightning

from pde_experiments.utils import metrics
from pde_experiments.utils.plotting import show_sequences


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

        # Recurrent dynamics function
        self.dynamics_func = None
        self.dynamics_out = None

        # Number of steps for training
        self.n_updates = 0

        # Losses
        self.reconstruction_loss = nn.MSELoss()

    def forward(self, x, D, labels, generation_training_len):
        """ Placeholder function for the dynamics forward pass """
        raise NotImplementedError("In forward: Latent Dynamics function not specified.")

    def model_specific_loss(self, x, domain, preds, train=True):
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
        indices, signals, domains, labels, params = batch

        # Get predictions
        preds = self(signals, domains, labels, generation_len)
        return signals, domains, labels, params, preds

    def get_step_losses(self, signals, domains, preds, is_train=True):
        """
        Handles getting the ELBO terms for the given step
        :param signals: ground truth signals
        :param preds: forward predictions from the model
        :return: likelihood, kl on z0, model-specific dynamics loss
        """
        # Reconstruction loss for the sequence and z0
        likelihood = self.reconstruction_loss(preds, signals)

        # Get the loss terms from the specific latent dynamics loss
        dynamics_loss = self.model_specific_loss(signals, domains, preds, is_train)
        return likelihood, dynamics_loss

    def get_epoch_metrics(self, outputs):
        """
        Takes the dictionary of saved batch metrics, stacks them, and gets outputs to log in the Tensorboard.
        :param outputs: list of dictionaries with outputs from each back
        :return: dictionary of metrics aggregated over the epoch
        """
        # Convert outputs to Tensors and then Numpy arrays
        signals = torch.vstack([out["signals"] for out in outputs])
        preds = torch.vstack([out["preds"] for out in outputs])
        params = torch.vstack([out["params"] for out in outputs])

        # Iterate through each metric function and add to a dictionary
        out_metrics = {}
        for met in self.args.metrics:
            metric_function = getattr(metrics, met)
            out_metrics[met] = metric_function(signals, preds, args=self.args, params=params)[0]

        # Return a dictionary of metrics
        return out_metrics

    def training_step(self, batch, batch_idx):
        """
        PyTorch-Lightning training step where the network is propagated and returns a single loss value,
        which is automatically handled for the backward update
        :param batch: list of dictionary objects representing a single image
        :param batch_idx: how far in the epoch this batch is
        """
        self.trainer.train_dataloader.dataset.datasets.split()

        # Get model outputs from batch
        signals, domains, labels, params, preds = self.get_step_outputs(batch, self.args.generation_training_len)

        # Get model loss terms for the step
        likelihood, dynamics_loss = self.get_step_losses(signals, domains, preds, is_train=True)

        # Build the full loss
        loss = likelihood + dynamics_loss

        # Log loss terms
        self.log_dict({
            "likelihood": likelihood,
            "dynamics_loss": dynamics_loss
        }, prog_bar=True, on_epoch=False, on_step=True)

        # Return outputs as dict
        self.n_updates += 1
        out = {"loss": loss, "labels": labels.detach(),
               "preds": preds.detach(), "signals": signals.detach(),
               "params": params.detach()}

        # For code vector based models (i.e. the proposed_nonstationary models) also add their local codes
        if hasattr(self.dynamics_func, 'embeddings'):
            out['code_vectors'] = self.dynamics_func.embeddings.detach()
        if hasattr(self.dynamics_func, 'ws'):
            out['ws'] = torch.concatenate([w.reshape([self.args.batch_size, -1]) for w in self.dynamics_func.ws], dim=1).detach()
        if hasattr(self.dynamics_func, 'bs'):
            out['bs'] = torch.concatenate([b.reshape([self.args.batch_size, -1]) for b in self.dynamics_func.bs], dim=1).detach()
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

    def validation_step(self, batch, batch_idx):
        """
        PyTorch-Lightning validation step. Similar to the training step but on the given val set under torch.no_grad()
        :param batch: list of dictionary objects representing a single image
        :param batch_idx: how far in the epoch this batch is
        """
        # Get model outputs from batch
        signals, domains, labels, params, preds = self.get_step_outputs(batch, self.args.generation_validation_len)

        # Get model loss terms for the step
        likelihood, dynamics_loss = self.get_step_losses(signals, domains, preds, is_train=False)

        # Log validation likelihood and metrics
        self.log("val_likelihood", likelihood, prog_bar=True)

        # Build the full loss
        loss = likelihood + dynamics_loss

        # Return outputs as dict
        return {"loss": loss, "preds": preds.detach(), "params": params.detach(), "signals": signals.detach()}

    def validation_epoch_end(self, outputs):
        """
        Every 4 epochs get a validation reconstruction sample
        :param outputs: list of outputs from the validation steps at batch 0
        """
        # Log epoch metrics on saved batches
        metrics = self.get_epoch_metrics(outputs)
        for metric in metrics.keys():
            self.log(f"val_{metric}", metrics[metric], prog_bar=True)

        # Show side-by-side reconstructions
        show_sequences(outputs[0]["signals"], outputs[0]["preds"],
                       f'{self.version_path}/signals/recon{self.current_epoch}val.png', num_out=4)

    def test_step(self, batch, batch_idx):
        """
        PyTorch-Lightning testing step.
        :param batch: list of dictionary objects representing a single image
        :param batch_idx: how far in the epoch this batch is
        """
        # Get model outputs from batch
        signals, domains, labels, params, preds = self.get_step_outputs(batch, self.args.generation_validation_len)

        # Build output dictionary
        out = {"labels": labels.detach(), "preds": preds.detach(),
               "params": params.detach(), "signals": signals.detach()}

        # For code vector based models also add their local codes
        if hasattr(self.dynamics_func, 'embeddings'):
            out['code_vectors'] = self.dynamics_func.embeddings.detach()
        if hasattr(self.dynamics_func, 'ws'):
            out['ws'] = torch.concatenate([w.reshape([self.args.batch_size, -1]) for w in self.dynamics_func.ws], dim=1).detach()
        if hasattr(self.dynamics_func, 'bs'):
            out['bs'] = torch.concatenate([b.reshape([self.args.batch_size, -1]) for b in self.dynamics_func.bs], dim=1).detach()
        return out

    def test_epoch_end(self, outputs):
        """
        For testing end, save the predictions, gt, and MSE to NPY files in the respective experiment folder
        :param outputs: list of outputs from the validation steps at batch 0
        """
        # Consolidate arrays into lists from output dictionaries
        preds, signals, states, embeddings, labels, params = [], [], [], [], [], []
        code_vectors, ws, bs = [], [], []
        for output in outputs:
            preds.append(output["preds"])
            signals.append(output["signals"])
            labels.append(output["labels"])
            params.append(output["params"])

            if "code_vectors" in output.keys():
                code_vectors.append(output["code_vectors"])
            if "ws" in output.keys():
                ws.append(output["ws"])
            if "bs" in output.keys():
                bs.append(output["bs"])

        # Stack all tensors at once and do one cpu transfer each
        preds = torch.vstack(preds)
        signals = torch.vstack(signals)
        labels = torch.vstack(labels)
        params = torch.vstack(params)
        del outputs

        # Iterate through each metric function and add to a dictionary
        out_metrics = {}
        for met in self.args.metrics:
            metric_function = getattr(metrics, met)
            metric_mean, metric_std = metric_function(signals, preds, args=self.args, params=params)
            out_metrics[f"{met}_mean"], out_metrics[f"{met}_std"] = float(metric_mean), float(metric_std)
            print(f"=> {met}: {metric_mean:4.5f}+-{metric_std:4.5f}")

        if not os.path.exists(f"{self.args.ckpt_path}/test_files/"):
            os.mkdir(f"{self.args.ckpt_path}/test_files/")

        # Save metrics to JSON in checkpoint folder
        with open(f"{self.args.ckpt_path}/test_files/test_metrics.json", 'w+') as f:
            json.dump(out_metrics, f)

        # Save metrics to an easy excel conversion style
        with open(f"{self.args.ckpt_path}/test_files/test_excel.txt", 'w+') as f:
            for metric in self.args.metrics:
                f.write(f"{out_metrics[f'{metric}_mean']:0.3f}({out_metrics[f'{metric}_std']:0.3f}),")

        # Save files
        if self.args.testing['save_files'] is True:
            np.save(f"{self.args.ckpt_path}/test_files/test_recons.npy", preds.cpu().numpy())
            np.save(f"{self.args.ckpt_path}/test_files/test_signals.npy", signals.cpu().numpy())
            np.save(f"{self.args.ckpt_path}/test_files/test_labels.npy", labels.cpu().numpy())

        # Show side-by-side reconstructions
        show_sequences(signals[:10], preds[:10], f"{self.args.ckpt_path}/test_files/test_examples.png", num_out=5)