"""
@file meta_main.py

Holds the general training script for the meta-learning models, defining a dataset and model to train on
"""
import os
import json
import torch
import argparse
import numpy as np
import pytorch_lightning

from utils.dataloader import EpisoticDataLoader, EpochDataLoader
from utils.utils import find_best_epoch, parse_args, get_exp_versions, strtobool
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


if __name__ == '__main__':
    # Set a consistent seed over the full set for consistent analysis
    pytorch_lightning.seed_everything(125125125, workers=True)

    # For Ampere-level GPUs, set the matmul precision to be mixed
    torch.set_float32_matmul_precision('high')

    # Define the parser with the configuration file path
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/grav16_baselines/metahyperssm.json',
                        help='path and name of the configuration .json to load')
    parser.add_argument('--train', type=strtobool, default=True,
                        help='whether to train or test the given model')
    parser.add_argument('--resume', type=strtobool, default=False,
                        help='whether to continue training from the checkpoint in the config')

    # Parse the config file and get the model function used
    args, model_type = parse_args(parser)

    # Get version numbers
    global top, exptop
    top, exptop = get_exp_versions(args.model, args.exptype)

    # Input generation
    if args.meta is True:
        dataset = EpisoticDataLoader(args=args, split='train')
        val_dataset = EpisoticDataLoader(args=args, split='valid', shuffle=False)
    else:
        dataset = EpochDataLoader(args=args, split='train')
        val_dataset = EpochDataLoader(args=args, split='valid', shuffle=False)

    # Initialize model
    model = model_type(args, top, exptop)

    # Callbacks for checkpointing and early stopping
    checkpoint_callback = ModelCheckpoint(monitor='val_reconstruction_mse',
                                          filename='epoch{epoch:02d}-val_reconstruction_mse{val_reconstruction_mse:.4f}',
                                          auto_insert_metric_name=False, save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Initialize trainer
    trainer = pytorch_lightning.Trainer.from_argparse_args(
        args,
        callbacks=[
            lr_monitor,
            checkpoint_callback
        ],
        deterministic=True,
        max_epochs=args.num_epochs,
        gradient_clip_val=5.0,
        check_val_every_n_epoch=10,
        num_sanity_val_steps=0,
        auto_select_gpus=True
    )

    # Starting training from scratch
    if args.train is True and args.resume is False:
        trainer.fit(model, dataset, val_dataset)

    # If resuming training, get the last ckpt if not given one
    elif args.train is True and args.resume is True:
        ckpt_path = args.ckpt_path + "/checkpoints/" + args.checkpt if args.checkpt != "None" \
            else f"{args.ckpt_path}/checkpoints/last.ckpt"

        trainer.fit(
            model, dataset, val_dataset,
            ckpt_path=f"{args.ckpt_path}/checkpoints/{os.listdir(f'{args.ckpt_path}/checkpoints/')[-1]}"
        )

    # Otherwise test the model for the given splits on the best ckpt
    elif args.train is False:
        # Get the best ckpt in training
        ckpt_path = args.ckpt_path + "/checkpoints/" + args.checkpt if args.checkpt != "None" \
            else f"{args.ckpt_path}/checkpoints/{find_best_epoch(args.ckpt_path)[0]}"

        # Test over each split
        args.z_amort = args.testing['z_amort_testing']
        for split in args.testing['splits']:
            test_dataset = EpisoticDataLoader(args=args, split=split, shuffle=False) \
                if args.meta is True else EpochDataLoader(args=args, split=split, shuffle=False)

            args.split = split
            trainer.test(model, test_dataset, ckpt_path=ckpt_path)

        # Aggregate all metrics into list
        metrics = {
            'reconstruction_mse_mean': [], 'reconstruction_mse_std': [],
            'vpt_mean': [], 'vpt_std': [],
            'dst_mean': [], 'dst_std': [],
            'vpd_mean': [], 'vpd_std': [],
        }
        query_idx = []
        for query in os.listdir(args.ckpt_path):
            if "test" not in query or ".txt" in query or "train" in query:
                continue

            query_idx.append(int(query.split('_')[-1]))
            metric = json.load(open(f"{args.ckpt_path}/{query}/{query}_metrics.json", 'r'))

            for m in metric.keys():
                if m in metrics.keys():
                    metrics[m].append(metric[m])

        # Sort indices by folder idx rather than listdir order
        sorted_indices = np.argsort(query_idx)
        mse_means = np.array(metrics['reconstruction_mse_mean'])[sorted_indices]
        mse_stds = np.array(metrics['reconstruction_mse_std'])[sorted_indices]
        vpt_means = np.array(metrics['vpt_mean'])[sorted_indices]
        vpt_stds = np.array(metrics['vpt_std'])[sorted_indices]
        dst_means = np.array(metrics['dst_mean'])[sorted_indices]
        dst_stds = np.array(metrics['dst_std'])[sorted_indices]
        vpd_means = np.array(metrics['vpd_mean'])[sorted_indices]
        vpd_stds = np.array(metrics['vpd_std'])[sorted_indices]

        """ Global metrics """
        # Save metrics to an easy excel conversion style
        with open(f"{args.ckpt_path}/test_all_excel.txt", 'a') as f:
            f.write(f"\n{np.mean(mse_means):0.4f}({np.mean(mse_stds):0.4f}) & "
                    f"{np.mean(vpt_means):0.3f}({np.mean(vpt_stds):0.3f}) & "
                    f"{np.mean(dst_means):0.3f}({np.mean(dst_stds):0.3f}) & "
                    f"{np.mean(vpd_means):0.3f}({np.mean(vpd_stds):0.3f}) ")

        """ Known grav metrics """
        # Save metrics to an easy excel conversion style
        known = [0, 2, 3, 5, 6, 7, 11, 13, 14, 15]
        with open(f"{args.ckpt_path}/test_known_excel.txt", 'a') as f:
            f.write(f"\n{np.mean(mse_means[known]):0.4f}({np.mean(mse_stds[known]):0.4f}) & "
                    f"{np.mean(vpt_means[known]):0.3f}({np.mean(vpt_stds[known]):0.3f}) & "
                    f"{np.mean(dst_means[known]):0.3f}({np.mean(dst_stds[known]):0.3f}) & "
                    f"{np.mean(vpd_means[known]):0.3f}({np.mean(vpd_stds[known]):0.3f})")

        """ Unknown grav metrics """
        # Save metrics to an easy excel conversion style
        unknown = [1, 4, 8, 9, 10, 12]
        with open(f"{args.ckpt_path}/test_unknown_excel.txt", 'a') as f:
            f.write(f"\n{np.mean(mse_means[unknown]):0.4f}({np.mean(mse_stds[unknown]):0.4f}) & "
                    f"{np.mean(vpt_means[unknown]):0.3f}({np.mean(vpt_stds[unknown]):0.3f}) & "
                    f"{np.mean(dst_means[unknown]):0.3f}({np.mean(dst_stds[unknown]):0.3f}) & "
                    f"{np.mean(vpd_means[unknown]):0.3f}({np.mean(vpd_stds[unknown]):0.3f})")
