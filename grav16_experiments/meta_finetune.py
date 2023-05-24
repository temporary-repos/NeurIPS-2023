"""
@file meta_main.py

Holds the general training script for the meta-learning models, defining a dataset and model to train on
"""
import os
import json
import torch
import shutil
import argparse
import numpy as np
import pytorch_lightning

from pytorch_lightning.callbacks import ModelCheckpoint
from utils.dataloader import EpisoticDataLoader
from utils.utils import get_exp_versions, parse_args, find_best_epoch


if __name__ == '__main__':
    # Set a consistent seed over the full set for consistent analysis
    pytorch_lightning.seed_everything(125125125, workers=True)

    # For Ampere-level GPUs, set the matmul precision to be mixed
    torch.set_float32_matmul_precision('high')

    # Define the parser with the configuration file path
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/grav16_baselines/CoDA.json',
                        help='path and name of the configuration .json to load')

    # Parse the config file and get the model function used
    args, model_type = parse_args(parser)

    # Get version numbers
    global top, exptop
    top, exptop = get_exp_versions(args.model, args.exptype)

    # Get exptype name and training z_amort
    original_path = args.ckpt_path
    exptype = args.exptype
    z_amort = args.z_amort

    # Set batch size to match domain size
    args.batch_size = args.domain_size

    # Callbacks for checkpointing and early stopping
    checkpoint_callback = ModelCheckpoint(monitor='val_reconstruction_mse',
                                          filename='epoch{epoch:02d}-val_reconstruction_mse{val_reconstruction_mse:.4f}',
                                          auto_insert_metric_name=False, save_last=True)

    # Get the checkpoint path from the best model
    best_ckpt_path, best_ckpt_epoch = find_best_epoch(args.ckpt_path)
    if args.checkpt != "None":
        ckpt_path = args.ckpt_path + "/checkpoints/" + args.checkpt
    else:
        ckpt_path = f"{args.ckpt_path}/checkpoints/{best_ckpt_path}"

    # For every split in the test, finetune for a number of epochs then
    if "train" in args.testing['splits']:
        args.testing['splits'] = args.testing['splits'][1:]

    for split, finetune_split in zip(args.testing['splits'], args.testing['finetune_splits']):
        print(split, finetune_split)

        # Set exptype to include the split
        args.exptype = f'{exptype}_finetune_{finetune_split}'
        args.ckpt_path = ckpt_path
        args.z_amort = z_amort

        # Initialize trainer
        trainer = pytorch_lightning.Trainer.from_argparse_args(
            args,
            callbacks=[checkpoint_callback],
            deterministic=True,
            max_epochs=args.testing['num_finetuning_epochs'] + best_ckpt_epoch,
            gradient_clip_val=5.0,
            check_val_every_n_epoch=1,
            num_sanity_val_steps=0,
            auto_select_gpus=True
        )

        # Initialize model
        model = model_type(args, top, exptop)

        # Get support set to finetune on based on the given domain size
        train_dataset = EpisoticDataLoader(args=args, split=finetune_split)
        train_dataset.dataset.images = train_dataset.dataset.images[:args.domain_size]
        train_dataset.dataset.state = train_dataset.dataset.state[:args.domain_size]
        train_dataset.dataset.labels = train_dataset.dataset.labels[:args.domain_size]
        train_dataset.dataset.qry_idx = train_dataset.dataset.qry_idx[:args.domain_size]
        train_dataset.dataset.label_idx[train_dataset.dataset.labels[0]] = \
            train_dataset.dataset.label_idx[train_dataset.dataset.labels[0]][:args.domain_size]
        train_dataset.dataset.split()

        # Finetune model
        trainer.fit(model, train_dataset, train_dataset, ckpt_path=ckpt_path)

        # Get test dataset
        test_dataset = EpisoticDataLoader(args=args, split=split, shuffle=False)

        # Test the fine-tuned model on the given split
        args.split = split
        args.z_amort = args.testing['z_amort_testing']
        args.ckpt_path = f"experiments/{args.exptype}/{args.model}/version_2/"
        trainer.test(model, test_dataset, ckpt_path=f"{args.ckpt_path}/checkpoints/last.ckpt")

        # Move the finetuning experiment over to its base folder
        os.rename(f"experiments/{args.exptype}/{args.model}/version_2/test_{split}/",
                  f"{original_path}/test_finetune_{finetune_split}/")
        shutil.rmtree(f"experiments/{args.exptype}/", ignore_errors=True)

    # Aggregate all metrics into list
    metrics = {
        'reconstruction_mse_mean': [], 'reconstruction_mse_std': [],
        'vpt_mean': [], 'vpt_std': [],
        'dst_mean': [], 'dst_std': [],
        'vpd_mean': [], 'vpd_std': [],
    }
    query_idx = []
    for query in os.listdir(original_path):
        if "finetune" not in query or ".txt" in query or "train" in query:
            continue

        query_cur_idx = int(query.split('_')[-1])
        query_idx.append(query_cur_idx)

        if "unknown" in query:
            metric = json.load(open(f"{original_path}/{query}/test_unknown_qry_{query_cur_idx}_metrics.json", 'r'))
        else:
            metric = json.load(open(f"{original_path}/{query}/test_qry_{query_cur_idx}_metrics.json", 'r'))

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
    with open(f"{original_path}/test_all_finetune_excel.txt", 'a') as f:
        f.write(f"\n{np.mean(mse_means):0.4f}({np.mean(mse_stds):0.4f}),"
                f"{np.mean(vpt_means):0.3f}({np.mean(vpt_stds):0.3f}),"
                f"{np.mean(dst_means):0.3f}({np.mean(dst_stds):0.3f}),"
                f"{np.mean(vpd_means):0.3f}({np.mean(vpd_stds):0.3f})")

    """ Known grav metrics """
    # Save metrics to an easy excel conversion style
    known = [0, 2, 3, 5, 6, 7, 11, 13, 14, 15]
    with open(f"{original_path}/test_known_finetune_excel.txt", 'a') as f:
        f.write(f"\n{np.mean(mse_means[known]):0.4f}({np.mean(mse_stds[known]):0.4f}),"
                f"{np.mean(vpt_means[known]):0.3f}({np.mean(vpt_stds[known]):0.3f}),"
                f"{np.mean(dst_means[known]):0.3f}({np.mean(dst_stds[known]):0.3f}),"
                f"{np.mean(vpd_means[known]):0.3f}({np.mean(vpd_stds[known]):0.3f})")

    """ Unknown grav metrics """
    # Save metrics to an easy excel conversion style
    unknown = [1, 4, 8, 9, 10, 12]
    with open(f"{original_path}/test_unknown_finetune_excel.txt", 'a') as f:
        f.write(f"\n{np.mean(mse_means[unknown]):0.4f}({np.mean(mse_stds[unknown]):0.4f}),"
                f"{np.mean(vpt_means[unknown]):0.3f}({np.mean(vpt_stds[unknown]):0.3f}),"
                f"{np.mean(dst_means[unknown]):0.3f}({np.mean(dst_stds[unknown]):0.3f}),"
                f"{np.mean(vpd_means[unknown]):0.3f}({np.mean(vpd_stds[unknown]):0.3f})")
