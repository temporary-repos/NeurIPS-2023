"""
@file meta_main.py

Holds the general training script for the meta-learning models, defining a dataset and model to train on
"""
import os
import torch
import shutil
import argparse
import numpy as np
import pytorch_lightning

from torch.utils.data import DataLoader
from utils.dataloader import LVDataset
from utils.utils import find_best_epoch, parse_args, get_exp_versions, strtobool
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

if __name__ == '__main__':
    # Set a consistent seed over the full set for consistent analysis
    pytorch_lightning.seed_everything(125125125, workers=True)

    # For Ampere-level GPUs, set the matmul precision to be mixed
    torch.set_float32_matmul_precision('high')

    # Define the parser with the configuration file path
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/CoDA_Additive.json',
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
    train_dataset = LVDataset(
        os.path.abspath('').replace('\\', '/') + f"/data/{args.dataset}/{args.dataset_ver}_train.npz",
        {'k_shot': args.domain_size, 'shuffle': True, 'dataset_percent': args.dataset_percent})
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    test_dataset = LVDataset(
        os.path.abspath('').replace('\\', '/') + f"/data/{args.dataset}/{args.dataset_ver}_test.npz",
        {'k_shot': args.domain_size, 'shuffle': False, 'dataset_percent': args.dataset_percent})
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    # Set number of steps to lower if there are more than one batch/epoch
    args.num_epochs = int(args.num_epochs // ((train_dataset.signals.shape[0] - (np.unique(train_dataset.labels).shape[0] * args.domain_size)) // args.batch_size))
    print(f"=> Number of Epochs: {args.num_epochs}")
    args.check_every_n_steps = int(args.num_epochs // 120)
    print(f"=> Check val every N steps: {args.check_every_n_steps}")

    # Initialize model
    model = model_type(args, top, exptop)

    # Callbacks for checkpointing and early stopping
    checkpoint_callback = ModelCheckpoint(monitor=f'val_rel_outdomain',
                                          filename='epoch{epoch:02d}-val_rel_outdomain{val_rel_outdomain:.6f}',
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
        gradient_clip_val=args.gradient_clip_val,
        check_val_every_n_epoch=args.check_every_n_steps,
        num_sanity_val_steps=0,
        auto_select_gpus=True
    )

    # Starting training from scratch
    if args.train is True and args.resume is False:
        trainer.fit(model, train_dataloader, test_dataloader)

        # Get the best ckpt in training
        ckpt_path = args.ckpt_path + "/checkpoints/" + args.checkpt if args.checkpt != "None" \
            else f"{args.ckpt_path}/checkpoints/{find_best_epoch(args.ckpt_path)[0]}"

        trainer.test(model, test_dataloader, ckpt_path=ckpt_path)

    # If resuming training, get the last ckpt if not given one
    elif args.train is True and args.resume is True:
        ckpt_path = args.ckpt_path + "/checkpoints/" + args.checkpt if args.checkpt != "None" \
            else f"{args.ckpt_path}/checkpoints/last.ckpt"

        trainer.fit(model, train_dataloader, test_dataloader, ckpt_path=ckpt_path)

    # Otherwise test the model with the best epoch in training
    else:
        ckpt_path = args.ckpt_path + "/checkpoints/" + args.checkpt if args.checkpt != "None" \
            else f"{args.ckpt_path}/checkpoints/{find_best_epoch(args.ckpt_path)[0]}"

        trainer.test(model, test_dataloader, ckpt_path=ckpt_path)

    # Remove the lightning log folder
    shutil.rmtree(f"lightning_logs/version_{top}")
