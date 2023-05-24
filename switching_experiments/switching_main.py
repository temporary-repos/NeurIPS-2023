"""
@file switching_main.py

Holds the general training script for the meta-learning models, defining a dataset and model to train on
"""
import os
import torch
import argparse
import pytorch_lightning
from torch.utils.data import DataLoader

from utils.dataloader import SwitchingDataset
from utils.utils import find_best_epoch, parse_args, get_exp_versions, strtobool
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


if __name__ == '__main__':
    # Set a consistent seed over the full set for consistent analysis
    pytorch_lightning.seed_everything(125125125, workers=True)

    # For Ampere-level GPUs, set the matmul precision to be mixed
    torch.set_float32_matmul_precision('high')

    # Define the parser with the configuration file path
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/Nascar_MetaHyperNet_GP.json',
                        help='path and name of the configurati2on .json to load')
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
    train_dataset = SwitchingDataset(os.path.abspath('').replace('\\', '/') + f"/data/{args.dataset}/{args.dataset_ver}_train.npz")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    test_dataset = SwitchingDataset(os.path.abspath('').replace('\\', '/') + f"/data/{args.dataset}/{args.dataset_ver}_test.npz")
    test_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

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
        gradient_clip_val=1.0,
        check_val_every_n_epoch=args.check_every_n_steps,
        num_sanity_val_steps=0,
        auto_select_gpus=True
    )

    # Starting training from scratch
    if args.train is True and args.resume is False:
        trainer.fit(model, train_dataloader, test_dataloader)

    # If resuming training, get the last ckpt if not given one
    elif args.train is True and args.resume is True:
        ckpt_path = args.ckpt_path + "/checkpoints/" + args.checkpt if args.checkpt != "None" \
            else f"{args.ckpt_path}/checkpoints/last.ckpt"

        trainer.fit(model, train_dataloader, test_dataloader, ckpt_path=ckpt_path)

    else:
        # Get the best ckpt in training
        ckpt_path = args.ckpt_path + "/checkpoints/" + args.checkpt if args.checkpt != "None" \
            else f"{args.ckpt_path}/checkpoints/{find_best_epoch(args.ckpt_path)[0]}"

        trainer.test(model, train_dataloader, ckpt_path=ckpt_path)
