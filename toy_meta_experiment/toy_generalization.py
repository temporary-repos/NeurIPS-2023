"""
@file meta_main.py

Holds the general training script for the meta-learning models, defining a dataset and model to train on
"""
import torch
import argparse
import pytorch_lightning

from torch.utils.data import DataLoader
from utils.dataloader import MetaPercentODEData
from utils.utils import find_best_epoch, parse_args, get_exp_versions, strtobool
from pytorch_lightning.callbacks import ModelCheckpoint


if __name__ == '__main__':
    # Set a consistent seed over the full set for consistent analysis
    pytorch_lightning.seed_everything(125125125, workers=True)

    # For Ampere-level GPUs, set the matmul precision to be mixed
    torch.set_float32_matmul_precision('high')

    # Define the parser with the configuration file path
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/bounds_experiments_additiveODE/MetaAdditive.json',
                        help='path and name of the configuration .json to load')
    parser.add_argument('--train', type=strtobool, default=True,
                        help='whether to train or test the given model')
    parser.add_argument('--interpolation', type=strtobool, default=False)

    # Parse the config file and get the model function used
    args, model_type = parse_args(parser)

    # Get version numbers
    global top, exptop
    top, exptop = get_exp_versions(args.model, args.exptype)

    # Input generation
    dataset = MetaPercentODEData(args, bounds=(args.bound_start, args.bound_end),
                                 u_start=args.u_start, u_end=args.u_end,
                                 is_interpolation=args.interpolation, shuffle=True, is_train=args.train)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize model
    model = model_type(args, top, exptop)

    # Callbacks for checkpointing and early stopping
    checkpoint_callback = ModelCheckpoint(monitor=f'train_recon_mse',
                                          filename='epoch{epoch:02d}-train_recon_mse{train_recon_mse:.6f}',
                                          auto_insert_metric_name=False, save_last=True)

    # Initialize trainer
    trainer = pytorch_lightning.Trainer.from_argparse_args(
        args,
        callbacks=[checkpoint_callback],
        max_epochs=args.num_epochs,
        auto_select_gpus=True,
        gradient_clip_val=1.0
    )

    # Starting training from scratch
    if args.train is True:
        trainer.fit(model, dataloader)
    else:
        ckpt_path = args.ckpt_path + "/checkpoints/" + args.checkpt if args.checkpt != "None" \
            else f"{args.ckpt_path}/checkpoints/{find_best_epoch(args.ckpt_path)[0]}"

        trainer.test(model, dataloader, ckpt_path=ckpt_path)
