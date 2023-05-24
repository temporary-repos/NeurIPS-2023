"""
@file meta_main.py

Holds the general training script for the meta-learning models, defining a dataset and model to train on
"""
import torch
import argparse
import numpy as np
import pytorch_lightning
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.dataloader import SingleODEData, MetaODEData
from utils.utils import find_best_epoch, parse_args, get_exp_versions


if __name__ == '__main__':
    # Set a consistent seed over the full set for consistent analysis
    pytorch_lightning.seed_everything(125125125, workers=True)

    # For Ampere-level GPUs, set the matmul precision to be mixed
    torch.set_float32_matmul_precision('high')

    # Define the parser with the configuration file path
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/multiplicative_multiplicative/modelMultiplicative_funcMultiplicative_dim1/modelMultiplicative_funcMultiplicative_dim1_4hidden.json',
                        help='path and name of the configuration .json to load')

    # Parse the config file and get the model function used
    args, model_type = parse_args(parser)

    # Get version numbers
    global top, exptop
    top, exptop = get_exp_versions(args.model, args.exptype)

    # Input generation
    if args.meta is True:
        dataset = MetaODEData(args)
    else:
        x_indices = np.random.randint(0, 25, size=20)
        dataset = SingleODEData(args, x_initials=x_indices[:args.in_dim])
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False)

    # Initialize model
    model = model_type(args, top, exptop)

    # Get the best ckpt in training and load it into the model
    ckpt_path = args.ckpt_path + "/checkpoints/" + args.checkpt if args.checkpt != "None" \
        else f"{args.ckpt_path}/checkpoints/{find_best_epoch(args.ckpt_path)[0]}"
    print(f"=> Ckpt: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path)['state_dict'], strict=True)

    with torch.no_grad():
        preds, gts = [], []
        for batch in tqdm(dataloader):
            # Stack batch and restrict to generation length
            indices, signals, controls = batch

            # Get predictions
            pred = model(signals, controls, args.timesteps)
            preds.append(pred)
            gts.append(signals)

        preds = torch.vstack(preds)
        gts = torch.vstack(gts)

        print(preds.shape, gts.shape)
        mse = (preds - gts) ** 2

    plt.plot(range(20), gts[:, :, 0].detach().cpu().numpy().T, c='b')
    plt.plot(range(20), preds[:, :, 0].detach().cpu().numpy().T, c='k', linestyle='--')
    plt.title(f"Plot of U[-2, 2]")
    plt.xlabel("Blue: GT | Black: Preds")
    plt.savefig(f"{args.ckpt_path}/reconstructedControls.png")
    plt.close()
