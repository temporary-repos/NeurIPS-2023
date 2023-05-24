import os
import json
import torch
import argparse
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import tqdm
from pde_experiments.utils.dataloader import LotkaVolterraGenerator
from pde_experiments.utils.utils import parse_args, get_exp_versions, find_best_epoch


def generate_dataset(params):
    """
    Handles generating a dataset of signals for a given set of Lotka-Volterra parameters.
    Saves it to a local .npz file to be reference within the for-loop
    """
    dataset_train_params = {
        "n_data_per_env": minibatch_size, "t_horizon": 10, "dt": 0.5,
        "method": "RK45", "group": "train", "params": [{
            "alpha": params[0], "beta": params[1], "gamma": params[2], "delta": params[3]
        }]
    }

    # Build datasets
    dataset = LotkaVolterraGenerator(**dataset_train_params)
    signals, labels, params = [], [], []
    for idx in range(dataset.len):
        data = dataset.__getitem__(idx)

        signals.append(data['state'])
        labels.append(data['env'])
        params.append(data['param'])

    signals = np.stack(signals)
    signals = np.swapaxes(signals, 1, 2)
    labels = np.stack(labels)
    np.savez(f"data/temp_dataset_{seed}.npz", signals=signals, labels=labels)


def plot_mape_diagram():
    """
    Handles plotting the MAPE percent for a testing set of 32 trajectories on
    a uniform 51x51 grid of gamma and delta, including training environment plotting
    and contours of errors
    """
    mapes = []
    indomain_mapes, outdomain_mapes = [], []
    cc75_mapes, cc50_mapes, cc25_mapes, cc00_mapes = [], [], [], []

    ccs = np.load('figures/ccs.npy')
    idx = 0
    for beta, delta in tqdm(zip(params_beta, params_delta), total=params_beta.shape[0]):
        # Generate the dataset to use
        generate_dataset([0.5, beta, 0.5, delta])

        # Initialize model
        model = model_type(args, top, exptop).cuda()

        # Get the best ckpt in training and load it into the model
        ckpt_path = args.ckpt_path + "/checkpoints/" + args.checkpt if args.checkpt != "None" \
            else f"{args.ckpt_path}/checkpoints/{find_best_epoch(args.ckpt_path)[0]}"

        model.load_state_dict(torch.load(ckpt_path)['state_dict'], strict=False)

        # Load in the dataset, generate context and inputs
        dataset = np.load(f"data/temp_dataset_{seed}.npz", allow_pickle=True)
        signals = dataset['signals']
        labels = dataset['labels']

        # Split into context and inputs
        inputs = torch.from_numpy(signals[:32]).cuda().to(torch.float32)
        context = torch.from_numpy(signals[32:]).cuda().to(torch.float32)

        # Define loss and optimizer
        mse = nn.MSELoss()
        optim = None

        # Optimize for a number of steps to the context set
        for bidx in range(10):
            preds, params = model(context, None, labels[32:], 20, finetune=True)

            if bidx == 0:
                optim = torch.optim.AdamW(params, lr=0.01)

            optim.zero_grad()
            loss = mse(preds, context)
            loss.backward()
            optim.step()

        # Predict on the test set
        with torch.no_grad():
            preds = model(inputs, None, labels[:32], 20)

        # Get the MAPE between the GT and Model phase space
        mape_top = torch.abs(preds - inputs)
        mape_bot = torch.abs(inputs).nanmean()
        mape = mape_top / mape_bot
        mape = mape.cpu().numpy()
        mapes.append(np.mean(mape))

        # Get the indomain MAPE
        if np.logical_and(0.5 <= beta, beta <= 1) & np.logical_and(0.5 <= delta, delta <= 1):
            indomain_mapes.append(mape)

        if ~np.logical_and(0.5 <= beta, beta <= 1) | ~np.logical_and(0.5 <= delta, delta <= 1):
            outdomain_mapes.append(mape)

        i, j = idx // 51, idx % 51
        idx += 1

        if ccs[i, j] >= 0.75:
            cc75_mapes.append(mape)
        elif ccs[i, j] >= 0.5 and ccs[i, j] < 0.75:
            cc50_mapes.append(mape)
        elif ccs[i, j] >= 0.25 and ccs[i, j] < 0.5:
            cc25_mapes.append(mape)
        elif ccs[i, j] < 0.25:
            cc00_mapes.append(mape)

    # Reshape back into grid
    mapes = np.reshape(np.array(mapes), [51, 51])

    # Output the JSON dictionary
    indomain_mapes = np.vstack(indomain_mapes)
    outdomain_mapes = np.vstack(outdomain_mapes)
    outputs_mape = {
        "rel_indomain_mean": float(np.mean(indomain_mapes)),
        "rel_indomain_std": float(np.std(indomain_mapes)),
        "rel_outdomain_mean": float(np.mean(outdomain_mapes)),
        "rel_outdomain_std": float(np.std(outdomain_mapes)),

        "rel_cc75_mean": float(np.mean(cc75_mapes)),
        "rel_cc75_std": float(np.std(cc75_mapes)),
        "rel_cc50_mean": float(np.mean(cc50_mapes)),
        "rel_cc50_std": float(np.std(cc50_mapes)),
        "rel_cc25_mean": float(np.mean(cc25_mapes)),
        "rel_cc25_std": float(np.std(cc25_mapes)),
        "rel_cc00_mean": float(np.mean(cc00_mapes)),
        "rel_cc00_std": float(np.std(cc00_mapes)),
    }

    # Save metrics to JSON in checkpoint folder
    with open(f"{args.ckpt_path}/test_files/test_full_metrics.json", 'w+') as f:
        json.dump(outputs_mape, f)

    # Plot heatmap
    fig, ax = plt.subplots()
    plt.imshow(mapes, cmap='coolwarm', vmin=0.00, vmax=0.2)
    plt.colorbar()
    plt.xlabel("<--- Beta --->")
    plt.ylabel("<--- Delta --->")
    plt.xticks(np.linspace(0, 50, 5), labels=['0.25', '0.5', '0.75', '1.00', '1.25'])
    plt.yticks(np.linspace(0, 50, 5), labels=reversed(['0.25', '0.5', '0.75', '1.00', '1.25']))

    # Plot contours of MAPE
    cnt = plt.contour(range(51), range(51), mapes, levels=[0.01, 0.025, 0.05, 0.10, 0.15, 0.2])
    ax.clabel(cnt, inline=1, fontsize=8)

    # Plot training environments
    plt.scatter(
        [int((fl - 0.25) * 51) for fl in [0.5, 0.75, 1.0, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0]],
        [int((fl - 0.25) * 51) for fl in [0.5, 0.5, 0.5, 0.75, 1.0, 0.75, 1.0, 0.75, 1.0]],
        c='yellow', s=10)
    plt.plot([int((0.5 - 0.25) * 51), int((1.0 - 0.25) * 51)], [int((0.5 - 0.25) * 51), int((0.5 - 0.25) * 51)], '--', c='yellow')
    plt.plot([int((0.5 - 0.25) * 51), int((0.5 - 0.25) * 51)], [int((0.5 - 0.25) * 51), int((1.0 - 0.25) * 51)], '--', c='yellow')
    plt.plot([int((0.5 - 0.25) * 51), int((1.0 - 0.25) * 51)], [int((1.0 - 0.25) * 51), int((1.0 - 0.25) * 51)], '--', c='yellow')
    plt.plot([int((1.0 - 0.25) * 51), int((1.0 - 0.25) * 51)], [int((1.0 - 0.25) * 51), int((0.5 - 0.25) * 51)], '--', c='yellow')
    plt.savefig(f"{args.ckpt_path}/mape_diagram.png")
    plt.close()


if __name__ == '__main__':
    global seed
    seed = np.random.randint(0, 100000)

    # Definition of parameters
    train_params = [
        [0.5, 0.5, 0.5, 0.5],
        [0.5, 0.75, 0.5, 0.5],
        [0.5, 1.0, 0.5, 0.5],
        [0.5, 0.5, 0.5, 0.75],
        [0.5, 0.5, 0.5, 1.0],
        [0.5, 0.75, 0.5, 0.75],
        [0.5, 0.75, 0.5, 1.0],
        [0.5, 1.0, 0.5, 0.75],
        [0.5, 1.0, 0.5, 1.]
    ]

    # Get grid of parameters
    alpha = 0.5
    betas = np.linspace(0.25, 1.25, 51)
    gamma = 0.5
    deltas = np.linspace(1.25, 0.25, 51)
    params_beta, params_delta = np.meshgrid(betas, deltas)
    params_beta = params_beta.ravel()
    params_delta = params_delta.ravel()

    # Define environment parameters
    minibatch_size = 35

    # Define the parser with the configuration file path
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str,
                        default="experiments/coda_additive/coda_additive/version_1/CoDA_Additive.json",
                        help='name of the configuration .json to load')

    # Parse the config file and get the model function used
    args, model_type = parse_args(parser)
    args.batch_size = minibatch_size - 3

    # Get version numbers
    global top, exptop
    top, exptop = get_exp_versions(args.model, args.exptype, to_stdout=False)

    # Get the 51x51 MAPE diagram across environments
    plot_mape_diagram()
