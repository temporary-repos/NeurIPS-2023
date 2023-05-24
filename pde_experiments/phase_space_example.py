import json
import torch
import argparse
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from openTSNE import TSNE

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


def get_phase_space(env_id, b, g):
    """ Handles getting a phase space portrait for a given beta and gamma set of the LV-Dataset """
    def dX_dt(t, X):
        """ Return the growth rate of fox and rabbit populations. """
        z = np.zeros(2)
        z[0] = 0.5 * X[0] - b * X[0] * X[1]
        z[1] = g * X[0] * X[1] - 0.5 * X[1]
        return z

    # -------------------------------------------------------
    # define a grid and compute direction at each point
    nb_points = 30
    lb, ub = 0, 6
    x = np.linspace(lb, ub, nb_points)
    y = np.linspace(lb, ub, nb_points)

    # Build input data for test
    inputs_x, inputs_y = np.meshgrid(x, y)
    inputs = np.stack((inputs_x, inputs_y), axis=2)
    inputs = np.reshape(inputs, [-1, 2])

    DX1, DY1 = [], []
    for input in inputs:
        dx, dy = dX_dt(None, input)
        DX1.append(dx)
        DY1.append(dy)

    DX1 = np.reshape(np.stack(DX1), [nb_points, nb_points])
    DY1 = np.reshape(np.stack(DY1), [nb_points, nb_points])
    M = (np.hypot(DX1, DY1))  # Norm of the growth rate
    M[M == 0] = 1.  # Avoid zero division errors
    DX1 /= M  # Normalize each arrows
    DY1 /= M

    # Initialize model
    model = model_type(args, top, exptop)

    # Get the best ckpt in training and load it into the model
    ckpt_path = args.ckpt_path + "/checkpoints/" + args.checkpt if args.checkpt != "None" \
        else f"{args.ckpt_path}/checkpoints/{find_best_epoch(args.ckpt_path)[0]}"

    model.load_state_dict(torch.load(ckpt_path)['state_dict'], strict=False)

    # Build input data for test
    inputs = torch.from_numpy(inputs).to(torch.float32).unsqueeze(1)

    # Get context set
    context = np.load("data/lotka_volterra/lv_test.npz", allow_pickle=True)
    context_signals = context['signals']
    context_labels = context['labels']
    context_signals = context_signals[np.where(context_labels == env_id)]

    rand_indices = np.random.choice(range(context_signals.shape[0]), 3, replace=False)
    context_signals = torch.from_numpy(context_signals[rand_indices]).repeat(inputs.shape[0], 1, 1, 1)
    context_labels = context_labels[rand_indices]

    # Get predictions and gradients from model
    with torch.no_grad():
        preds = model(inputs, context_signals, context_labels, 100)

    gradients = model.dynamics_func.vector_gradients[0]
    gradients = gradients.reshape([nb_points ** 2, 2]).reshape([nb_points, nb_points, 2])
    gradients_x, gradients_y = gradients[:, :, 0], gradients[:, :, 1]
    M = (np.hypot(gradients_x, gradients_y))  # Norm of the growth rate
    M[M == 0] = 1.  # Avoid zero division errors
    gradients_x /= M  # Normalize each arrows
    gradients_y /= M

    # -------------------------------------------------------
    grads, dxs = np.stack((gradients_x, gradients_y), axis=2), np.stack((DX1, DY1), axis=2)

    # Get the MAPE between the GT and Model phase space
    mape = np.mean((np.abs(grads - dxs) + 1e-6) / (np.abs(dxs) + 1e-6))
    print(f"=>MAPE for {env_id}: {mape}")

    # -------------------------------------------------------
    # Drow direction fields, using matplotlib 's quiver function
    fig, ax = plt.subplots()
    plt.title(f'Trajectories and direction fields for Env {env_id}')
    plt.grid()

    # Plot the section of initial conditions for the model
    pp1 = plt.Rectangle((1, 1), 2, 2, alpha=0.5)
    ax.add_patch(pp1)

    # Plot the GT and Model phase spaces
    plt.quiver(inputs_x, inputs_y, DX1, DY1)
    plt.quiver(inputs_x, inputs_y, gradients_x, gradients_y, color='blue')
    plt.xlabel('Number of rabbits')
    plt.ylabel('Number of foxes')
    plt.legend(['Init Cond.', 'GT', 'Pred'])
    plt.xlim(lb, ub)
    plt.ylim(lb, ub)
    plt.show()


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
        model = model_type(args, top, exptop)

        # Get the best ckpt in training and load it into the model
        ckpt_path = args.ckpt_path + "/checkpoints/" + args.checkpt if args.checkpt != "None" \
            else f"{args.ckpt_path}/checkpoints/{find_best_epoch(args.ckpt_path)[0]}"

        model.load_state_dict(torch.load(ckpt_path)['state_dict'], strict=False)

        # Load in the dataset, generate context and inputs
        dataset = np.load(f"data/temp_dataset_{seed}.npz", allow_pickle=True)
        signals = dataset['signals']
        labels = dataset['labels']

        # Split into context and inputs
        inputs = torch.from_numpy(signals[:32]).to(torch.float32)
        context = torch.from_numpy(signals[32:]).to(torch.float32).repeat(minibatch_size - 3, 1, 1, 1)

        # Get predictions and gradients from model
        try:
            with torch.no_grad():
                preds = model(inputs, context, labels, 20)
        except:
            mapes.append(1.0)
            print(f"Error in solution of params {beta, delta}")
            continue

        # Get the MAPE between the GT and Model phase space
        mape_top = torch.abs(preds - inputs)
        mape_bot = torch.abs(inputs).nanmean()
        mape = mape_top / mape_bot
        mapes.append(torch.mean(mape))

        # Get the indomain MAPE
        if np.logical_and(0.5 <= beta, beta <= 1) & np.logical_and(0.5 <= delta, delta <= 1):
            indomain_mapes.append(mape.numpy())

        if ~np.logical_and(0.5 <= beta, beta <= 1) | ~np.logical_and(0.5 <= delta, delta <= 1):
            outdomain_mapes.append(mape.numpy())

        i, j = idx // 51, idx % 51
        idx += 1

        if ccs[i, j] >= 0.75:
            cc75_mapes.append(mape.numpy())
        elif ccs[i, j] >= 0.5 and ccs[i, j] < 0.75:
            cc50_mapes.append(mape.numpy())
        elif ccs[i, j] >= 0.25 and ccs[i, j] < 0.5:
            cc25_mapes.append(mape.numpy())
        elif ccs[i, j] < 0.25:
            cc00_mapes.append(mape.numpy())

    os.system(f'rm data/temp_dataset.npz')
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


def plot_parameter_distributions():
    """
    Over the uniform parameter grid, plots the derived local codes from the given context set
    and colors it by the base parameter being changed.
    """
    # Get training embeddings
    train_parameters = []
    for ps in train_params:
        # Generate the dataset to use
        generate_dataset([0.5, ps[1], 0.5, ps[3]])

        # Initialize model
        model = model_type(args, top, exptop)

        # Get the best ckpt in training and load it into the model
        ckpt_path = args.ckpt_path + "/checkpoints/" + args.checkpt if args.checkpt != "None" \
            else f"{args.ckpt_path}/checkpoints/{find_best_epoch(args.ckpt_path)[0]}"

        model.load_state_dict(torch.load(ckpt_path)['state_dict'], strict=False)

        # Load in the dataset, generate context and inputs
        dataset = np.load("data/temp_dataset.npz", allow_pickle=True)
        signals = dataset['signals']
        labels = dataset['labels']

        # Split into context and inputs
        inputs = torch.from_numpy(signals[:32]).to(torch.float32)
        context = torch.from_numpy(signals[32:]).to(torch.float32).repeat(minibatch_size - 3, 1, 1, 1)

        # Get predictions and gradients from model
        with torch.no_grad():
            _ = model(inputs, context, labels, 20)

        # Get the Code vector for this context set
        if hasattr(model.dynamics_func, 'embeddings'):
            params = model.dynamics_func.embeddings[0]
        elif hasattr(model.dynamics_func, 'weights_out'):
            params = model.dynamics_func.weights_out[0]
        else:
            raise NotImplementedError("Model doesn't have parameters to visualize!")
        train_parameters.append(params)

    # Do the loop
    parameters = []
    for beta, delta in tqdm(zip(params_beta, params_delta), total=params_beta.shape[0]):
        # Generate the dataset to use
        generate_dataset([0.5, beta, 0.5, delta])

        # Initialize model
        model = model_type(args, top, exptop)

        # Get the best ckpt in training and load it into the model
        ckpt_path = args.ckpt_path + "/checkpoints/" + args.checkpt if args.checkpt != "None" \
            else f"{args.ckpt_path}/checkpoints/{find_best_epoch(args.ckpt_path)[0]}"

        model.load_state_dict(torch.load(ckpt_path)['state_dict'], strict=False)

        # Load in the dataset, generate context and inputs
        dataset = np.load("data/temp_dataset.npz", allow_pickle=True)
        signals = dataset['signals']
        labels = dataset['labels']

        # Split into context and inputs
        inputs = torch.from_numpy(signals[:32]).to(torch.float32)
        context = torch.from_numpy(signals[32:]).to(torch.float32).repeat(minibatch_size - 3, 1, 1, 1)

        # Get predictions and gradients from model
        try:
            with torch.no_grad():
                _ = model(inputs, context, labels, 20)
        except:
            parameters.append(torch.zeros_like(params))
            continue

        # Get the Code vector for this context set
        if hasattr(model.dynamics_func, 'embeddings'):
            params = model.dynamics_func.embeddings[0]
        elif hasattr(model.dynamics_func, 'weights_out'):
            params = model.dynamics_func.weights_out[0]
        else:
            raise NotImplementedError("Model doesn't have parameters to visualize!")
        parameters.append(params)

    # Reshape back into grid
    parameters = torch.stack(parameters).cpu().numpy()
    train_parameters = torch.stack(train_parameters).cpu().numpy()
    print(np.min(parameters), np.max(parameters))

    if parameters.shape[1] > 2:
        # Get a TSNE plot over the codes
        tsne = TSNE(n_components=2, perplexity=30, initialization="pca", metric="cosine", n_jobs=8, random_state=3)
        parameters = tsne.fit(parameters)

    # Plot lattice work
    plt.figure()
    plt.scatter(train_parameters[:, 0], train_parameters[:, 1], c='y', marker='x', s=10)
    plt.scatter(parameters[:, 0], parameters[:, 1], c=params_beta, cmap='coolwarm')
    plt.title("Scatterplot of Latent Weight Codes over LV Beta")
    plt.colorbar()
    plt.savefig(f"{args.ckpt_path}/codes_beta.png")
    plt.close()

    plt.figure()
    plt.scatter(train_parameters[:, 0], train_parameters[:, 1], c='y', marker='x', s=10)
    plt.scatter(parameters[:, 0], parameters[:, 1], c=params_delta, cmap='coolwarm')
    plt.title("Scatterplot of Latent Weight Codes over LV Delta")
    plt.colorbar()
    plt.savefig(f"{args.ckpt_path}/codes_delta.png")
    plt.close()


def plot_training_slice_reconstructions():
    """
    Handles plotting out the reconstructions across an interpolation of the beta dimension
    for the 3 rows that the delta parameters exist on in the training data. Gives a sense
    of how the signals change in and out of domain and how the model performs at those
    farther ranges.
    """
    norm = Normalize(vmin=-0.25, vmax=1.25)

    for delta in [0.5, 0.75, 1.0]:
        plt.figure()
        for beta in tqdm(np.linspace(0.25, 1.25, 50)):
            # Generate the dataset to use
            generate_dataset([0.5, beta, 0.5, delta])

            # Initialize model
            model = model_type(args, top, exptop)

            # Get the best ckpt in training and load it into the model
            ckpt_path = args.ckpt_path + "/checkpoints/" + args.checkpt if args.checkpt != "None" \
                else f"{args.ckpt_path}/checkpoints/{find_best_epoch(args.ckpt_path)[0]}"

            model.load_state_dict(torch.load(ckpt_path)['state_dict'], strict=False)

            # Load in the dataset, generate context and inputs
            dataset = np.load(f"data/temp_dataset_{seed}.npz", allow_pickle=True)
            signals = dataset['signals']
            labels = dataset['labels']

            # Split into context and inputs
            inputs = torch.from_numpy(signals[:32]).to(torch.float32)
            context = torch.from_numpy(signals[32:]).to(torch.float32).repeat(minibatch_size - 3, 1, 1, 1)

            with torch.no_grad():
                outs = model(inputs, context, labels, 20)

            plt.plot(inputs[0, :, 1], c=cm.coolwarm(norm(beta)), linestyle='solid', zorder=6)
            plt.plot(outs[0, :, 1], c='k', linestyle='dashed', zorder=4)

        plt.savefig(f"{args.ckpt_path}/betareconstructions_delta{delta}.png")
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

    # Get local codes grid over the testing environments
    # plot_parameter_distributions()

    # Plot reconstructions across the three modes of training
    # plot_training_slice_reconstructions()