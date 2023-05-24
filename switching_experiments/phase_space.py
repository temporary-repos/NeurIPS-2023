import torch
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import stats

from tqdm import tqdm
from torch.utils.data import DataLoader
from switching_experiments.utils.dataloader import SwitchingDataset
from switching_experiments.utils.utils import parse_args, get_exp_versions, find_best_epoch

# Define environment parameters
config = "Nascar_MetaNeuralODE"


color_names = ["windows blue",
               "red",
               "amber",
               "faded green",
               "dusty purple",
               "orange",
               "clay",
               "pink",
               "greyish",
               "mint",
               "cyan",
               "steel blue",
               "forest green",
               "pastel purple",
               "salmon",
               "dark brown"]
colors = sns.xkcd_palette(color_names)


if __name__ == '__main__':
    # Define the parser with the configuration file path
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default=f'configs/{config}.json',
                        help='path and name of the configuration .json to load')

    # Parse the config file and get the model function used
    args, model_type = parse_args(parser)
    args.batch_size = 1

    # Get version numbers
    global top, exptop
    top, exptop = get_exp_versions(args.model, args.exptype)

    # Initialize model
    model = model_type(args, top, exptop)

    # Get the best ckpt in training and load it into the model
    ckpt_path = args.ckpt_path + "/checkpoints/" + args.checkpt if args.checkpt != "None" \
        else f"{args.ckpt_path}/checkpoints/{find_best_epoch(args.ckpt_path)[0]}"

    model.load_state_dict(torch.load(ckpt_path)['state_dict'], strict=False)

    mse = torch.nn.MSELoss()

    # Get dataset
    dataset = SwitchingDataset(f'data/{args.dataset}/{args.dataset_ver}_train.npz')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Get predictions and gradients from model
    preds, signals, domains = [], [], []
    gradients, embeddings, labels, mses = [], [], [], []
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(dataloader)):
            # Stack batch and restrict to generation length
            indices, signal, domain, label = batch

            # Get predictions and embeddings
            pred, embeds = model(signal, domain, label, 7)
            gradients.append(model.dynamics_func.vector_gradients[0])
            embeddings.append(embeds[0, 0])
            labels.append(stats.mode(label[0])[0])
            mses.append(mse(pred, signal).mean())

            preds.append(pred)
            signals.append(signal)
            domains.append(domain)

    if args.latent_dim == 2:
        # -------------------------------------------------------
        # Stack gradients and shape
        gradients = torch.vstack(gradients).detach().cpu().numpy()
        gradients_x, gradients_y = gradients[:, 0], gradients[:, 1]

        # Stack z0
        embeddings = torch.stack(embeddings).detach().cpu().numpy()
        embeddings_x, embeddings_y = embeddings[:, 0], embeddings[:, 1]

        # Stack labels and mses
        labels = np.vstack(labels)
        mses = np.vstack(mses)

        preds = torch.vstack(preds).detach().cpu().numpy()
        signals = torch.vstack(signals).detach().cpu().numpy()
        domains = torch.vstack(domains).detach().cpu().numpy()

        # -------------------------------------------------------
        # Plot the example with the worst MSE
        worst_idx = np.argmax(mses)

        plt.figure()
        plt.plot(domains[worst_idx, 0, :, 0], range(7), marker='x')
        plt.plot(domains[worst_idx, 1, :, 0], range(7, 14), marker='x')
        plt.plot(domains[worst_idx, 2, :, 0], range(14, 21), marker='x')
        plt.plot(signals[worst_idx, :, 0], range(21, 28), marker='x')
        plt.show()

        plt.figure()
        plt.plot(preds[worst_idx], '--')
        plt.plot(signals[worst_idx])
        plt.show()

        # -------------------------------------------------------
        # Drow direction fields, using matplotlib 's quiver function
        fig, ax = plt.subplots()
        plt.title(f'Trajectories and direction field')
        plt.grid()

        # Plot the GT and Model phase spaces
        plt.quiver(embeddings_x, embeddings_y, gradients_x, gradients_y, color=plt.cm.jet(plt.Normalize()(labels)))
        plt.savefig(f"{args.ckpt_path}/latentvectorfield.png")
        plt.close()

        # -------------------------------------------------------
        # Drow direction fields, using matplotlib 's quiver function
        plt.figure()
        fig, ax = plt.subplots()
        plt.title(f'Vector field with MSE as the color')
        plt.grid()

        # Plot the GT and Model phase spaces
        plt.scatter(embeddings_x, embeddings_y, c=mses, cmap='coolwarm')
        plt.colorbar()
        plt.show()
        plt.close()

    elif args.latent_dim == 3:
        # -------------------------------------------------------
        # Stack gradients and shape
        gradients = torch.vstack(gradients).detach().cpu().numpy()
        embeddings = torch.stack(embeddings).detach().cpu().numpy()
        labels = np.vstack(labels)
        mses = np.vstack(mses)

        gradients = np.reshape(gradients, [gradients.shape[0], 3])

        preds = torch.vstack(preds).detach().cpu().numpy()
        signals = torch.vstack(signals).detach().cpu().numpy()
        domains = torch.vstack(domains).detach().cpu().numpy()

        # -------------------------------------------------------
        # Drow direction fields, using matplotlib 's quiver function
        ax = plt.figure().add_subplot(projection='3d')
        plt.title(f'Trajectories and direction field')
        plt.grid()

        # Plot the GT and Model phase spaces
        plt.quiver(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2],
                   gradients[:, 0], gradients[:, 1], gradients[:, 2], color=plt.cm.jet(plt.Normalize()(labels)))
        plt.savefig(f"{args.ckpt_path}/latentvectorfield.png")
        plt.close()

    else:
        # -------------------------------------------------------
        # Stack gradients and shape
        gradients = torch.vstack(gradients).detach().cpu().numpy()
        embeddings = torch.stack(embeddings).detach().cpu().numpy()
        labels = np.vstack(labels)
        mses = np.vstack(mses)

        preds = torch.vstack(preds).detach().cpu().numpy()
        signals = torch.vstack(signals).detach().cpu().numpy()
        domains = torch.vstack(domains).detach().cpu().numpy()

        n_vis = 2

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111)
        q_s = dmfa.q_s[-n_vis:, max(1):].argmax(-1).reshape(-1)[:-1].detach().numpy()
        for j in range(2):
            xy = embeddings[-n_vis:, max(1):].reshape(-1, embeddings.shape[-1])[:-1, 2 * j:2 * j + 2]
            dxydt_m = xy_next[:, 1:, 2 * j:2 * j + 2] - xy
            for k in range(args.latent_dim):
                zk = q_s == k
                if j == 0:
                    ax.plot(xy[zk, 0][::2], xy[zk, 1][::2],
                            color=colors[k % len(colors)], alpha=1.0, label='State %d' % k, linewidth=1.5)
                else:
                    ax.quiver(xy[zk, 0][::1], xy[zk, 1][::1],
                              dxydt_m[k, zk, 0][::1], dxydt_m[k, zk, 1][::1],
                              color=colors[k % len(colors)], alpha=1.0, angles='xy', linewidths=0.1)
        ax.set_xlabel('$x$', fontsize=21)
        ax.set_ylabel('$y$', fontsize=21)
        ax.legend(fontsize=14, loc='lower right')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('Inferred Dynamics', fontsize=20, loc='center', y=0.94)
        ax.tick_params(axis='both', which='major', labelsize=10)
        plt.close('all')
