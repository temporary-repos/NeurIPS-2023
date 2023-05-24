"""
@file plotting.py

Holds general plotting functions for reconstructions of the bouncing ball dataset
"""
import numpy as np
import matplotlib.pyplot as plt


def show_sequences(seqs, preds, out_loc, num_out=None):
    plt.figure(0)
    if not isinstance(seqs, np.ndarray):
        seqs = seqs.cpu().numpy()
        preds = preds.cpu().numpy()

    if num_out is not None:
        seqs = seqs[:num_out]
        preds = preds[:num_out]

    figure, axis = plt.subplots(num_out, 1)

    for i in range(num_out):
        axis[i].plot(seqs[i])
        axis[i].plot(preds[i], '--')

    # Save to out location
    plt.savefig(out_loc)
    plt.close()


def plot_parameter_distribution(params, labels, filename, param_label="weights"):
    # Get weight histogram
    for i in np.unique(labels):
        subset = np.mean(params[np.where(labels == i)[0], :], axis=0)
        plt.hist(subset, bins=100, alpha=0.5, label=f"{i}")

    plt.legend()
    plt.title(f"Distribution of Generated {param_label.title()} from Codes")
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
