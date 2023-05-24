"""
@file tsne_nfssm_visualization.py

Handles visualizing the TSNE plot of all embeddings of grav16 directions
for the local+global NFSSM model
"""
import os
import json
import itertools
import numpy as np
import matplotlib.pyplot as plt

from openTSNE import TSNE
from scipy import interpolate
from scipy.spatial import ConvexHull

interpolation = True
path = "../experiments_completed/Grav16/grav16_metanode/metanode/version_2/"

# Load in and stack all codes/labels
codes, labels = [], []
for query in os.listdir(path):
    print(query)
    if "test" not in query or ".txt" in query or "train" in query:
        continue

    code = np.load(f"{path}/{query}/{query}_codevectors.npy", allow_pickle=True)
    label = np.load(f"{path}/{query}/{query}_labels.npy", allow_pickle=True)

    codes.append(code)
    labels.append(label)


codes = np.vstack(codes)
labels = np.vstack(labels)
print(codes.shape, labels.shape, np.unique(labels, return_counts=True))

# Get TSNE embedding over code vectors
tsne = TSNE(n_components=2, perplexity=30, initialization="pca", metric="cosine", n_jobs=8, random_state=3)
tsne_embedding = tsne.fit(codes)
print(tsne_embedding.shape)

# Plot codes in TSNE
marker = itertools.cycle(('o', 'v', '^', '<', '>', 's', '8', 'p'))
for i in np.unique(labels):
    subset = tsne_embedding[np.where(labels == i)[0], :]
    color = next(plt.gca()._get_lines.prop_cycler)['color']

    # Get convex hull
    hull = ConvexHull(subset)
    x_hull = np.append(subset[hull.vertices, 0], subset[hull.vertices, 0][0])
    y_hull = np.append(subset[hull.vertices, 1], subset[hull.vertices, 1][0])

    # Interpolate
    dist = np.sqrt((x_hull[:-1] - x_hull[1:]) ** 2 + (y_hull[:-1] - y_hull[1:]) ** 2)
    dist_along = np.concatenate(([0], dist.cumsum()))
    spline, u = interpolate.splprep([x_hull, y_hull], u=dist_along, s=0, per=1)
    interp_d = np.linspace(dist_along[0], dist_along[-1], 50)
    interp_x, interp_y = interpolate.splev(interp_d, spline)

    # Plot points in cluster
    if interpolation is True:
        test_envs = [1, 4, 8, 9, 10, 12]
    else:
        test_envs = [8, 9, 10, 11, 12, 13, 14, 15]
    plot_label = f"{i}*" if i in test_envs else f"{i}"

    if i not in test_envs:
        plt.scatter(subset[:, 0], subset[:, 1], alpha=0.5, c='k', marker=next(marker),label=plot_label)

        # Plot boundaries
        plt.fill(interp_x, interp_y, '--', alpha=0.2, c='k')
    else:
        plt.scatter(subset[:, 0], subset[:, 1], alpha=0.5, c=color, marker=next(marker), label=plot_label)

        # Plot boundaries
        plt.fill(interp_x, interp_y, '--', alpha=0.2, c=color)

plt.title("t-SNE Plot of Generated Codes")
plt.legend(loc='upper right')

if interpolation is True:
    plt.savefig(f"Grav16CodeTSNE_MetaNODE_Interpolation.png", bbox_inches='tight')
else:
    plt.savefig(f"Grav16CodeTSNE_MetaNODE_Extrapolation.png", bbox_inches='tight')
plt.close()
