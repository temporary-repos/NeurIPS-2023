"""
@file metrics.py

Holds a variety of metric computing functions for time-series forecasting models
"""
import numpy as np


def vpt(gt, preds, epsilon=0.020, **kwargs):
    """
    Computes the Valid Prediction Time metric, as proposed in https://openreview.net/pdf?id=qBl8hnwR0px
    VPT = argmin_t [MSE(gt, pred) > epsilon]
    :param gt: ground truth sequences
    :param preds: model predicted sequences
    :param epsilon: threshold for valid prediction
    """
    # Ensure on CPU and numpy
    if not isinstance(gt, np.ndarray):
        gt = gt.cpu().numpy()
        preds = preds.cpu().numpy()

    # Get dimensions
    _, timesteps, dim = gt.shape

    # Get pixel_level MSE at each timestep
    mse = (gt - preds) ** 2
    mse = np.sum(mse, axis=2) / dim

    # Get VPT
    vpts = []
    for m in mse:
        # Get all indices below the given epsilon
        indices = np.where(m < epsilon)[0] + 1

        # If there are none below, then add 0
        if len(indices) == 0:
            vpts.append(0)
            continue

        # Append last in list
        vpts.append(indices[-1])

    # Return VPT mean over the total timesteps
    return np.mean(vpts) / timesteps, np.std(vpts) / timesteps


def reconstruction_mse(output, target, **kwargs):
    """ Gets the mean of the per-pixel MSE for the given length of timesteps used for training """
    full_pixel_mses = (output[:, :kwargs['args'].generation_training_len] - target[:, :kwargs['args'].generation_training_len]) ** 2
    sequence_pixel_mse = np.mean(full_pixel_mses, axis=(1, 2))
    return np.mean(sequence_pixel_mse), np.std(sequence_pixel_mse)


def nrmse(y, y_hat, **kwargs):
    idxs = [(len(i), ~np.isnan(i)) for i in y]
    RMSE = [np.power(y[i][idxs[i][1]] - y_hat[i][:idxs[i][0]][idxs[i][1]],2) for i in range(len(y))]
    power = [y[i][idxs[i][1]]**2 for i in range(len(y))]
    NRMSE = RMSE / np.sqrt(sum([i.sum() for i in power])/sum([len(i) for i in power]))*100
    return np.mean(NRMSE), np.std(NRMSE)
