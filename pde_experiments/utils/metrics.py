"""
@file metrics.py

Holds a variety of metric computing functions for time-series forecasting models
"""
import torch


def recon_mse(output, target, **kwargs):
    """ Gets the mean of the per-pixel MSE for the given length of timesteps used for training """
    full_pixel_mses = (output[:, :kwargs['args'].generation_training_len] - target[:, :kwargs['args'].generation_training_len]) ** 2
    sequence_pixel_mse = torch.mean(full_pixel_mses, dim=(1, 2))
    return torch.mean(sequence_pixel_mse), torch.std(sequence_pixel_mse)


def rel_indomain(output, target, **kwargs):
    """ Handles getting the relative (MAPE) loss for the two given sequences """
    params = kwargs['params']

    output = output[
        torch.where(torch.logical_and(0.5 <= params[:, 1], params[:, 1] <= 1) & torch.logical_and(
            0.5 <= params[:, 3], params[:, 3] <= 1))[0]
    ]

    target = target[
        torch.where(torch.logical_and(0.5 <= params[:, 1], params[:, 1] <= 1) &
                 torch.logical_and(0.5 <= params[:, 3], params[:, 3] <= 1))[0]
    ]

    mape_top = torch.abs(output - target)
    mape_bot = torch.abs(target).nanmean()
    mape = mape_top / mape_bot
    return torch.mean(mape), torch.std(mape)


def rel_outdomain(output, target, **kwargs):
    """ Handles getting the relative (MAPE) loss for the two given sequences """
    params = kwargs['params']

    output = output[
                torch.where(~torch.logical_and(0.5 <= params[:, 1], params[:, 1] <= 1) |
                         ~torch.logical_and(0.5 <= params[:, 3], params[:, 3] <= 1))[0]
    ]

    target = target[
                torch.where(~torch.logical_and(0.5 <= params[:, 1], params[:, 1] <= 1) |
                         ~torch.logical_and(0.5 <= params[:, 3], params[:, 3] <= 1))[0]
    ]

    mape_top = torch.abs(output - target)
    mape_bot = torch.abs(target).nanmean()
    mape = mape_top / mape_bot
    return torch.mean(mape), torch.std(mape)
