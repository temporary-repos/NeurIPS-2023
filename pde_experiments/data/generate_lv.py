"""
@file generate_lv.py

Generates a dataset file for the Lotka-Volterra dataset, saved in .npz format for loading in with
the already generated dataloaders.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from pde_experiments.utils.dataloader import LotkaVolterraGenerator

np.random.seed(111)

# Needed dataset parameters
for minibatch_size in [8]:  # should be 4
    for indomain_envs in [0, 8]:
        for outdomain_envs in [0, 8, 16, 32]:
            # Define environment parameters
            factor = 1.0
            state_c = 2
            init_gain = 0.15
            method = "rk4"

            """ Building Training Dataset """
            dataset_train_params = {
                "n_data_per_env": minibatch_size, "t_horizon": 10, "dt": 0.5, "method": "RK45", "group": "train",
                "params": [
                    # Classic training environments
                    {"alpha": 0.5, "beta": 0.5, "gamma": 0.5, "delta": 0.5},
                    {"alpha": 0.5, "beta": 0.75, "gamma": 0.5, "delta": 0.5},
                    {"alpha": 0.5, "beta": 1.0, "gamma": 0.5, "delta": 0.5},
                    {"alpha": 0.5, "beta": 0.5, "gamma": 0.5, "delta": 0.75},
                    {"alpha": 0.5, "beta": 0.5, "gamma": 0.5, "delta": 1.0},
                    {"alpha": 0.5, "beta": 0.75, "gamma": 0.5, "delta": 0.75},
                    {"alpha": 0.5, "beta": 0.75, "gamma": 0.5, "delta": 1.0},
                    {"alpha": 0.5, "beta": 1.0, "gamma": 0.5, "delta": 0.75},
                    {"alpha": 0.5, "beta": 1.0, "gamma": 0.5, "delta": 1.0},
                ]
            }

            # Add indomain environments to training
            indomain_env_params = [
                # {"alpha": 0.5, "beta": 0.75, "gamma": 0.5, "delta": 0.75},
                # {"alpha": 0.5, "beta": 0.5, "gamma": 0.5, "delta": 0.5},
                # {"alpha": 0.5, "beta": 0.75, "gamma": 0.5, "delta": 0.5},
                # {"alpha": 0.5, "beta": 1.0, "gamma": 0.5, "delta": 0.5},
                # {"alpha": 0.5, "beta": 0.5, "gamma": 0.5, "delta": 0.75},
                # {"alpha": 0.5, "beta": 0.5, "gamma": 0.5, "delta": 1.0},
                # {"alpha": 0.5, "beta": 0.75, "gamma": 0.5, "delta": 1.0},
                # {"alpha": 0.5, "beta": 1.0, "gamma": 0.5, "delta": 0.75},
                # {"alpha": 0.5, "beta": 1.0, "gamma": 0.5, "delta": 1.0},

                {"alpha": 0.5, "beta": 0.625, "gamma": 0.5, "delta": 0.625},
                {"alpha": 0.5, "beta": 0.875, "gamma": 0.5, "delta": 0.875},
                {"alpha": 0.5, "beta": 0.875, "gamma": 0.5, "delta": 0.625},
                {"alpha": 0.5, "beta": 0.625, "gamma": 0.5, "delta": 0.875},
                {"alpha": 0.5, "beta": 0.625, "gamma": 0.5, "delta": 0.75},
                {"alpha": 0.5, "beta": 0.875, "gamma": 0.5, "delta": 0.75},
                {"alpha": 0.5, "beta": 0.75, "gamma": 0.5, "delta": 0.625},
                {"alpha": 0.5, "beta": 0.75, "gamma": 0.5, "delta": 0.875},
            ]
            if indomain_envs > 0:
                dataset_train_params['params'].extend(indomain_env_params[:indomain_envs])

            # Add out-of-domain environments to training
            test_indomains = 12
            test_outdomains = 56
            outdomain_env_params = [
                # Out boundary
                {"alpha": 0.5, "beta": 0.25, "gamma": 0.5, "delta": 1.25},
                {"alpha": 0.5, "beta": 0.25, "gamma": 0.5, "delta": 0.25},
                {"alpha": 0.5, "beta": 1.25, "gamma": 0.5, "delta": 0.25},
                {"alpha": 0.5, "beta": 1.25, "gamma": 0.5, "delta": 1.25},
                {"alpha": 0.5, "beta": 0.75, "gamma": 0.5, "delta": 0.25},
                {"alpha": 0.5, "beta": 0.75, "gamma": 0.5, "delta": 1.25},
                {"alpha": 0.5, "beta": 0.25, "gamma": 0.5, "delta": 0.75},
                {"alpha": 0.5, "beta": 1.25, "gamma": 0.5, "delta": 0.75},

                # Inner loop
                {"alpha": 0.5, "beta": 0.35, "gamma": 0.5, "delta": 0.35},
                {"alpha": 0.5, "beta": 0.35, "gamma": 0.5, "delta": 1.15},
                {"alpha": 0.5, "beta": 1.15, "gamma": 0.5, "delta": 0.35},
                {"alpha": 0.5, "beta": 1.15, "gamma": 0.5, "delta": 1.15},
                {"alpha": 0.5, "beta": 0.35, "gamma": 0.5, "delta": 0.75},
                {"alpha": 0.5, "beta": 0.75, "gamma": 0.5, "delta": 0.35},
                {"alpha": 0.5, "beta": 0.75, "gamma": 0.5, "delta": 1.15},
                {"alpha": 0.5, "beta": 1.15, "gamma": 0.5, "delta": 0.75},

                # Denser out boundary
                {"alpha": 0.5, "beta": 0.25, "gamma": 0.5, "delta": 1.0},
                {"alpha": 0.5, "beta": 0.25, "gamma": 0.5, "delta": 0.5},
                {"alpha": 0.5, "beta": 0.5, "gamma": 0.5, "delta": 1.0},
                {"alpha": 0.5, "beta": 0.5, "gamma": 0.5, "delta": 0.5},
                {"alpha": 0.5, "beta": 1.0, "gamma": 0.5, "delta": 1.0},
                {"alpha": 0.5, "beta": 1.0, "gamma": 0.5, "delta": 0.5},
                {"alpha": 0.5, "beta": 1.25, "gamma": 0.5, "delta": 1.0},
                {"alpha": 0.5, "beta": 1.25, "gamma": 0.5, "delta": 0.5},

                # Denser out domains
                {"alpha": 0.5, "beta": 0.35, "gamma": 0.5, "delta": 1.0},
                {"alpha": 0.5, "beta": 0.35, "gamma": 0.5, "delta": 0.5},
                {"alpha": 0.5, "beta": 0.55, "gamma": 0.5, "delta": 1.15},
                {"alpha": 0.5, "beta": 0.55, "gamma": 0.5, "delta": 0.35},
                {"alpha": 0.5, "beta": 0.95, "gamma": 0.5, "delta": 1.15},
                {"alpha": 0.5, "beta": 0.95, "gamma": 0.5, "delta": 0.35},
                {"alpha": 0.5, "beta": 1.15, "gamma": 0.5, "delta": 1.0},
                {"alpha": 0.5, "beta": 1.15, "gamma": 0.5, "delta": 0.5},

            ]
            if outdomain_envs > 0:
                dataset_train_params['params'].extend(outdomain_env_params[:outdomain_envs])

            """ Building Testing Dataset"""
            dataset_test_params = {
                "n_data_per_env": 32, "t_horizon": 10, "dt": 0.5, "method": "RK45", "group": "test",
                "params": []
            }

            # Sample in and out testing environments
            testing_envs = np.random.uniform(0.25, 1.25, [128, 2])
            testing_indomain_envs = testing_envs[
                np.where(np.logical_and(0.5 <= testing_envs[:, 0], testing_envs[:, 0] <= 1) & np.logical_and(
                    0.5 <= testing_envs[:, 1], testing_envs[:, 1] <= 1))[0]
            ]
            testing_outdomain_envs = testing_envs[
                np.where(~np.logical_and(0.5 <= testing_envs[:, 0], testing_envs[:, 0] <= 1) | ~np.logical_and(
                    0.5 <= testing_envs[:, 1], testing_envs[:, 1] <= 1))[0]
            ]

            testing_indomain_envs = testing_indomain_envs[:test_indomains]
            testing_outdomain_envs = testing_outdomain_envs[:test_outdomains]

            # Add these to the generator
            for env in testing_indomain_envs:
                dataset_test_params['params'].append(
                    {"alpha": 0.5, "beta": env[0], "gamma": 0.5, "delta": env[1]}
                )

            for env in testing_outdomain_envs:
                dataset_test_params['params'].append(
                    {"alpha": 0.5, "beta": env[0], "gamma": 0.5, "delta": env[1]}
                )

            # Show the distribution of train, test indomain, test outdomain
            plot = True
            if plot is True:
                plt.figure()
                for params in dataset_train_params['params']:
                    plt.scatter(params['beta'], params['delta'], c='k')

                plt.scatter(testing_indomain_envs[:, 0], testing_indomain_envs[:, 1], marker='x')
                plt.scatter(testing_outdomain_envs[:, 0], testing_outdomain_envs[:, 1], marker='^')
                plt.grid()
                plt.show()

            """ Build and generate datasets"""
            dataset_train = LotkaVolterraGenerator(**dataset_train_params)
            dataset_test = LotkaVolterraGenerator(**dataset_test_params)

            # Generate datasets
            dataset_version = f"_{minibatch_size}perenv_{indomain_envs}indomainenvs_{outdomain_envs}outdomainenvs"
            set_type = [('train', dataset_train), ('test', dataset_test)]

            if not os.path.exists(f"lotka_volterra{dataset_version}/"):
                os.mkdir(f"lotka_volterra{dataset_version}/")

            for data_type, dataset in set_type:
                signals, labels, params = [], [], []
                for idx in range(dataset.len):
                    data = dataset.__getitem__(idx)

                    signals.append(data['state'])
                    labels.append(data['env'])
                    params.append(data['param'])

                signals = np.stack(signals)
                signals = np.swapaxes(signals, 1, 2)

                labels = np.stack(labels)
                params = np.stack(params)
                print(f"Signals {signals.shape} Labels {labels.shape} Params {params.shape}")
                print(f"Labels Unique {np.unique(labels, return_counts=True)}")

                np.savez(f"lotka_volterra{dataset_version}/lotka_volterra{dataset_version}_{data_type}.npz",
                         signals=signals, labels=labels, params=params)

                plt.plot(signals[2, :, 0])
                plt.plot(signals[2, :, 1])
                plt.show()

                print(signals)
