import matplotlib.pyplot as plt
import torch
import numpy as np

from scipy.integrate import odeint
from torch.utils.data import Dataset


class MetaPercentODEData(Dataset):
    def __init__(self, args, u_start, u_end, bounds=(0, 100), shuffle=True, is_interpolation=False, is_train=True):
        self.func_type = args.func_type

        self.shuffle = shuffle
        self.k_shot = args.k_shot
        self.ts = np.linspace(-1, 0, args.timesteps)
        self.xs = torch.linspace(1, 5, args.num_samples)
        self.us = torch.linspace(u_start, u_end, args.num_tasks)

        if is_interpolation is True:
            self.us = self.us + (0.2 * torch.rand_like(self.us))

        def f(x, t, u):
            return u * np.sin(x)

        data, labels, us = [], [], []
        for label, u0 in enumerate(zip(self.us.numpy())):
            for x0 in self.xs.numpy():
                data.append(odeint(f, y0=x0, t=self.ts, args=(u0,)))
                us.append(u0)
                labels.append(label)
        self.data = torch.from_numpy(np.stack(data)).float()
        self.labels = torch.from_numpy(np.stack(labels)).float()
        self.us = torch.from_numpy(np.stack(us)).float()

        self.data = self.data.reshape([args.num_tasks, args.num_samples, args.timesteps, 1])
        self.labels = self.labels.reshape([args.num_tasks, args.num_samples, 1])
        self.us = self.us.reshape([args.num_tasks, args.num_samples, 1])

        if is_train is True or is_interpolation is True:
            self.data = self.data[bounds[0]:bounds[1]]
            self.labels = self.labels[bounds[0]:bounds[1]]
            self.us = self.us[bounds[0]:bounds[1]]
        # If it is testing, get the extrapolation bounds
        elif is_train is False and is_interpolation is False:
            self.data = torch.concatenate((self.data[bounds[0] - args.extrapolation_bound_window:bounds[0]], self.data[bounds[1]:bounds[1] + args.extrapolation_bound_window]))
            self.labels = torch.concatenate((self.labels[bounds[0] - args.extrapolation_bound_window:bounds[0]], self.labels[bounds[1]:bounds[1] + args.extrapolation_bound_window]))
            self.us = torch.concatenate((self.us[bounds[0] - args.extrapolation_bound_window:bounds[0]], self.us[bounds[1]:bounds[1] + args.extrapolation_bound_window]))

        self.data = self.data.reshape([-1, args.timesteps, 1])
        self.labels = self.labels.reshape([-1, 1])
        self.us = self.us.reshape([-1, 1])

        print(f"=> Data {self.data.shape}")
        print(f"=> Labels {self.labels.shape}")
        print(f"=> Controls {self.us.shape}")

        # Get labels of environments
        self.label_idx = {}
        for label in np.unique(self.labels):
            idx = np.where(self.labels == label)[0]
            self.label_idx[label] = idx

        # Get data dimensions
        self.sequences, self.timesteps, self.dim = self.data.shape
        self.split()

    def __len__(self):
        return self.qry_idx.shape[0]

    def __getitem__(self, idx):
        """ Get signal and support set """
        label_qry = int(self.labels[self.qry_idx[idx]])
        control_qry = self.us[self.qry_idx[idx]]
        signal_qry = self.data[self.qry_idx[idx], :]
        signal_spt = self.data[self.spt_idx[label_qry], :]
        return torch.Tensor([idx]), signal_qry, signal_spt, label_qry, control_qry

    def split(self):
        self.spt_idx = {}
        self.qry_idx = []
        for label_id, samples in self.label_idx.items():
            sample_idx = np.arange(0, len(samples))
            if len(samples) < self.k_shot:
                self.spt_idx[label_id] = samples
            else:
                if self.shuffle:
                    np.random.shuffle(sample_idx)
                    spt_idx = np.sort(sample_idx[0:self.k_shot])
                else:
                    spt_idx = sample_idx[0:self.k_shot]
                self.spt_idx[label_id] = samples[spt_idx]

            # Build mask of support indices to remove from current query indices
            mask = np.full(len(samples), True, dtype=bool)
            mask[spt_idx] = False

            # Append remaining samples to query indices
            self.qry_idx.extend(samples[mask])

        self.qry_idx = np.array(self.qry_idx)
        self.qry_idx = np.sort(self.qry_idx)
