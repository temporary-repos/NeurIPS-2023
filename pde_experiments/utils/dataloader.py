"""
@file dataloader.py

Holds the WebDataset classes for the available datasets
"""
import torch
import numpy as np

from functools import partial
from torch.utils.data import Dataset, DataLoader
from scipy.integrate import solve_ivp


##################
# Lotka-Volterra #
##################

class ODEDataset(Dataset):
    def __init__(self, n_data_per_env, t_horizon, params, dt, random_influence=0.2, method='RK45', group='train',
                 rdn_gen=1.):
        super().__init__()
        self.n_data_per_env = n_data_per_env
        self.num_env = len(params)
        self.len = n_data_per_env * self.num_env
        self.t_horizon = float(t_horizon)
        self.dt = dt
        self.random_influence = random_influence
        self.params_eq = params
        self.test = (group == 'test')
        self.max = np.iinfo(np.int32).max
        self.buffer = dict()
        self.indices = [list(range(env * n_data_per_env, (env + 1) * n_data_per_env)) for env in range(self.num_env)]
        self.method = method
        self.rdn_gen = rdn_gen

    def _f(self, t, x, env=0):
        raise NotImplemented

    def _get_init_cond(self, index):
        raise NotImplemented

    def __getitem__(self, index):
        env = index // self.n_data_per_env
        env_index = index % self.n_data_per_env
        t = torch.arange(0, self.t_horizon, self.dt).float()
        out = {'t': t, 'env': env}
        if self.buffer.get(index) is None:
            y0 = self._get_init_cond(env_index)
            y = solve_ivp(partial(self._f, env=env), (0., self.t_horizon), y0=y0, method=self.method,
                          t_eval=np.arange(0., self.t_horizon, self.dt))
            y = torch.from_numpy(y.y).float()
            out['state'] = y
            self.buffer[index] = y.numpy()
        else:
            out['state'] = torch.from_numpy(self.buffer[index])

        out['index'] = index
        out['param'] = torch.tensor(list(self.params_eq[env].values()))
        return out

    def __len__(self):
        return self.len


class LotkaVolterraGenerator(ODEDataset):
    def _f(self, t, x, env=0):
        alpha = self.params_eq[env]['alpha']
        beta = self.params_eq[env]['beta']
        gamma = self.params_eq[env]['gamma']
        delta = self.params_eq[env]['delta']
        d = np.zeros(2)
        d[0] = alpha * x[0] - beta * x[0] * x[1]
        d[1] = delta * x[0] * x[1] - gamma * x[1]
        return d

    def _get_init_cond(self, index):
        np.random.seed(index if not self.test else self.max - index)
        return np.random.random(2) + self.rdn_gen


class LVDataset(Dataset):
    """
    Load sequences of signals
    """

    def __init__(self, file_path, config):
        self.k_shot = config['k_shot']
        self.shuffle = config['shuffle']

        # Load data
        npzfile = np.load(file_path)
        self.signals = npzfile['signals'].astype(np.float32)
        print(f"=> Signals: {self.signals.shape}")

        # Get labels of environments
        self.labels = npzfile['labels'].astype(np.int16)
        self.label_idx = {}
        for label in np.unique(self.labels):
            idx = np.where(self.labels == label)[0]
            self.label_idx[label] = idx

        # Load environment parameters
        self.params = npzfile['params']

        # Get data dimensions
        self.sequences, self.timesteps, self.dim = self.signals.shape
        self.split()

    def __len__(self):
        return self.qry_idx.shape[0]

    def __getitem__(self, idx):
        """ Get signal and support set """
        label_qry = self.labels[self.qry_idx[idx]]
        param_qry = self.params[self.qry_idx[idx]]
        signal_qry = self.signals[self.qry_idx[idx], :]
        signal_spt = self.signals[self.spt_idx[label_qry], :]

        return torch.Tensor([idx]), signal_qry, signal_spt, label_qry, param_qry

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


if __name__ == '__main__':
    dataset = LVDataset('../data/lv_train.npz', {'k_shot': 3, 'shuffle': True, 'dataset_percent': 1.0})
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for batch in dataloader:
        indices, signals, domains, labels, params = batch
        print()