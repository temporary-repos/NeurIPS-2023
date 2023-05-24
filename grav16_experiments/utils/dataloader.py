"""
@file dataloader.py

Holds the WebDataset classes for the available datasets
"""
import os
import h5py
import torch
import numpy as np

from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset


""" 
Mixed Physics Dataloaders for non-meta models
"""
class PymunkData(Dataset):
    """
    Load sequences of images
    """

    def __init__(self, file_path, config):
        self.shuffle = config['shuffle']

        # Load data
        npzfile = np.load(file_path)
        self.images = npzfile['images'].astype(np.float32)
        self.images = (self.images > 0).astype('float32')

        self.labels = npzfile['label'].astype(np.int16)

        # Load ground truth position and velocity (if present). This is not used in the KVAE experiments in the paper.
        if 'state' in npzfile:
            # Only load the position, not velocity
            self.state = npzfile['state'].astype(np.float32)[:, :, :2]

            # Set state dimension
            self.state_dim = self.state.shape[-1]

        # Modify based on dataset percent
        rand_idx = np.random.choice(range(self.images.shape[0]), size=int(self.images.shape[0] * config['dataset_percent']), replace=False)
        self.images = self.images[rand_idx]
        self.labels = self.labels[rand_idx]
        self.state = self.state[rand_idx]

        print(f"Images: {self.images.shape}")

        # Get data dimensions
        self.sequences, self.timesteps = self.images.shape[0], self.images.shape[1]

        # We set controls to zero (we don't use them even if dim_u=1). If they are fixed to one instead (and dim_u=1)
        # the B matrix represents a bias.
        self.controls = np.zeros((self.sequences, self.timesteps, config['dim_u']), dtype=np.float32)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        """ Couple images and controls together for compatibility with other datasets """
        label = torch.Tensor([self.labels[idx]])
        image = torch.from_numpy(self.images[idx, :])
        state = torch.from_numpy(self.state[idx, :])
        control = torch.from_numpy(self.controls[idx, :])
        return torch.Tensor([idx]), image, state, control, label


def bouncingball_collate(batch):
    """
    Collate function for the bouncing ball experiments
    Args:
        batch: given batch at this generator call
    Returns: indices of batch, images, controls
    """
    images, states, labels = [], [], []

    for b in batch:
        _, image, state, _, label = b
        images.append(image)
        states.append(state)
        labels.append(label)

    images = torch.stack(images)
    states = torch.stack(states)
    labels = torch.stack(labels)
    return images, None, states, None, labels


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=bouncingball_collate):
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)


class EpochDataLoader(DataLoader):
    """
    Dataloader for the base implementation of bouncing ball w/ gravity experiments, available here:
    https://github.com/simonkamronn/kvae
    """
    def __init__(self, args, split='train', shuffle=True):

        # Choose which normalization to use
        if args.dataset == 'meta_learning':
            out_distr = 'bernoulli'
        else:
            out_distr = 'none'

        # Generate dataset and initialize loader
        config = {
            'out_distr': out_distr,
            'dim_u': 1,
            'shuffle': shuffle,
            'dataset_percent': args.dataset_percent
        }

        self.init_kwargs = {
            'batch_size': args.batch_size,
            'shuffle': shuffle,
            'num_workers': args.num_workers,
            'collate_fn': bouncingball_collate,
            'drop_last': True
        }

        self.dataset = PymunkData(os.path.abspath('').replace('\\', '/') + f"/data/{args.dataset}/{args.dataset_ver}_{split}.npz", config)
        self.split = split

        super().__init__(self.dataset, **self.init_kwargs)


""" 
Meta-Learning Components 
"""
class PymunkEpisoticData(Dataset):
    """
    Load sequences of images
    """

    def __init__(self, file_path, config):
        self.k_shot = config['k_shot']
        self.shuffle = config['shuffle']

        # Load data
        npzfile = np.load(file_path)
        self.images = npzfile['images'].astype(np.float32)

        if self.images.shape[-1] == 64:
            self.images = self.images[:, ::3]
        print(f"=> Images: {self.images.shape}")

        # Modify based on dataset percent
        rand_idx = np.random.choice(range(self.images.shape[0]), size=int(self.images.shape[0] * config['dataset_percent']), replace=False)
        self.images = self.images[rand_idx]
        if config['out_distr'] == 'bernoulli':
            self.images = (self.images > 0).astype('float32')
        elif config['out_distr'] == 'norm':
            self.images = self.normalize(self.images)
        elif config['out_distr'] == 'none':
            pass

        self.labels = npzfile['label'].astype(np.int16)
        self.labels = self.labels[rand_idx]
        self.label_idx = {}
        for label in np.unique(self.labels):
            idx = np.where(self.labels == label)[0]
            self.label_idx[label] = idx

        # Load ground truth position and velocity (if present). This is not used in the KVAE experiments in the paper.
        if 'state' in npzfile:
            # Only load the position, not velocity
            self.state = npzfile['state'].astype(np.float32)
            self.state = self.state[rand_idx]

            # Set state dimension
            self.state_dim = self.state.shape[-1]

        # Get data dimensions
        self.sequences, self.timesteps = self.images.shape[0], self.images.shape[1]

        # We set controls to zero (we don't use them even if dim_u=1). If they are fixed to one instead (and dim_u=1)
        self.controls = np.zeros((self.sequences, self.timesteps, config['dim_u']), dtype=np.float32)
        self.split()

    def __len__(self):
        return self.qry_idx.shape[0]

    def __getitem__(self, idx):
        """ Couple images and controls together for compatibility with other datasets """
        label_qry = self.labels[self.qry_idx[idx]]
        image_qry = self.images[self.qry_idx[idx], :]
        state_qry = self.state[self.qry_idx[idx], :]
        control_qry = self.controls[self.qry_idx[idx], :]
        image_spt = self.images[self.spt_idx[label_qry], :]
        state_spt = self.state[self.spt_idx[label_qry], :]
        control_spt = self.controls[self.spt_idx[label_qry], :]

        image_qry = torch.from_numpy(image_qry)
        state_qry = torch.from_numpy(state_qry)
        control_qry = torch.from_numpy(control_qry)
        label_qry = torch.Tensor([label_qry])
        image_spt = torch.from_numpy(image_spt)
        state_spt = torch.from_numpy(state_spt)
        control_spt = torch.from_numpy(control_spt)

        return torch.Tensor([idx]), image_qry, state_qry, control_qry, label_qry, \
               image_spt, state_spt, control_spt

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

            self.qry_idx.extend(samples.tolist())

        self.qry_idx = np.array(self.qry_idx)
        self.qry_idx = np.sort(self.qry_idx)

    def normalize(self, images):
        norm_images = images / 100
        return norm_images


class BaseMetaDataLoader(DataLoader):
    """ Base class for all data loaders """
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn):
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers,
            'drop_last': True
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)


def bouncingball_episotic_collate(batch):
    """
    Collate function for the bouncing ball experiments
    Args:
        batch: given batch at this generator call
    Returns: indices of batch, images, controls
    """
    images, states, labels, \
        images_D, states_D \
         = [], [], [], [], []

    for b in batch:
        _, image, state, _, label, image_D, state_D, _ = b
        images.append(image)
        states.append(state)
        labels.append(label)
        images_D.append(image_D)
        states_D.append(state_D)

    images = torch.stack(images)
    states = torch.stack(states)
    labels = torch.stack(labels)
    images_D = torch.stack(images_D)
    states_D = torch.stack(states_D)

    return images, images_D, states, states_D, labels


class EpisoticDataLoader(BaseMetaDataLoader):
    """
    Dataloader for the base implementation of bouncing ball w/ gravity experiments, available here:
    https://github.com/simonkamronn/kvae
    """
    def __init__(self, args, split='train', shuffle=True):

        # Choose which normalization to use
        if args.dataset == 'meta_learning':
            out_distr = 'bernoulli'
        else:
            out_distr = 'none'

        # Generate dataset and initialize loader
        config = {
            'out_distr': out_distr,
            'dim_u': 1,
            'k_shot': args.domain_size,
            'shuffle': shuffle,
            'dataset_percent': args.dataset_percent
        }

        self.init_kwargs = {
            'batch_size': args.batch_size,
            'shuffle': shuffle,
            'validation_split': 0.0,
            'num_workers': args.num_workers,
            'collate_fn': bouncingball_episotic_collate
        }

        self.dataset = PymunkEpisoticData(os.path.abspath('').replace('\\', '/') + f"/data/{args.dataset}/{args.dataset_ver}_{split}.npz", config)
        self.split = split

        super().__init__(self.dataset, **self.init_kwargs)

    def next(self):
        self.dataset.split()
        self.init_kwargs['validation_split'] = 0.0
        return BaseMetaDataLoader(dataset=self.dataset,
                                  batch_size=self.init_kwargs['batch_size'],
                                  shuffle=self.init_kwargs['shuffle'],
                                  validation_split=self.init_kwargs['validation_split'],
                                  num_workers=self.init_kwargs['num_workers'],
                                  collate_fn=self.init_kwargs['collate_fn'])


""" 
PDE Experiment Datasets
"""
class FNODatasetMult(Dataset):
    def __init__(self, filename, initial_step=10, saved_folder='data/pdes/', if_test=False, test_ratio=0.1):
        """
        :param filename: filename that contains the dataset
        :type filename: STR
        :param filenum: array containing indices of filename included in the dataset
        :type filenum: ARRAY
        :param initial_step: time steps taken as initial condition, defaults to 10
        :type initial_step: INT, optional
        """
        # Define path to files
        self.file_path = os.path.abspath(saved_folder + filename + ".h5")

        # Extract list of seeds
        with h5py.File(self.file_path, 'r') as h5_file:
            data_list = sorted(h5_file.keys())

        test_idx = int(len(data_list) * (1 - test_ratio))
        if if_test:
            self.data_list = np.array(data_list[test_idx:])
        else:
            self.data_list = np.array(data_list[:test_idx])

        # Time steps used as initial conditions
        self.initial_step = initial_step

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # Open file and read data
        with h5py.File(self.file_path, 'r') as h5_file:
            seed_group = h5_file[self.data_list[idx]]

            # data dim = [t, x1, ..., xd, v]
            data = np.array(seed_group["data"], dtype='f')
            data = torch.tensor(data, dtype=torch.float)

            # convert to [x1, ..., xd, t, v]
            permute_idx = list(range(1, len(data.shape) - 1))
            permute_idx.extend(list([0, -1]))
            data = data.permute(permute_idx)

            # Extract spatial dimension of data
            dim = len(data.shape) - 2

            # x, y and z are 1-D arrays
            # Convert the spatial coordinates to meshgrid
            if dim == 1:
                grid = np.array(seed_group["grid"]["x"], dtype='f')
                grid = torch.tensor(grid, dtype=torch.float).unsqueeze(-1)
            elif dim == 2:
                x = np.array(seed_group["grid"]["x"], dtype='f')
                y = np.array(seed_group["grid"]["y"], dtype='f')
                x = torch.tensor(x, dtype=torch.float)
                y = torch.tensor(y, dtype=torch.float)
                X, Y = torch.meshgrid(x, y)
                grid = torch.stack((X, Y), axis=-1)

        return data, grid, None
