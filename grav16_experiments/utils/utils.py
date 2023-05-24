"""
@file utils.py

Utility functions across files
"""
import os
import math
import json
import numpy as np
import torch.nn as nn
import pytorch_lightning

from torch.optim.lr_scheduler import _LRScheduler


def parse_args(parser):
    """
    Parse the cmd for a given configuration file and updates the arguments with its content
    As well, given the arguments, gets which model dynamics function is being used.
    :return: parsed arguments and model class
    """
    # Parse cmd line args
    parser = pytorch_lightning.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # Load in config file and update parser
    with open(parser.parse_args().config_path, 'rt') as f:
        args.__dict__.update(json.load(f))

    # Get the model type from args and add its specific arguments
    model_type = get_model(args.model)
    args.__dict__.update(model_type.get_model_specific_args())
    return args, model_type


def get_model(name):
    """ Import and return the specific latent dynamics function by the given name"""
    # Lowercase name in case of misspellings
    name = name.lower()

    #### Instance-Specific Models
    if name == "nfssm_is":
        from grav16_experiments.models.dynamics_functions.NeuralFuncSSM_InstanceSpecific import NFSSM
        return NFSSM

    #### Meta-Learning Models
    if name == "nfssm_local":
        from grav16_experiments.models.dynamics_functions.NeuralFuncSSM_Meta import NeuralFuncSSM
        return NeuralFuncSSM

    if name == "nfssm_global":
        from grav16_experiments.models.dynamics_functions.NeuralFuncSSM_Global import NeuralFuncSSM
        return NeuralFuncSSM

    if name == "nfssm_localglobal":
        from grav16_experiments.models.dynamics_functions.NeuralFuncSSM_LocalNGlobal import NeuralFuncSSM_Meta
        return NeuralFuncSSM_Meta

    if name == "leads":
        from grav16_experiments.models.dynamics_functions.LEADS import LEADS
        return LEADS

    if name == "coda":
        from grav16_experiments.models.dynamics_functions.CoDA import CoDA
        return CoDA

    if name == "metanode":
        from grav16_experiments.models.dynamics_functions.MetaNeuralODE import MetaNeuralODE
        return MetaNeuralODE

    if name == "metahyperssm":
        from grav16_experiments.models.dynamics_functions.MetaHyperNet_v2 import MetaHyperNet
        return MetaHyperNet

    if name == "hyperssm":
        from grav16_experiments.models.dynamics_functions.HyperSSM import HyperSSM
        return HyperSSM

    #### Global Baselines
    if name == "node":
        from grav16_experiments.models.dynamics_functions.NeuralODE import NeuralODE
        return NeuralODE

    if name == "rgnres":
        from grav16_experiments.models.dynamics_functions.RGNRes import RGNRes
        return RGNRes

    if name == "rgn":
        from grav16_experiments.models.dynamics_functions.RGN import RGN
        return RGN

    if name == "lstm":
        from grav16_experiments.models.dynamics_functions.LSTM import LSTM
        return LSTM

    if name == "vrnn":
        from grav16_experiments.models.dynamics_functions.VRNN import VRNN
        return VRNN

    if name == "dvbf":
        from grav16_experiments.models.dynamics_functions.DVBF import DVBF
        return DVBF

    if name == "kvae":
        from grav16_experiments.models.dynamics_functions.KVAE import KVAE
        return KVAE

    if name == "dkf":
        from grav16_experiments.models.dynamics_functions.DKF import DKF
        return DKF

    if name == "dkf_snp":
        from grav16_experiments.models.dynamics_functions.DKF_SNP import DKF
        return DKF

    # Given no correct model type, raise error
    raise NotImplementedError("Model type {} not implemented.".format(name))


def get_act(act="relu"):
    """
    Return torch function of a given activation function
    :param act: activation function
    :return: torch object
    """
    if act == "relu":
        return nn.ReLU()
    elif act == "leaky_relu":
        return nn.LeakyReLU(0.1)
    elif act == "sigmoid":
        return nn.Sigmoid()
    elif act == "tanh":
        return nn.Tanh()
    elif act == "linear":
        return nn.Identity()
    elif act == 'softplus':
        return nn.modules.activation.Softplus()
    elif act == 'softmax':
        return nn.Softmax()
    elif act == "swish":
        return nn.SiLU()
    else:
        return None


def get_exp_versions(model, exptype):
    """ Return the version number for the latest lightning log and experiment num """
    # Find version folder path
    top = 0
    for folder in os.listdir("lightning_logs/"):
        try:
            num = int(folder.split("_")[-1])
            top = num if num > top else top
        except ValueError:
            continue

    top += 1
    print("Version {}".format(top))

    # Set up paths if they don't exist
    if not os.path.exists("experiments/"):
        os.mkdir("experiments/")

    if not os.path.exists("experiments/{}".format(exptype)):
        os.mkdir("experiments/{}/".format(exptype))

    if not os.path.exists("experiments/{}/{}".format(exptype, model)):
        os.mkdir("experiments/{}/{}".format(exptype, model))

    # Find version folder path
    exptop = 0
    for folder in os.listdir("experiments/{}/{}/".format(exptype, model)):
        try:
            num = int(folder.split("_")[-1])
            exptop = num if num > exptop else exptop
        except ValueError:
            continue

    exptop += 1
    print("Exp Top {}".format(exptop))
    return top, exptop


def determine_annealing_factor(n_updates, min_anneal_factor=0.0, anneal_update=10000):
    """
    Handles annealing the KL restriction over a number of update steps to slowly introduce the regularization
    to ensure a strong initial fit has been set
    :param min_anneal_factor: minimum
    :param anneal_update: over how long of updates to apply the annealing factor
    :param epoch: current epoch number
    :param n_batch: number of total batches within an epoch
    :param batch_idx: current batch idx within the epoch
    :return: weight of the kl annealing factor for the loss term
    """
    if anneal_update > 0 and n_updates < anneal_update:
        anneal_factor = min_anneal_factor + \
            (1.0 - min_anneal_factor) * (
                (n_updates / anneal_update)
            )
    else:
        anneal_factor = 1.0
    return anneal_factor


class CosineAnnealingWarmRestartsWithDecayAndLinearWarmup(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr, :math:`T_{cur}`
    is the number of epochs since the last restart and :math:`T_{i}` is the number
    of epochs between two warm restarts in SGDR:
    """

    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, verbose=False, warmup_steps=350, decay=1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch
        super(CosineAnnealingWarmRestartsWithDecayAndLinearWarmup, self).__init__(optimizer, last_epoch, verbose)

        # Decay attributes
        self.decay = decay
        self.initial_lrs = self.base_lrs

        # Warmup attributes
        self.warmup_steps = warmup_steps
        self.current_steps = 0

    def get_lr(self):
        return [
            (self.current_steps / self.warmup_steps) *
            (self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2)
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        """Step could be called after every batch update"""
        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if self.T_cur + 1 == self.T_i:
            if self.verbose:
                print("multiplying base_lrs by {:.4f}".format(self.decay))
            self.base_lrs = [base_lr * self.decay for base_lr in self.base_lrs]

        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1

            if self.current_steps < self.warmup_steps:
                self.current_steps += 1

            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult

        self.last_epoch = math.floor(epoch)

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
                param_group, lr = data
                param_group['lr'] = lr
                self.print_lr(self.verbose, i, lr, epoch)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


def strtobool(val):
    """Convert a string representation of truth to true (1) or false (0).
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.
    Raises ValueError if 'val' is anything else.
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1', 'True',  'T',  'true'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0', 'False', 'F', 'false'):
        return False
    else:
        raise ValueError("invalid truth value %r" % (val,))


def find_best_epoch(ckpt_folder):
    """
    Find the highest epoch in the Test Tube file structure.
    :param ckpt_folder: dir where the checpoints are being saved.
    :return: float of the highest epoch reached by the checkpoints.
    """
    best_ckpt = None
    best_epoch = None
    best = np.inf
    filenames = os.listdir(f"{ckpt_folder}/checkpoints/")
    for filename in filenames:
        if "last" in filename:
            continue

        test_value = float(filename[:-5].split("mse")[-1])
        test_epoch = int(filename.split('-')[0].replace('epoch', ''))
        if test_value < best:
            best = test_value
            best_ckpt = filename
            best_epoch = test_epoch

    return best_ckpt, best_epoch