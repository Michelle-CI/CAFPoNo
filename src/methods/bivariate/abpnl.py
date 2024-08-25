# from collections.abc import Callable, Sequence, Iterable
from typing import Any, TypeVar, cast, Callable, Sequence, Iterable

import numpy as np
from numpy.typing import NDArray, NBitBase
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch import nn, Tensor
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import trange
import random as py_random
from src.methods.modules.AbPNL.train import AbPNLTrainer
from src.HSIC import HSIC, max_Gaussian_eHSIC
from src.utils import make_eval, standardize

# T = TypeVar('T', bound=np.floating[NBitBase])

def AbPNL(X, Y, args):
    with torch.random.fork_rng():
        torch.random.manual_seed(args.seed) 
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        py_random.seed(args.seed)

        X, Y = map(standardize, (X, Y))
        X, Y = map(lambda x: x.reshape(-1, 1), (X, Y)) 
        data = np.concatenate([X, Y], axis=1)
        data = torch.FloatTensor(data)

        train_size = int(data.shape[0] * args.train_ratio)
        train_data = data[:train_size]
        test_data = data[train_size:]

        params = vars(args)
        params['activation'] = make_eval(params['activation'])
        params['optimizer'] = make_eval(params['optimizer'])
        abpnl = AbPNLTrainer(params)
        score_XY, score_YX = abpnl.doit_bivariate(train_data, test_data)
        return score_XY, score_YX