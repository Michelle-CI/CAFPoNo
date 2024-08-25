import numpy as np
import torch
from src.methods.modules.AbPNL.train import AbPNLTrainer
from src.utils import full_dag, make_eval
import torch.nn.functional as F
from torch.optim import Adam
import random as py_random


def AbPNL(data, args, prunning=False):
    with torch.random.fork_rng():
        torch.random.manual_seed(args.seed) 
        data = torch.FloatTensor(data)
        data = (data - data.mean(axis=0)) / data.std(axis=0)
        train_size = int(data.shape[0] * args.train_ratio)
        train_data = data[:train_size]
        test_data = data[train_size:]
        if len(test_data) == 0:
            test_data = data.clone()
        params = vars(args)
        params['activation'] = make_eval(params['activation'])
        params['optimizer'] = make_eval(params['optimizer'])
        abpnl = AbPNLTrainer(params)
        abpnl.doit(train_data, test_data)
        top_order = abpnl.causal_order[::-1]
        return full_dag(top_order), top_order