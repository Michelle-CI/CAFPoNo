from argparse import Namespace
import contextlib
import logging
from multiprocessing import Pool
import time
from typing import NamedTuple
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import yaml
import torch.nn.functional as F
from torch.optim import Adam

@contextlib.contextmanager
def Timer(name=None, verbose=None):
    if verbose:
        logging.info(f'Starting {name}...')
    start = time.perf_counter()
    timer = NamedTuple('timer', elapsed=str)
    yield timer
    timer.elapsed = time.perf_counter() - start
    if verbose:
        logging.info(f'Finished {name} in {timer.elapsed:.3f}s\n')

def pmap(func, tasks, n_jobs, verbose):
    if n_jobs == 1:
        res = list(map(func, tqdm(tasks, disable=not verbose)))
    else:
        with Pool(n_jobs) as p:
            res = list(tqdm(p.imap_unordered(func, tasks), total=len(tasks), disable=not verbose))
    return res

def make_eval(func_name):
    if isinstance(func_name, str):
        return eval(func_name)
    return func_name

def parse_numeric(df):
    parse_cell = lambda cell: np.fromstring(cell.replace('[', '').replace(']', ''), dtype=np.float64, sep=" ").flatten()
    df = df.applymap(parse_cell)
    return df

def read_cfg(cfg_file):
    cfg_file = open(cfg_file, 'r')
    cfg = yaml.load(cfg_file, Loader=yaml.FullLoader)
    if cfg is None: cfg = {}
    cfg = Namespace(**cfg) 
    return cfg

def full_dag(top_order):
    d = len(top_order)
    A = np.zeros((d, d))
    for i, var in enumerate(top_order):
        A[var, top_order[i+1: ]] = 1
    return A

def AUC(y_true, df_res):
    """
    The mean AUC score of 2 directions A->B and B->A.
    """
    predictions = df_res.score_XY - df_res.score_YX 
    try:
        ret = (roc_auc_score(y_true == 1, predictions) + roc_auc_score(y_true == -1, -predictions)) / 2
        return ret
    except:
        return 0

def accuracy(y_true, df_res):
    predictions = (df_res.score_XY > df_res.score_YX) * 1
    predictions[predictions == 0] = -1
    true_pred = (y_true == predictions).sum()
    return true_pred / len(y_true)

def Dorder(order, adj):
    err = 0
    for i in range(len(order)):
        err += adj[order[i+1:], order[i]].sum()
    return err

def preprocess_data(name, data_dir='data/', return_dataset=False):
    data_file = f'{data_dir}/{name}.csv'
    df = pd.read_csv(data_file, index_col=['SampleID'])
    df[['A', 'B']] = parse_numeric(df[['A', 'B']])
    data = df[['A', 'B']]
    labels = df['Target'].to_numpy()
    if return_dataset:
        return name, data, labels, df['Dataset'].to_numpy()
    return name, data, labels

def standardize(x):
    """
    X: nparray (n)
    """
    return (x - x.mean()) / x.std()

def d_edit(a, b):
    return np.logical_xor((a != 0), (b != 0)).sum()

def n_rev(a, b):
    return ((a != 0) * (b != 0).T).sum()

def strip_outliers(x):
    l, r = np.quantile(x, q=[0.025, 0.975], axis=0)
    x = np.clip(x, l, r)
    return x

def Missing(A_true, A_pred):
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(A_pred + A_pred.T))
    cond_lower = np.flatnonzero(np.tril(A_true + A_true.T))
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    return len(missing_lower)

def Extra(A_true, A_pred):
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(A_pred + A_pred.T))
    cond_lower = np.flatnonzero(np.tril(A_true + A_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    return len(extra_lower)

def Reverse(A_true, A_pred):
    # linear index of nonzeros
    pred = np.flatnonzero(A_pred == 1)
    cond = np.flatnonzero(A_true)
    cond_reversed = np.flatnonzero(A_true.T)
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    return len(reverse)

# https://github.com/xunzheng/notears/blob/master/notears/utils.py
def SHD(A_true, A_pred, method_name=None):
    # linear index of nonzeros
    pred = np.flatnonzero(A_pred == 1)
    cond = np.flatnonzero(A_true)
    cond_reversed = np.flatnonzero(A_true.T)
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(A_pred + A_pred.T))
    cond_lower = np.flatnonzero(np.tril(A_true + A_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    # print(f'{method_name = }: {extra_lower = }, {missing_lower = }, {reverse = }')
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return shd