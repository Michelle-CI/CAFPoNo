from argparse import ArgumentParser, Namespace
from collections import defaultdict
import os
import sys
from cdt.data import load_dataset
import numpy as np
import pandas as pd
import networkx as nx
import torch
sys.path.append(os.path.abspath(os.getcwd()))

from src.utils import SHD, Extra, Missing, Reverse, Dorder, read_cfg
from cdt.metrics import SID
from src.methods import CAFPoNoMulti, AbPNLMulti, CAM, SCORE, RESIT, NPVar
from src.pruning import cnf_pruning

CUR_DIR = os.path.abspath(os.path.dirname(__file__))

def run_model(model, data, ground_truth, args):
    full_A, top_order = model(torch.FloatTensor(data), args, prunning=False)
    pred_dag = cnf_pruning(data, top_order, threshold=0.001, normalize=True, lr=0.01, max_epochs=200, n_components=8)
    return {
        'gt_A': ground_truth,
        'pred_A': pred_dag,
        'pred_order': top_order,
        'full_A': full_A,
    }

def evaluate(ground_truth, pred_DAG, metrics):
    res = defaultdict(float)
    for m in metrics:
        res[m.__name__] = m(ground_truth, pred_DAG)
    return res

def read_data(data_dir):
    # data_dir = 'data/sachs/'
    dag = pd.read_csv(f'{data_dir}/dag.csv', index_col=['source']).to_numpy()
    data = pd.read_csv(f'{data_dir}/data.csv').to_numpy()
    print('Data shape: ', data.shape)
    print('Num edges: ', dag.sum())
    return dag, data

def main(models):
    cfg_dict = vars(read_cfg(f'{CUR_DIR}/config.yaml'))
    adj, data = read_data(cfg_dict['data_dir'])
    metrics = [SHD, SID, Missing, Reverse, Extra]
    total_res = []

    for model_name in models:
        cfg = cfg_dict[model_name]
        if cfg is None: cfg = {}
        cfg = Namespace(**cfg)
        cfg.num_variables = data.shape[1]
        print(cfg)
        model = eval(model_name)
        # Sample data 
        res = run_model(model, data, adj, cfg)
        res_eval = evaluate(res['gt_A'], res['pred_A'], metrics=metrics)
        res_eval['D_order'] = Dorder(res['pred_order'], res['gt_A'])
        total_res.append(dict(Method=model_name, **res, **res_eval, Cfg=cfg))
    total_res = pd.DataFrame(total_res)
    metrics = [m.__name__ for m in metrics] + ['D_order'] 
    print(total_res.groupby(['Method'])[metrics].mean())
    total_res.to_csv(f'{CUR_DIR}/results.csv', index=False)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--models', type=str, nargs='+', default=['CAFPoNoMulti', 'AbPNLMulti', 'CAM', 'RESIT', 'NPVar'],
                            help='Config path for running experiment')
    args = parser.parse_args() 
    main(args.models)
