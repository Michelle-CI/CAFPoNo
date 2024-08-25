from argparse import Namespace
from collections import defaultdict
from itertools import product
import json
import os
import sys
import pandas as pd
import numpy as np
import torch
import yaml
sys.path.append(os.path.abspath(os.getcwd()))

from src.data.synthproblems import generate_samples, PostNonlinearNToOne
from src.methods import CAFPoNoMulti, AbPNLMulti, CAM, SCORE, RESIT, NPVar
from src.utils import full_dag, read_cfg, Missing, Reverse, Extra, SHD, Dorder
from src.pruning import cam_pruning, cnf_pruning
from cdt.metrics import SID

CUR_DIR = os.path.abspath(os.path.dirname(__file__))


def run_model(model, data, ground_truth, args):
    full_A, top_order = model(data, args, prunning=False)
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

def map_method_cfg(method, sample_size):
    cfg_file = f'{CUR_DIR}/cfg/{method}.yaml'
    cfg = read_cfg(cfg_file)
    # cfg.num_samples = sample_size
    return cfg

def write_results(method, num_samples, results):
    res_dir =f"{CUR_DIR}/results/{method}/" 
    os.makedirs(res_dir, exist_ok=True)
    file_name = f'{num_samples}'
    df_res = pd.DataFrame(results)
    df_res.to_csv(f'{res_dir}/{file_name}.csv')

def print_results(methods, num_variables, metrics = ['D_order', 'SHD', 'SID']):
    for method, num_vars in product(methods, num_variables):
        file_name = f'./results/{method}/{num_vars}variables.csv'
        df = pd.read_csv(file_name)
        mean = df[metrics].mean(axis=0)
        std = df[metrics].std(axis=0)
        print(f'Method: {method} - {num_vars} variables')
        result = pd.DataFrame({
            'mean': mean, 
            'std': std
        })
    print(result)

def cal_epoch_by_sample_size(sample_size):
    if sample_size == 100:
        return 50
    if sample_size == 200:
        return 100
    if sample_size == 300 or sample_size == 400:
        return 200
    else:
        return 300

def main(args):
    args.data['n_to_one'] = eval(args.data['n_to_one'])
    min_samples = 100
    max_samples = 1_000
    # num_samples = list(range(min_samples, max_samples + 1, 100))
    num_samples = [100, 300, 600, 800, 1000]
    cfg_files = [f'{CUR_DIR}/cfg/{method}.yaml' for method in args.methods]
    cfg_dicts = {f'{method}': read_cfg(file) for method, file in zip(args.methods, cfg_files)}

    for method, sample_size in product(args.methods, num_samples):
        print(f'Method: {method} - {sample_size} samples')
        results = [] 
        cfg = cfg_dicts[method]
        cfg.num_variables = args.data['num_variables']
        if method  == 'OursMulti':
            cfg.epochs = cal_epoch_by_sample_size(sample_size)
        elif method == 'AbPNLMulti': 
            cfg.n_epoch = cal_epoch_by_sample_size(sample_size)
        args.data['num_samples'] = sample_size
        model = eval(method)
        for irun in range(args.num_trials):
            print(f'========== Trials {irun} =========')
            # Generate data
            data, causal_model = generate_samples(seed=irun, **args.data)
            ground_truth = causal_model.adj
            # Run model evaluation
            res = run_model(model, data, ground_truth, cfg)
            # Evaluate
            # print('D_order: ', num_errors(res['pred_order'], res['gt_A']))
            metrics = [SHD, SID, Missing, Reverse, Extra]
            res_eval = evaluate(res['gt_A'], res['pred_A'], metrics=metrics)
            res_eval['D_order'] = Dorder(res['pred_order'], res['gt_A'])
            print(res_eval.items())
            # Update results
            results.append(dict(Trials=irun, Num_edges=ground_truth.sum(), **res, **res_eval, Cfg=cfg))
        if args.save:
            write_results(method=method, num_samples=sample_size, results=results) 

if __name__ == '__main__':
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(cur_dir, 'config.yaml'), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        args = Namespace(**config) 
    main(args)
    