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

from src.data.synthproblems import generate_samples, PostNonlinearNToOne, LinearNToOne
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

def map_method_cfg(method, num_variables):
    cfg_file = f'{CUR_DIR}/cfg/{method}.yaml'
    cfg = read_cfg(cfg_file)
    cfg.num_variables = num_variables
    return cfg

def write_results(method, num_vars, results, func_type):
    res_dir =f"{CUR_DIR}/results/{func_type}/{method}/" 
    os.makedirs(res_dir, exist_ok=True)
    file_name = f'{num_vars}variables'
    df_res = pd.DataFrame(results)
    df_res.to_csv(f'{res_dir}/{file_name}.csv')

def print_results(methods, num_variables, metrics = ['D_order', 'SHD', 'SID']):
    total_df = []
    for method, num_vars in product(methods, num_variables):
        file_name = f'{CUR_DIR}/results/{method}/{num_vars}variables.csv'
        df = pd.read_csv(file_name)
        df['num_vars'] = num_vars
        df['method'] = method
        total_df.append(df) 
    total_df = pd.concat(total_df)
    f = {m: ['mean', 'std'] for m in metrics}
    print(total_df.groupby(['num_vars', 'method'])[metrics].agg(f))

def main(args):
    func_type = args.data['n_to_one']
    args.data['n_to_one'] = eval(args.data['n_to_one'])
    cfg_dicts = {f'{method}-{num_vars}': map_method_cfg(method, num_vars) 
                for method, num_vars in product(args.methods, args.num_variables)} 
    for method, num_vars in product(args.methods, args.num_variables):
        print(f'Method: {method} - {num_vars} variables')
        results = [] 
        for irun in range(args.num_trials):
            print(f'========== Trials {irun} =========')
            # Generate data
            args.data['num_variables'] = num_vars
            data, causal_model = generate_samples(seed=irun, **args.data)
            ground_truth = causal_model.adj
            # Run model evaluation
            model = eval(method)
            cfg = cfg_dicts[f'{method}-{num_vars}']
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
            write_results(method=method, num_vars=num_vars, results=results, func_type=func_type) 

if __name__ == '__main__':
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(cur_dir, 'config.yaml'), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        args = Namespace(**config) 
    main(args)
    print_results(args.methods, args.num_variables)
    