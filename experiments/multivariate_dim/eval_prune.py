from collections import defaultdict
from pathlib import Path
import sys
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append(os.path.abspath(os.getcwd()))
from argparse import Namespace
from itertools import product
import yaml
import pandas as pd
import numpy as np
from src.utils import pmap, read_cfg
from src.utils import full_dag, read_cfg, Missing, Reverse, Extra, SHD, Dorder
from cdt.metrics import SID
from src.data.synthproblems import generate_samples, PostNonlinearNToOne
from src.pruning import cnf_pruning, cam_pruning

CUR_DIR = os.path.abspath(os.path.dirname(__file__))

def map_method_cfg(method, num_variables):
    cfg_file = f'{CUR_DIR}/cfg/{method}.yaml'
    cfg = read_cfg(cfg_file)
    cfg.num_variables = num_variables
    return cfg

def parse_arr(x):
    x = x.replace("[", "").replace("]", "").replace("\n", "").replace(",", "")
    arr = x.split(" ")
    arr = np.array([int(float(i)) for i in arr])
    return arr

def read_result(filename): 
    df_res = pd.read_csv(filename)
    obj_cols = ['gt_A', 'pred_A', 'pred_order', 'full_A']
    df_res[obj_cols] = df_res[obj_cols].applymap(parse_arr)
    return df_res

def evaluate(ground_truth, pred_DAG, metrics):
    N = int(len(ground_truth) ** 0.5)
    return [m(ground_truth.reshape(N, N), pred_DAG.reshape(N, N)) for m in metrics]

def get_data(args):
    data = []
    for irun in range(args.num_trials):
        data_point, _ = generate_samples(seed=irun, **args.data)
        data.append(data_point)
    return data

def process(task): 
    i, X, perm, gt, method = task
    if method == 'cnf': 
        pred = cnf_pruning(X, perm, threshold=0.001, normalize=True, lr=0.01, max_epochs=200, n_components=8)
        # pred = cnf_pruning(X, perm, threshold=0.1, normalize=True,) 
    elif method == 'cam': 
        pred = cam_pruning(X, perm, threshold=0.1)
    else: 
        raise ValueError('Invalid pruning method.')
    return dict(Trial=i, Data=X, pred_order=perm, gt_A=gt, pred_A=pred.flatten(), prune_method=method)

def to_string(obj):
    if isinstance(obj, np.ndarray):
        return str(obj.tolist())
    elif isinstance(obj, dict):
        return {key: to_string(val) for key, val in obj.items()}
    return obj
    
def save_file(data: pd.DataFrame, outdir: Path, outfile: str):
    outdir.mkdir(parents=True, exist_ok=True)
    data = data.applymap(to_string)
    data.to_csv(outdir / outfile, index=False)

def load_file(path: Path):
    data = pd.read_csv(path, header=0)
    object_cols = ['Data', 'pred_order', 'gt_A', 'pred_A']
    data[object_cols] = data[object_cols].applymap(lambda x: np.array(eval(x)))
    return data

def run(args):
    args.data['n_to_one'] = eval(args.data['n_to_one'])
    for method, num_vars in product(args.methods, args.num_variables):
        print(f'{num_vars=}')
        if not args.run:
            print_result(filename=Path(CUR_DIR)/f'results/pruning-{num_vars}.csv')
            continue
        args.data['num_variables'] = num_vars
        data = get_data(args)
        print(f'Method: {method} - {num_vars} variables')
        res_file = f'{CUR_DIR}/results/{method}/{num_vars}variables.csv'
        df_res = read_result(res_file).iloc[:args.num_trials]
        prune_methods = ['cnf', 'cam']
        tasks = []
        for method in prune_methods:
            tasks = tasks + [(i, data_point, order, gt, method) 
                    for i, (data_point, order, gt) in enumerate(zip(data, df_res['pred_order'].values, df_res['gt_A'].values))]
        prune_res = pmap(process, tasks, n_jobs=6, verbose=True)
        df_prune = pd.DataFrame(prune_res)
        df_full = pd.DataFrame({
            'Trial': range(args.num_trials),
            'Data': data,
            'pred_order': df_res['pred_order'].values,
            'gt_A': df_res['gt_A'].values,
            'pred_A': df_res['pred_order'].apply(full_dag).values,
            'prune_method': ['full' for _ in range(args.num_trials)]
        })
        df = pd.concat([df_prune, df_full]) 
        save_file(df, outdir=Path(CUR_DIR)/'results', outfile=f'pruning-{num_vars}.csv')
        print_result(filename=Path(CUR_DIR)/f'results/pruning-{num_vars}.csv')

def print_result(filename):
    df = load_file(filename)
    metrics = [SID, SHD, Missing, Reverse, Extra]
    df[[m.__name__ for m in metrics]] = df.apply(lambda x: evaluate(x['gt_A'], x['pred_A'], metrics=metrics), axis=1, result_type='expand')
    df['SID'] = df['SID'].astype(int)
    print(df.groupby(['prune_method'])['SID', 'SHD', 'Missing', 'Reverse', 'Extra'].mean())

def plot(args):
    matplotlib.rc('font', family='DejaVu Sans')
    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['lines.linewidth'] = 1
    matplotlib.rcParams['text.latex.preamble'] = r'\boldmath'
    colors = ['orange', 'green', 'purple', 'steelblue', 'red']
    num_vars = args.num_variables
    df_list = []
    for num_var in num_vars:
        file = Path(CUR_DIR)/f'results/pruning-{num_var}.csv' 
        res = load_file(file)
        res['Num_vars'] = num_var
        df_list.append(res)
    df = pd.concat(df_list)
    df[['SHD']] = df.apply(lambda x: evaluate(x['gt_A'], x['pred_A'], metrics=[SHD]), axis=1, result_type='expand')
    plt.figure(figsize=(6,3))
    ax = sns.lineplot(df, x='Num_vars', y='SHD', hue='prune_method', errorbar='se', err_style='bars',)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xlabel(r'\textbf{\#Variables}', fontsize=10)
    ax.set_ylabel(r'SHD', fontsize=12)
    legend_handles, _= ax.get_legend_handles_labels()
    ax.legend(legend_handles, ['CI','CAM','No Pruning'], 
          bbox_to_anchor=(0.25,1), 
          title='')
    ax.grid(axis='y', linestyle='--')
    plt.tight_layout()
    ax.get_figure().savefig(f'{CUR_DIR}/prune.pdf', dpi=300)

if __name__ == '__main__':
    with open(os.path.join(CUR_DIR, 'config.yaml'), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        args = Namespace(**config) 
    # run(args)
    plot(args)
   