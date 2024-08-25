from argparse import Namespace
from collections import defaultdict
from itertools import product
import os
import sys
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import yaml
sys.path.append(os.path.abspath(os.getcwd()))

from src.data.synthproblems import generate_samples, PostNonlinearNToOne, LinearNToOne
from src.methods import CAFPoNoMulti, AbPNLMulti, CAM, SCORE, RESIT, NPVar
from src.utils import full_dag, read_cfg, Missing, Reverse, Extra, SHD, Dorder, Timer
from src.pruning import cam_pruning, cnf_pruning
from cdt.metrics import SID

CUR_DIR = os.path.abspath(os.path.dirname(__file__))


def map_method_cfg(method, num_variables):
    cfg_file = f'{CUR_DIR}/cfg/{method}.yaml'
    cfg = read_cfg(cfg_file)
    cfg.num_variables = num_variables
    return cfg


def main(args):
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
            with Timer() as timer:
                full_A, top_order = model(data, cfg, prunning=False)
            # Measure running time
            running_time = timer.elapsed
            print(f'{running_time=}')
            results.append(dict(Trial=irun, Method=method, Time=running_time))
        results = pd.DataFrame(results)
        results.to_csv(f'{CUR_DIR}/results/time-{num_vars}-{method}.csv')

def plot(args):
    df = []
    for method, num_vars in product(args.methods, args.num_variables):
        file = f'{CUR_DIR}/results/time-{num_vars}-{method}.csv'
        res = pd.read_csv(file)
        res['Method'] = method
        res['Num_vars'] = num_vars
        df.append(res)
    df = pd.concat(df)
    matplotlib.rc('font', family='DejaVu Sans')
    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['lines.linewidth'] = 1
    matplotlib.rcParams['text.latex.preamble'] = r'\boldmath'
    plt.figure(figsize=(7,2))
    ax = sns.lineplot(df, x='Num_vars', y='Time', hue='Method', errorbar='se', err_style='band', palette=['steelblue', 'red'], marker='o')
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xlabel(r'\textbf{\#Variables}', fontsize=10)
    ax.set_ylabel(r'Running time (s)', fontsize=12)
    legend_handles, _= ax.get_legend_handles_labels()
    ax.legend(legend_handles, ['AbPNL', 'CAF-PoNo (Ours)'], 
          bbox_to_anchor=(0.3,1), 
          title='')
    ax.grid(axis='y', linestyle='--')
    plt.tight_layout()
    ax.get_figure().savefig(f'{CUR_DIR}/running-time.pdf', dpi=300)

if __name__ == '__main__':
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(cur_dir, 'config-time.yaml'), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        args = Namespace(**config) 
    main(args)
    plot(args)