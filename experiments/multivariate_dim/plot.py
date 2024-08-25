import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import matplotlib

matplotlib.rc('font', family='DejaVu Sans')
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['lines.linewidth'] = 1
matplotlib.rcParams['text.latex.preamble'] = r'\boldmath'

def format_metric(metric):
    if metric == 'D_order':
        texts = metric.split('_')
        return (r'${main}_\textrm{{{sub}}}$ ($\downarrow$)'.format(main=texts[0], sub=texts[1]))
    return (r'\textbf{{{metric}}} ($\downarrow$)'.format(metric=metric))


CUR_DIR = os.path.abspath(os.path.dirname(__file__))
methods = ['RESIT',  'NPVar', 'SCORE', 'AbPNLMulti', 'OursMulti',]
colors = ['orange', 'green', 'purple', 'steelblue', 'red']
num_variables = [4, 7, 9, 11]
metrics = ['D_order', 'SHD', 'SID']
total_res= []
for method, num_vars in product(methods, num_variables):
    file_name = f'{CUR_DIR}/results/{method}/{num_vars}variables.csv'
    df = pd.read_csv(file_name)
    df = df.loc[:, metrics]
    df['Method'] = method
    df['Num_variables'] = num_vars
    total_res.append(df)
total_res = pd.concat(total_res)
df = pd.melt(total_res, id_vars=['Method', 'Num_variables'], value_vars=metrics, value_name='Value', var_name='Metric')
df.Metric = df.Metric.map(format_metric)
df.Method = df.Method.map(lambda x: {'OursMulti': r'\textbf{CAF-PoNo (Ours)}'}.get(x, x))
df['Same'] = ''
palette = dict(zip(df.Method.unique(), colors))
g = sns.FacetGrid(df, row='Same', col='Metric', sharey=False, aspect=1.2)
g.map_dataframe(sns.lineplot, x='Num_variables', y='Value', hue='Method', 
                marker='o', markersize=7, errorbar='se', palette=palette, err_style='bars')
g.set_xlabels(r'\textbf{\#Variables}', fontsize=10)
g.set_ylabels('')
g.set_titles(r'{col_name}', size=12)
g.add_legend(label_order=df.Method.unique())
for ax in g.axes.flat:
    ax.grid(axis='y', linestyle='--')
g.tight_layout()
g.savefig(f'{CUR_DIR}/exp_multi_sim.pdf', dpi=300)