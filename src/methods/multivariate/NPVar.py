import numpy as np
from pygam import GAM
from pygam.terms import s, TermList
from src.utils import full_dag
import torch

def NPVar(X, args, **kwargs):
    if isinstance(X, torch.Tensor):
        X = X.numpy()
    n, p = X.shape
    node_index = list(range(p))
    verbose = args.verbose
    method = args.method
    condvars = np.var(X, axis=0)
    source = np.argmin(condvars)
    ancestors = [source]
    if verbose:
        print(dict(zip(node_index, condvars)))
        print(ancestors)
    
    while len(ancestors) < p - 1:
        descendants = np.delete(node_index, ancestors)
        condvars = est_condvars(X, descendants, ancestors, verbose, method)
        min_index = np.argmin(condvars)
        source_next = descendants[min_index]
        ancestors.append(source_next)
        if verbose:
            print(dict(zip(descendants, condvars)))
            print(ancestors)

    descendants = np.delete(node_index, ancestors)
    if len(ancestors) < p:
        ancestors += list(descendants)
    if verbose:
        print(ancestors)
    causal_order = ancestors[::-1]
    dag = full_dag(causal_order)
    return dag, causal_order

def est_condvars(X, descendants, ancestors, verbose, method):
    assert method in ['np', 'mgcv']
    condvars = [float('nan')] * len(descendants)
    for i in range(len(descendants)):
        current_node = descendants[i]
        if verbose:
            print("Checking", "X" + str(current_node), " ~ ", " + ".join(["X" + str(a) for a in ancestors]))
        if method == 'np':
            pass
        elif method == 'mgcv':
            mgcv_formula = [s(a, 10) for a in ancestors]
            b1 = GAM(terms=TermList(*mgcv_formula)).fit(X=X, y=X[:, current_node])
            fit_gam = b1.predict(X=X)
            condvar_gam = np.var(X[:, current_node]) - np.var(fit_gam)
            condvars[i] = condvar_gam
    return condvars