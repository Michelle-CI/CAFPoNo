from collections import defaultdict
import sys
import os
sys.path.append(os.path.abspath(os.getcwd()))

from argparse import Namespace
from pathlib import Path
import numpy as np
import pandas as pd
from src.utils import parse_numeric, AUC, accuracy, preprocess_data
from src.methods import *
from multiprocessing import Pool
from tqdm import tqdm
from itertools import product, chain
import yaml
import traceback

CUR_DIR = os.path.abspath(os.path.dirname(__file__))

def process(task):
    task_id, (method, args, (X, Y)) = task
    try:
        score_XY, score_YX = method(X, Y, args)
        return score_XY, score_YX
    except Exception:
        print(f'Error {task_id = }\n{traceback.format_exc()}')
        return 0, 1

def run(method, data, labels, args):
    n_jobs = args.n_jobs
    tasks = list(enumerate((method, args, (row[0], row[1])) for _, row in data.iterrows()))
    with Pool(n_jobs) as p:
        res = list(tqdm(p.imap(process, tasks, chunksize=1), total=len(tasks)))
    score_XY = [x[0] for x in res]
    score_YX = [x[1] for x in res]
    pred_df = pd.DataFrame(data={
            'SampleID': data.index,
            'score_XY': score_XY,
            'score_YX': score_YX,
            'labels': labels,
        })
    return pred_df

def write_results(method_name, data_name, cfg, df_res):
    res_dir =f"{CUR_DIR}/results/{method_name}/" 
    os.makedirs(res_dir, exist_ok=True)
    # Save prediction
    prediction_file = f"{res_dir}/{data_name}.csv"
    df_res.to_csv(prediction_file, index=False)
    # Save config
    cfg_file = f"{res_dir}/{data_name}_cfg"
    with open(cfg_file, 'w') as cfg_out:
        yaml.dump(vars(cfg), cfg_out, default_flow_style=False)

def read_prediction(method_name, data_name):
    res_dir =f"{CUR_DIR}/results/{method_name}/" 
    prediction_file = f"{res_dir}/{data_name}.csv"
    df_res = pd.read_csv(prediction_file, index_col=['SampleID'])
    return df_res

def evaluate(df_res, metrics):
    score = defaultdict(float)
    for k, func in metrics.items():
        score[k] = func(df_res.labels, df_res)
    return score

def main(args):
    # Main process
    datasets = list(map(preprocess_data, args.data_names))
    methods_datas = product(args.methods, datasets)
    results = [] 
    for method_name, (data_name, data, labels) in methods_datas:
        method = eval(method_name)
        # Prepare data
        if args.num_samples:
            data = data.iloc[:args.num_samples]
            labels = labels[:args.num_samples]
        print(f'-----{method.__name__}: {data_name}------')
        # Read config
        cfg_file = open(f'{CUR_DIR}/cfgs/{method_name}.yaml', 'r')
        cfg = yaml.load(cfg_file, Loader=yaml.FullLoader)
        if cfg is None: cfg = {}
        cfg = Namespace(**cfg) 
        print(cfg)
        # Process
        if args.run:
            pred_df = run(method, data, labels, cfg)
            write_results(method_name=method_name, data_name=data_name, cfg=cfg, df_res=pred_df)
        # Evaluate result
        pred_df = read_prediction(method_name, data_name)
        metrics = {'AUC':AUC, 'Acc': accuracy}
        res_eval = evaluate(pred_df, metrics)
        print(res_eval.items())
        results.append(dict(Method=method_name, Data=data_name, **res_eval))

    if args.save:
        df_res = pd.DataFrame(results)
        df_res.to_csv(f"{CUR_DIR}/results/{args.save_file}", index=False)
            

if __name__ == '__main__':
    with open(f"{CUR_DIR}/config.yaml", 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        args = Namespace(**config) 
    print(args)
    main(args)