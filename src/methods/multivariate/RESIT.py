import lingam
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from src.utils import full_dag

def RESIT(data, args, **kwargs):
    df_data = pd.DataFrame(data=data)
    reg = GaussianProcessRegressor(random_state=args.seed)
    model = lingam.RESIT(regressor=reg)
    model.fit(df_data)
    causal_order = model.causal_order_
    dag = full_dag(causal_order)
    return dag, causal_order
