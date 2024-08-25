from dodiscover import make_context, toporder
import pandas as pd
from src.utils import full_dag

def CAM(data, args, **kwargs):
    df_data = pd.DataFrame(data=data)
    context = make_context().variables(data=df_data).build()
    model = toporder.CAM()
    model.fit(df_data, context)
    causal_order = model.order_
    dag = full_dag(causal_order)
    return dag, causal_order