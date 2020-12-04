
import os
import pandas as pd
import numpy as np
from fbprophet import Prophet
import mlflow
import mlflow.sklearn
import warnings



def item_sales (df,target_attribute,sales_criteria,ranked_items,i):
    item_name=ranked_items[i-1]
    print(item_name)
    df =df[df[target_attribute] == item_name]
    #display(df)
    df=df.groupby([target_attribute,'Date'],as_index=False)[sales_criteria].sum()
    
    df=df.sort_values(['Date'])
    df = df[(df[sales_criteria] >= 0) & (df[sales_criteria] <= 10000)] # remoing the outliers
    return df

