from scipy.stats.mstats import mquantiles
import numpy as np
import pandas as pd

def remove_outlier_by_IQR(x, alpha=1.5):
    q1, q3 = mquantiles(x,[0.25,0.75])
    IQR = q3-q1
    mask = (x<=q3+alpha*IQR) & (x>=q1-alpha*IQR)
    return x[mask]

def remove_outlier_by_quantile(x, limits=[0.05,0.95]):
    l1, l2 = mquantiles(x,limits)
    mask = (x<=l2) & (x>=l1)
    return x[mask]

def remove_outlier_by_std(x, alpha=3):
    x_ = remove_outlier_by_quantile(x)
    mean = np.mean(x_)
    std = np.std(x_)
    mask = (x<=mean+alpha*std) & (x>=mean-alpha*std)
    return x[mask]

remove_null_values_in_dict = lambda x: dict((k,v) for k, v in x.items() if pd.notnull(v))

def df_to_records(df):
    return df.to_dict('records')

def df_to_clean_records(df):
    return list(map(remove_null_values_in_dict, df_to_records(df)))

