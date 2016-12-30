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

import os
import pickle
def with_cache(cache_file):
    def f(g):
        def gg(*args, **kargs):
            if os.path.exists(cache_file):
                return pickle.load(open(cache_file, 'rb'))
            else:
                res = g(*args, **kargs)
                pickle.dump(res, open(cache_file, 'wb'),
                            protocol=pickle.HIGHEST_PROTOCOL)
                return res
        return gg
    return f


from collections import OrderedDict
def list_to_location_map(l):
    return dict(zip(l, range(len(l))))
class DataFrameNDArrayWrapper:

    def __init__(self, df):
        self.d = df.values
        self.row_names = df.index.tolist()
        self.column_names = df.columns.tolist()
        self.row_name_to_loc = list_to_location_map(self.row_names)
        self.column_name_to_loc = list_to_location_map(self.column_names)

    def get_row_as_dict(self, row_name):
        i = self.row_name_to_loc[row_name]
        row = self.d[i]
        return OrderedDict(zip(self.column_names, row))

def auto_convert_dataframe_for_ndarray_function(f):
    def g(X, *args, **kwargs):
        need_convert = isinstance(X, pd.DataFrame)
        if need_convert:
            X_ = X.values
        else:
            X_ = X
        Y = f(X_, *args, **kwargs)
        if need_convert:
            return pd.DataFrame(Y, index=X.index, columns=X.columns)
        else:
            return Y
    return g