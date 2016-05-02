from sklearn.base import BaseEstimator, TransformerMixin
from collections import OrderedDict
import numpy as np
import pandas as pd
class ResponseAggregationImputer(BaseEstimator, TransformerMixin):
    def __init__(self, min_freq=100, min_freq_ratio=0.0,
                 aggregators = ['mean'], report_freq_ratio=False,
                 keep_old = False
                ):
        self.min_freq = min_freq
        self.min_freq_ratio = min_freq_ratio
        self.aggregators = aggregators
        self.report_freq_ratio = report_freq_ratio
        self.keep_old = keep_old
    def fit(self, df, y=None):
        N = df.shape[0]
        min_freq = max(self.min_freq, np.ceil(N*self.min_freq_ratio))
        self.mappers_ = {}
        self.features_ = list(df.columns)
        for col in self.features_:
            self.mappers_[col] = {}
            group_by = pd.groupby(y, by=df[col])
            group_by_count = group_by.count()
            active_groups = group_by_count[group_by_count >= min_freq].index
            if self.report_freq_ratio:
                self.mappers_[col]['freq_ratio'] = (group_by_count/N)[active_groups].to_dict()
            for agg in self.aggregators:
                self.mappers_[col][agg] = group_by.agg(agg)[active_groups].to_dict()
        return self
    def transform(self, df):
        new_df = OrderedDict()
        for col in self.features_:
            if self.keep_old:
                new_df[col] = df[col]
            if self.report_freq_ratio:
                new_df[col+'__'+'freq_ratio'] = [self.mappers_[col]['freq_ratio'].get(x, np.nan) for x in df[col]]
            for agg in self.aggregators:
                new_df[col+'__response_'+agg] = [self.mappers_[col][agg].get(x, np.nan) for x in df[col]]
        return pd.DataFrame(new_df)