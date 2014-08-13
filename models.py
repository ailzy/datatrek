from sklearn.base import BaseEstimator, TransformerMixin, clone
from scipy.stats.mstats import mquantiles
import numpy as np
import numpy.ma as ma
class MissingValueFiller(BaseEstimator, TransformerMixin):
    def __init__(self, v, add_missing_indicator=True):
        self.v = v
        self.add_missing_indicator = add_missing_indicator
    def fit(self, X, y=None):
            return self
    def transform(self, X):
        import pandas as pd
        import numpy as np
        X = pd.DataFrame(X).fillna(self.v).values
        if self.add_missing_indicator:
            X_ = pd.isnull(X)
            return np.hstack((X,X_))
        else:
            return X
class Winsorizer(BaseEstimator, TransformerMixin):
    """
    squeeze values in [Q1-alpha*IQR, Q3+alpha*IQR]
    keep nan values
    """
    def __init__(self, alpha=5):
        self.alpha = alpha
    def fit(self, X, y=None):
        """
        expect 2-d np.ndarry
        """
        X_ = ma.masked_invalid(X)
        quartiles_ = mquantiles(X_, axis=0, prob=[0.25, 0.75])
        assert not quartiles_.mask.any()
        quartiles_ = quartiles_.data
        IQRs = np.diff(quartiles_,axis=0).ravel()
        self.limits_ = quartiles_
        self.limits_[0,:] -= self.alpha*IQRs
        self.limits_[1,:] += self.alpha*IQRs
        return self
    def transform(self, X):
        X_new = X.copy()
        for j in range(X.shape[1]):
            X_j =  X_new[:,j]
            mask = np.isfinite(X_j)
            X_j[(X_j>self.limits_[1,j]) & mask] = self.limits_[1,j]
            X_j[(X_j<self.limits_[0,j]) & mask] = self.limits_[0,j]
        return X_new

class EachGroupModel(BaseEstimator):
    """
    train separate models in different models
    """
    def __init__(self, group_index, base_estimator=None):
        """
        group_index: int, index of column of group
        baseestimator: estimator for each group
        exact one of estimator and estimators should be set
        """
        self.group_index = group_index
        self.estimator = estimator
    def fit(self, X, y):
        G = X[:, self.group_index] # G will not be changed
        self.group_levels_ = np.unique(G)
        self.group_levles_.sort()
        
        mask_c = np.zeros(X.shape[1],dtype=bool)
        mask_c[self.group_index] = True

        self.estimators_ = {}
        for g in self.groups_:
            mask_r = G==g
            self.estimators_[g] = clone(self.estimator).fit(X_[mask,:], y[mask])
        

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
