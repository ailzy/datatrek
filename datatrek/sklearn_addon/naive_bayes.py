from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.utils.extmath import safe_sparse_dot
import numbers

__all__ = ['MultinomialNB2']

class MultinomialNB2(MultinomialNB):
    """
    changes:
    1. smoothing factor for a class is alpha*raw_event_count; this avoid artifact with zero count
    2. add min_df filter; this remove noise of small support feature; don't use this with partial_fit
    """
    def __init__(self, alpha=1.0, min_df=0, fit_prior=True, class_prior=None):
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.min_df = min_df

    def _count(self, X, Y):
        super()._count(X, Y)
        if self.min_df > 0:
            df = safe_sparse_dot(Y.T, X>0)
            min_df = self.min_df*np.ones_like(self.class_count_) if isinstance(self.min_df, numbers.Integral) else np.floor(self.class_count_ * self.min_df)
            self.feature_count_[df<min_df[:, np.newaxis]] = 0

    def _update_feature_log_prob(self):
        """Apply smoothing to raw counts and recompute log probabilities"""
        smoothed_fc = self.feature_count_ + self.alpha*self.class_count_[:, np.newaxis]
        smoothed_cc = smoothed_fc.sum(axis=1)

        self.feature_log_prob_ = (np.log(smoothed_fc) -
                                  np.log(smoothed_cc.reshape(-1, 1)))