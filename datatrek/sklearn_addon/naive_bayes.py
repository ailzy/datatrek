from sklearn.naive_bayes import MultinomialNB
import numpy as np

class MultinomialNB2(MultinomialNB):
    """
    smoothing factor for a class is alpha*raw_event_count
    """
    def _update_class_log_prior(self):
        """Apply smoothing to raw counts and recompute log probabilities"""
        smoothed_fc = self.feature_count_ + self.alpha*self.class_count_[:, np.newaxis]
        smoothed_cc = smoothed_fc.sum(axis=1)

        self.feature_log_prob_ = (np.log(smoothed_fc) -
                                  np.log(smoothed_cc.reshape(-1, 1)))