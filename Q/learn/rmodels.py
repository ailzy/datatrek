import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.numpy2ri import numpy2ri
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings
import numpy as np
robjects.conversion.py2ri = numpy2ri
R = robjects.r

class LogisticRegressionCV(BaseEstimator, ClassifierMixin):
    def __init__(self, binary_classification=True, n_jobs=1, random_state=None, glmnet_params={}):
        """
        binary_classification: if True use family = binomial, else multinomial
        random_state: int or None, set.seed
        n_jobs: number of workers
        glmnet_params: params for glmnet
        """
        self.random_state = random_state
        self.glmnet_params = glmnet_params
        self.binary_classification = binary_classification
        warnings.warn('No validity check for glmnet_params')
        self.n_jobs = n_jobs
    def __del__(self):
        if  hasattr(self, 'cluster_'):
            R['stopCluster'](self.cluster_)
    def fit(self, X, y):
        importr('glmnet')
        family = 'binomial' if self.binary_classification else 'multinomial'
        if self.random_state is not None:
            R['set.seed'](self.random_state)
        if self.n_jobs > 1:
            importr('doParallel')
            self.cluster_ = R['makeCluster'](self.n_jobs)
            R['registerDoParallel'](self.cluster_)
        
        self.classes_ = np.unique(y)
        self.classes_.sort()
        y = R['factor'](y, levels=self.classes_)
        self.R_model_ = R['cv.glmnet'](X, y, family=family, parallel=(self.n_jobs>1), **self.glmnet_params)

        return self
    def predict_proba(self, X):
        importr('glmnet')
        pred = R['predict'](self.R_model_, X, type="response")
        pred = np.squeeze(np.asarray(pred))
        if self.binary_classification:
            pred = np.vstack((1-pred, pred)).T
        return pred
    def predict(self, X):
        prob = self.predict_proba(X)
        return self.classes_[prob.argmax(axis=1)]
