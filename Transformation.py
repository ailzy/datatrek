from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state, check_arrays
from itertools import combinations
from collections import Counter
import numpy as np
import scipy.sparse as sp
from scipy.misc import comb
__all__ = ['Relabel' ,'Interaction']

class Relabel(BaseEstimator, TransformerMixin):
    '''
    relabel the nominal features by integers
    '''
    def __init__(self, threshold=0, sparse=False):
        self.threshold = threshold
        self.sparse = sparse
    def fit(self, X, y=None):
        '''
        X is an array of shape [n_samples, n_features]

        '''
        X = check_arrays(X, sparse_format='dense')[0]
        m, n = X.shape
        self.map_ = []
        for j in xrange(n):
            counter = Counter(X[:,j])
            keys = [k for k, c in counter.iteritems() if c > self.threshold]
            keys.sort()
            local_map = dict((k,i+1) for i,k in enumerate(keys))
            self.map_.append(local_map)
        return self
    def transform(self, X):
        '''
        labels learned in train set start from 1 and is continous
        label 0 means unknown label
        if sparse is true, Only labels learned from train set has a column
        '''
        if self.sparse:
            return self.transform_sparse(X)
        else:
            return self.transform_dense(X)
    def transform_dense(self, X):
        X = check_arrays(X, sparse_format='dense')[0]
        m, n = X.shape
        newX = np.empty((m, n), dtype=int)
        for j in xrange(n):
            for i in xrange(m):
                try:
                    newX[i,j] = self.map_[j][X[i,j]]
                except KeyError:
                    newX[i,j] = 0
        return newX 
    def transform_sparse(self, X):
        X = check_arrays(X, sparse_format='dense')[0]
        m, n = X.shape
        n_values = np.array([len(d) for d in self.map_])
        start = n_values.cumsum()
        start = np.hstack((0, start))
        ncol = start[-1]
        row_indices = []
        col_indices = []
        for j in xrange(n):
            for i in xrange(m):
                try:
                    label = self.map_[j][X[i,j]]
                    row_indices.append(i)
                    col_indices.append(start[j]+label-1)
                except KeyError:
                    pass
        nnz = len(row_indices)
        data = np.ones(nnz)
        ret = sp.csr_matrix((data, (row_indices, col_indices)), shape=(m, ncol))
        return ret
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
            
class Interaction(BaseEstimator, TransformerMixin):
    '''
    Get interaction features from nominal features
    '''
    def __init__(self, degree=2, threshold=0):
        self.degree = degree
        self.threshold = threshold
    def fit(self, X, y=None):
        '''
        Parameters
        ----------
        X: dense matrix of shape [n_samples, n_features] and type int
        each column is a nominal feature

        Returns
        -------
        dense matrix of shape [n_samples, nchoosek(n_features, self.degree)]
        '''
        X = check_arrays(X, sparse_format='dense', dtype=np.int)[0]
        m, n = X.shape
        self.map_ = []
        for i, indices in enumerate(combinations(range(n), self.degree)):
            counter = Counter()
            for v in X[:, indices]:
                counter[tuple(v)] += 1
            local_map = {}
            new_label = 1
            for v in counter.keys():
                if counter[v] > self.threshold:
                    local_map[v] = new_label
                    new_label += 1
            self.map_.append(local_map)
        return self
    def transform(self, X):
        '''
        if new feature is labeled 0, then it is unknown
        '''
        X = check_arrays(X, sparse_format='dense', dtype=np.int)[0]
        m, n = X.shape
        ret = np.empty((m,comb(n, self.degree, exact=1)))
        for j, indices in enumerate(combinations(range(n), self.degree)):
            for i, v in enumerate(X[:, indices]):
                try:
                    ret[i,j] = self.map_[j][tuple(v)]
                except KeyError:
                    ret[i,j] = 0
        return ret
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
