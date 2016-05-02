import numpy as np
from sklearn.base import is_classifier, clone
from sklearn.utils import check_arrays
from sklearn.cross_validation import check_cv
from sklearn.externals.joblib import Parallel, delayed
__all__ = ['cross_val_predict']
def _cross_val_predict(estimator, X, y, train, test, predict_fun):
    X_train =  X[train]
    X_test = X[test]
    y_train = y[train]
    estimator.fit(X_train, y_train)
    y_pred = getattr(estimator, predict_fun)(X_test)
    return y_pred
# there is one cross_val_predict in cross_val_predict now with no predict_fun support
def cross_val_predict(estimator, X, y, cv=5, n_jobs=1, refit=False, predict_fun="predict"):
    X, y = check_arrays(X, y, sparse_format='csr', allow_lists=True)
    cv = check_cv(cv, X, y, classifier=is_classifier(estimator))
    pred = Parallel(n_jobs=n_jobs)(
        delayed(_cross_val_predict)(
            clone(estimator), X, y, train, test, predict_fun)
        for train, test in cv)
    pred = np.concatenate(pred)
    if cv.indices:
        index = np.concatenate([test for _, test in cv])
    else:
        index = np.concatenate([np.where(test)[0] for _, test in cv])
    ## pred[index] = pred doesn't work as expected
    pred[index] = pred.copy()
    if refit:
        return pred, clone(estimator).fit(X,y)
    else:
        return pred
