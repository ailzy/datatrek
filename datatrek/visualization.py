import matplotlib.pylab as plt
import numpy as np
from .util import remove_outlier_by_IQR, remove_outlier_by_quantile, remove_outlier_by_std
def visualize_binary_classification(y_train, pred_train, y_test, pred_test):
    """
    visualize binary classification
    y_train, y_test: true classes, assume labels are [0,1]
    pred_train, pred_test: predicted probability for class 1
    """
    
    # predicted probability
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    levels_train = np.unique(y_train)
    plt.hist([pred_train[y_train==x] for x in levels_train], label=list(map(str,levels_train)))
    plt.legend(loc=0)
    plt.title('probability predicted on train set')
    plt.subplot(1,2,2)
    levels_test = np.unique(y_test)
    plt.hist([pred_test[y_test==x] for x in levels_test], label=list(map(str,levels_test)))
    plt.legend(loc=0)
    plt.title('probability predicted on test set')
    plt.show()
    
def visualize_regression(y_train, pred_train, y_test, pred_test):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(y_train,pred_train,'.')
    plt.subplot(1,2,2)
    plt.plot(y_test,pred_test,'.')
    plt.show()

def visualize_distribution(x):
    x = x[~np.isnan(x)]
    n = len(x)
    funcs = [('original', lambda x:x),
             ('[Q1-1.5*IQR,Q3+1.5*IQR]', remove_outlier_by_IQR),
             ('[Q1-5*IQR,Q3+5*IQR]', lambda x: remove_outlier_by_IQR(x,5)),
             ('tmean$\pm$ 3*std', remove_outlier_by_std),
             ('quantile [0.05,0.95]', remove_outlier_by_quantile)]
    fig, axes = plt.subplots(2,len(funcs))
    fig.set_figheight(10)
    fig.set_figwidth(5*len(funcs))
    for i, (name, f) in enumerate(funcs):
        x_new = f(x)
        coverage = len(x_new)/n
        axes[0,i].boxplot(x_new)
        axes[1,i].hist(x_new)
        axes[0,i].set_title(name+', coverage {:.2g}'.format(coverage))
    return fig

from scipy.stats.mstats import mquantiles
import fastcluster as fst
import seaborn as sns
from scipy.cluster import hierarchy
import matplotlib.patches as patches
def plot_corr_cluster(m, method=1, **kargs):
    sns.set_style('white')

    l = fst.linkage(m, method='average')

    if method == 1:
        # Threshold 1: MATLAB-like behavior
        t = 0.7*max(l[:, 2])
    elif method == 2:
        t = np.median(hierarchy.maxdists(l))
    elif method == 3:
        t= mquantiles(hierarchy.maxdists(l), prob=0.75)[0]
    else:
        raise RuntimeError('no such method')

    # Plot the clustermap
    # Save the returned object for further plotting
    mclust = sns.clustermap(m,
                            linewidths=0,
                            cmap=plt.get_cmap('RdBu'),
                            vmax=1,
                            vmin=-1,
                            row_linkage=l,
                            col_linkage=l,
                            **kargs)

    # Draw the threshold lines
    mclust.ax_col_dendrogram.hlines(t,
                                    0,
                                    m.shape[0]*10,
                                    colors='g',
                                    linewidths=2,
                                    zorder=1)
    mclust.ax_row_dendrogram.vlines(t,
                                    0,
                                    m.shape[0]*10,
                                    colors='g',
                                    linewidths=2,
                                    zorder=1)

    # Extract the clusters
    clusters = hierarchy.fcluster(l, t, 'distance')
    for c in set(clusters):
        # Retrieve the position in the clustered matrix
        index = [x for x in range(m.shape[0])
                 if mclust.data2d.columns[x] in m.index[clusters == c]]
        # No singletons, please
        if len(index) == 1:
            continue

        # Draw a rectangle around the cluster
        mclust.ax_heatmap.add_patch(
            patches.Rectangle(
                (min(index),
                 m.shape[0] - max(index) - 1),
                len(index),
                len(index),
                facecolor='none',
                edgecolor='g',
                lw=3)
        )

    plt.title('Cluster matrix')