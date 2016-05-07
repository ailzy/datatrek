import numpy as np

def sample_n(data, n, seed=None):
    rng = np.random.RandomState(seed)
    N = data.shape[0]
    return data.iloc[rng.choice(range(N), n)]