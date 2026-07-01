"""Train/test splits for the multi-parameter dynamics study. Three difficulty
levels, each testing something distinct about whether QCL predicts or fits."""
import numpy as np
from sklearn.model_selection import train_test_split


def interpolation_split(X, y, test_size=0.2, random_state=42):
    """Random split: tests whether the model can REPRESENT the 2-D surface."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def parameter_extrapolation_split(X, y, held_out_masses, tol=1e-9):
    """Hold out entire mass slices: train on some m, predict dynamics at UNSEEN m.
    This is the strongest 'unsupervised prediction' test (Ikeda's challenge)."""
    m = X[:, 1]
    is_test = np.zeros(len(m), dtype=bool)
    for mh in held_out_masses:
        is_test |= np.abs(m - mh) < tol
    return X[~is_test], X[is_test], y[~is_test], y[is_test]


def time_extrapolation_split(X, y, t_split):
    """Train on early times (all masses), forecast late times: tests forecasting
    of the parameterized family."""
    t = X[:, 0]
    train = t <= t_split
    return X[train], X[~train], y[train], y[~train]