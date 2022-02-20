# %load src/encode
import numpy as np
from numpy.random import permutation
from sklearn.metrics import pairwise_distances, r2_score


def get_metric(metric_name):
    """

    Parameters
    ----------
    metric_name : str
        Should be in
        dico = {
        "correlate": correlate,
        "t_correlate": t_correlate,
        "v2v": v2v,
        "v2v_per_voxel": v2v_per_voxel,
        "jr_2v2": jr_2v2,
    }

    Returns
    -------
    metric_name
    """
    dico = {
        "correlate": correlate,
        "t_correlate": t_correlate,
        "v2v": v2v,
        "v2v_per_voxel": v2v_per_voxel,
        "jr_2v2": jr_2v2,
        "r2": r2_score,
    }
    assert metric_name in dico
    return dico[metric_name]


def v2v(true, pred, metric="cosine"):
    assert len(true) == len(pred)
    if len(true) <= 1:
        acc = 1
    else:
        ns = len(true)
        first = permutation(ns)  # first group of TR
        second = permutation(ns)  # second group of TR
        i = 0
        while (first == second).any() and i < 10:  # check that distinct TRs in pairs
            print("invalid", len(first))
            first[first == second] = np.random.choice((first == second).sum())
            i += 1

        correct = 0
        for i, j in zip(first, second):
            r = pairwise_distances(
                true[[i, j]], pred[[i, j]], metric
            )  # compute the 4 distances
            diag = np.diag(r).sum()  # distances of corresponding TR
            cross = r.sum() - diag  # distance of cross TR
            correct += 1 * (diag < cross)  # comparison
        acc = correct / ns
    return np.array(acc)[None]


def jr_2v2(true, pred, metric="cosine"):
    """Tsonova et al 2019 https://arxiv.org/pdf/2009.08424.pdf"""
    assert len(true) == len(pred)
    ns = len(true)
    first = permutation(ns)  # first group of TR
    second = permutation(ns)  # second group of TR
    while (first == second).any():  # check that distinct TRs in pairs
        first[first == second] = np.random.choice((first == second).sum())

    r = pairwise_distances(true, pred)
    s1 = r[first, first] + r[second, second]
    s2 = r[first, second] + r[second, first]

    acc = np.mean(1.0 * (s1 < s2))
    return acc[None]


def v2v_per_voxel(true, pred):
    assert len(true) == len(pred)
    if len(true) <= 2:
        print("invalid")
        return np.ones(true.shape[-1])
    ns = len(true)
    first = permutation(ns)
    second = permutation(ns)
    i = 0
    while (first == second).any() and i < 10:
        print("invalid", len(first))
        first[first == second] = np.random.choice((first == second).sum())
        i += 1

    correct = np.zeros(true.shape[-1])
    for i, j in zip(first, second):
        r = np.abs(true[[i, j]][None] - pred[[i, j]][:, None])
        diag = r[0, 0] + r[1, 1]
        correct += 1 * (diag < r.sum((0, 1)) - diag)
    acc = correct / ns
    return acc


def correlate(X, Y):
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    SX2 = (X ** 2).sum(0) ** 0.5
    SY2 = (Y ** 2).sum(0) ** 0.5
    SXY = (X * Y).sum(0)
    return SXY / (SX2 * SY2)


def t_correlate(X, Y):
    return correlate(X.T, Y.T)
