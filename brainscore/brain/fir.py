import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def sum_between_tr(feature, word_events, n_TR=None, TR=1.5, merge_func="sum"):
    """
    feature of shape [n_words, dim]
    n_TR < n_words
    """
    word_events["onset"] = word_events["onset"].interpolate()
    word_events["TR"] = (word_events["onset"] // TR).astype(int) + 1
    word_events["word_index"] = np.arange(len(word_events))

    if merge_func == "sum":
        merge_func = np.sum
    elif merge_func == "mean":
        merge_func = np.mean

    X = np.zeros((n_TR, feature.shape[1]))
    for tr in np.arange(n_TR):
        idx = word_events.query("TR==@tr")["word_index"].values
        if not len(idx):
            continue
        X[tr] = merge_func(feature[idx], axis=0)
    return X


class FIRTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_delays=5, start=0):
        self.n_delays = n_delays
        self.start = start

    def fit(self, X=None, y=None):
        return self

    def transform(self, X):
        out = np.concatenate(
            [
                np.roll(X, k, axis=0)
                for k in np.arange(self.start, self.start + self.n_delays)
            ],
            axis=-1,
        )
        assert out.shape[-1] == self.n_delays * X.shape[-1]
        return out
