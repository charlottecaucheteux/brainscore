import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

# from .encode_hierarch import encode_hierarch, encode_hierarch_stacking
from .brain.fir import FIRTransformer


def mapping(
    X,  # (n, 768 * n_future)
    Y,
    # Encoding
    corr_function=r2_score,
    model=RidgeCV(np.logspace(-1, 8, 10)),
    n_folds=5,
    n_jobs=10,
    average_folds=True,
    # Y
    y_pca=0,
    n_delays_start=1,
    n_delays=6,
    apply_fir=True,
    # X
    x_pca=0,
    return_coef=False,
):

    def clone_list(x):
        clones = []
        for k, est in x:
            clones.append((k, clone(est)))
        return clones

    cv = KFold(n_folds, shuffle=False)

    steps = []
    if x_pca:
        pca_steps = [("pca_scaler", StandardScaler()), ("pca", PCA(x_pca))]
        steps.append(("pca", Pipeline(clone_list(pca_steps))))

    if apply_fir:
        steps.append(("fir", FIRTransformer(n_delays, start=n_delays_start)))

    steps.extend(
        [
            ("scaler", StandardScaler()),
            ("ridge", clone(model)),
        ]
    )
    pipe = Pipeline(steps)

    y_steps = [("scaler", RobustScaler(quantile_range=(0.1, 99.9)))]
    if y_pca:
        y_steps.append(("pca", PCA(y_pca)))

    if return_coef:
        print("Returning coefficients")
        xdim = (x_pca if x_pca else X.shape[1])
        R = np.zeros((Y.shape[-1], xdim * n_delays, n_folds))
    else:
        R = np.zeros((Y.shape[-1], n_folds))

    for i, (train, test) in enumerate(cv.split(X)):
        print(".", end="")
        y_pipe = Pipeline(y_steps)
        y_pipe.fit(Y[train])
        Y_true = y_pipe.transform(Y)

        pipe.fit(X[train], Y_true[train])
        Y_pred = pipe.predict(X[test])
        Y_pred = y_pipe.inverse_transform(Y_pred)
        if return_coef:
            R[:, :, i] = pipe["ridge"].coef_
        else:
            R[:, i] = corr_function(Y[test], Y_pred)
    
    if average_folds:
        R = np.nanmean(R, -1)
        
    return R
