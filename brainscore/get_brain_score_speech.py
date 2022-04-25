"""From a model_name or path to its brain score"""


import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import RidgeCV

from . import paths
from .brain.data import get_bold, get_mean_bold, get_stimulus, get_task_df
from .brain.fir import sum_between_tr
# from .encode_hierarch import encode_hierarch, encode_hierarch_stacking
from .brain.to_rois import bold_to_rois
from .mapping import mapping
from .metrics import get_metric

logger = logging.getLogger(__name__)


def get_brain_score_speech(
    feature_files,
    subject="avg",
    layers=None,
    # Exp type
    singlesub=True,
    # X
    x_pca=0,
    # Y
    rois=False,
    hemi="L",
    space="fsaverage6",
    TR=1.5,
    y_pca=0,
    # FIR
    n_delays=5,
    n_delays_start=1,
    # Model
    fit_intercept=False,
    alpha_per_target=True,
    n_folds=5,
    n_jobs=10,
    # Output
    metric="correlate",
    average_folds=True,
    audio=True,
    concat_layers=False,
    concat_conv_trick=False,
    hierarch_concat=False,
    apply_log=False,
):
    corr_function = get_metric(metric)

    if subject == "avg":
        singlesub = False
        print("Using average subject")
        print("Loading BOLD ...")
        mean_bold = get_mean_bold(hemi=hemi)
        params = list(mean_bold.keys())
        print("Done")
    else:
        singlesub = True
        df_task = get_task_df()
        df_task = df_task.query("subject==@subject")
        assert len(df_task), f"Subject {subject} does not exists"
        params = zip(df_task.audio_task, df_task.bold_task,
                     df_task.subject, df_task.onset)

    # ---------- Build features (X) and brain (Y) ----------
    features = []
    Y = []
    padding = None
    for params_ in params:

        # try:
        if singlesub:
            task, bold_task, subject, onset = params_
            # Load bold responses
            gii_fname = f"{subject}_task-{bold_task}_*space-{space}_hemi-{hemi}_desc-clean.func.gii"
            subj_data = get_bold(
                gii_fname, subject, exclude=True,
                afni_dir=paths.afni_dir_nosmooth)
            if subj_data is None:
                continue
            subj_data = subj_data[onset:, :]

        else:
            task = params_
            subj_data = mean_bold[task]
        print(f"Processing task {task}")

        if subj_data is None:
            continue

        if rois:
            subj_data = bold_to_rois(subj_data, hemi=hemi)

        # Load stimulus
        stimuli = get_stimulus(task)

        # Extract features from stimulus
        f = Path(feature_files[task])
        assert f.is_file(), f"{f} does not exists"
        feat = torch.load(f)
        if apply_log:
            feat = torch.log(feat)
        feat = feat.numpy()
        n_scans = min(len(subj_data), feat.shape[1])
        if concat_conv_trick:
            assert audio
            assert concat_layers
            print("WARNING CONCAT TRICK WITH CONV !!!!!!")
            f_conv = str(f).replace("_tr_minmax.pth", "_conv_minmax.pth")
            feat_conv = torch.load(f_conv).numpy()
            feat = list(feat_conv) + list(feat)
        if layers is None:
            layers = np.arange(len(feat))
        if audio:  # HRF already precomputed
            fir_feat = [feat[k][:n_scans] for k in layers]
            subj_data = subj_data[:n_scans]
        else:
            fir_feat = np.zeros(
                (len(layers), len(subj_data), feat.shape[-1]))
            for k, layer in enumerate(layers):
                fir_feat[k] = sum_between_tr(
                    feat[layer], stimuli,
                    n_TR=len(subj_data),
                    TR=TR,
                    merge_func="sum")
        if hierarch_concat:
            assert concat_layers
            logger.info("Concatenating layers")
            concat_fir_feat = np.concatenate(list(fir_feat), axis=-1)
            padding = np.zeros((len(fir_feat), concat_fir_feat.shape[-1]))
            current = 0
            for k, feat in enumerate(list(fir_feat)):
                padding[k, :(current+feat.shape[1])] = 1
                current += feat.shape[1]
                print(f"layer {k}, {current}")
            fir_feat = np.stack([concat_fir_feat]*len(fir_feat))
            logger.info(f"to {fir_feat.shape} and padding: {padding.shape}")
        elif concat_layers:
            logger.info("Concatenating layers")
            fir_feat = np.concatenate(list(fir_feat), axis=-1)[None]
            logger.info(f"to {fir_feat.shape}")
        features.append(fir_feat)

        # Update
        Y.append(subj_data.copy())
        # except Exception as e:
        #     print(f"ERROR for task {params_}, {e}")

    # Check only one task
    task_len = [len(i) for i in Y]
    if len(task_len) == 0:
        print("No task left")
        return np.array([np.nan])

    # Concatenate
    features = np.concatenate(features, axis=1)  # (nl, ntr, dim)
    Y = np.concatenate(Y, axis=0)  # (ntr, nvox)
    assert len(Y) == features.shape[1]
    print(
        f"Shapes after concatenation \t deep net: {features.shape} \t brain: {Y.shape}")

    # Brain mapping
    brain_scores = []
    for k, X in enumerate(features):

        print(f"Running for layer {layers[k]}")

        if padding is not None:
            X = X[:, np.where(padding[k])[0]]

        print(f"X.shape {X.shape}")

        r = np.zeros((Y.shape[1], n_folds))
        valid = Y.std(0) > 0

        r[valid] = mapping(
            X,
            Y[:, valid],
            corr_function=corr_function,
            model=RidgeCV(
                np.logspace(-1, 8, 10),
                fit_intercept=fit_intercept,
                alpha_per_target=alpha_per_target,
            ),
            n_folds=n_folds,
            n_jobs=n_jobs,
            average_folds=False,
            y_pca=y_pca,
            n_delays=n_delays,
            apply_fir=not audio,
            n_delays_start=n_delays_start,
            x_pca=x_pca,
            return_coef=False,
        )

        brain_scores.append(r)

    # Reorder
    brain_scores = np.stack(brain_scores)
    if average_folds:
        brain_scores = np.nanmean(brain_scores, -1)
    return brain_scores
