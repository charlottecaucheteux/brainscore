"""From a model_name or path to its brain score"""

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


def get_brain_score_speech(
    feature_files,  # {task:embedding_file}
    subject="avg",  # e.g. "sub-004" or "avg"
    # X
    layers=None,  # tuple, embedding layers to select
    x_pca=0,  # apply PCA on embeddings
    concat_layers=False,  # whether to run for each layer or concat layers
    # Y
    rois=False,  # compute scores on rois
    hemi="L",
    space="fsaverage6",
    TR=1.5,
    y_pca=0,  # apply PCA on brain data
    select_tasks=None,  # wehther to run on a subset of audio tasks
    # Model
    fit_intercept=True,
    alpha_per_target=True,
    n_folds=5,
    n_jobs=10,
    metric="correlate",
    # Output
    average_folds=True,
):
    corr_function = get_metric(metric)

    if subject == "avg":
        # Load avg bold
        print("Using average subject")
        print("Loading BOLD ...")
        # load or compute if does not exists
        mean_bold = get_mean_bold(hemi=hemi)
        print("Done")
        # Select tasks
        audio_tasks = list(mean_bold.keys())
        params = [(task, None, None, None) for task in audio_tasks]
    else:
        # Select audio tasks
        df_task = get_task_df()
        df_task = df_task.query("subject==@subject")
        assert len(df_task), f"Subject {subject} does not exists"
        params = zip(df_task.audio_task, df_task.bold_task,
                     df_task.subject, df_task.onset)
    # Restrict tasks
    if select_tasks is not None:
        params = [k for k in params if k[0] in select_tasks]

    # ---------- Build features (X) and brain (Y) ----------
    features = []
    Y = []
    for task, bold_task, subject, onset in params:  # Loop over tasks

        # --- Load brain data
        if subject == "avg":
            gii_fname = f"{subject}_task-{bold_task}_*space-{space}_hemi-{hemi}_desc-clean.func.gii"
            subj_data = get_bold(
                gii_fname, subject, exclude=True,
                afni_dir=paths.afni_dir_nosmooth)
            if subj_data is None:
                continue
            subj_data = subj_data[onset:, :]
        else:
            subj_data = mean_bold[task]
        print(f"Processing task {task}")

        if subj_data is None:
            # skip task
            continue

        # --- Convert brain data to ROI
        if rois:
            subj_data = bold_to_rois(subj_data, hemi=hemi)

        # --- Load stimulus
        stimuli = get_stimulus(task)  # [T, V]

        # --- Extract features from stimulus
        f = Path(feature_files[task])
        assert f.is_file(), f"{f} does not exists"
        feat = torch.load(f).numpy()
        if layers is None:  # select layers
            layers = np.arange(len(feat))
        feat = feat[list(layers)]  # [K, T, D]

        # --- Concatenate embeddings if necessary
        if concat_layers:
            print("Concatenating layers")
            feat = np.concatenate(list(feat), axis=-1)[None]  # [1, T, K*D]
            print(f"to {feat.shape}")

        # --- Cut extra samples
        n_scans = min(len(subj_data), feat.shape[1])
        feat = feat[:, :n_scans]
        subj_data = subj_data[:n_scans]

        # --- Update X and Y
        features.append(feat)
        Y.append(subj_data.copy())

    # Check not empty
    task_len = [len(i) for i in Y]
    if len(task_len) == 0:
        print("No task left")
        return np.array([np.nan])

    # Merge audio tasks
    features = np.concatenate(features, axis=1)  # (nl, ntr, dim)
    Y = np.concatenate(Y, axis=0)  # (ntr, nvox)
    assert len(Y) == features.shape[1]
    print(
        f"Shapes after concatenation \t deep net: {features.shape} \t brain: {Y.shape}")

    # Brain mapping
    brain_scores = []
    for k, X in enumerate(features):

        print(f"Running for layer {layers[k]}")
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
            apply_fir=False,  # HRF already precomputed
            x_pca=x_pca,
            return_coef=False,
        )

        brain_scores.append(r)

    # Merge layers
    brain_scores = np.stack(brain_scores)
    if average_folds:
        brain_scores = np.nanmean(brain_scores, -1)
    return brain_scores
