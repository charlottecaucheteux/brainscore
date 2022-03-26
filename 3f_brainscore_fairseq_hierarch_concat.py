
import os
from pathlib import Path

import numpy as np
import pandas as pd
from submitit import AutoExecutor

from brainscore import paths
from brainscore.brain.data import get_task_df
from brainscore.get_brain_score_speech import get_brain_score_speech

# Submitit job
LOCAL = False
OVERWRITE = False
SLURM_PARTITION = "learnlab"
OUTPUT_NAME = "fairseq/vox_hierarch_concat_st5_ct10"

# fMRI
AVERAGE_BOLD = False
N_SUBJECTS = None
HEMIS = ["L", "R"]
TO_ROIS = False
IGNORE_TASK = []
CONCAT_CONV_TRICK = True
HIERARCH_CONCAT = True

# Features
FEATURE_FOLDER = "fairseq_st5_ct10"
types = ["tr"]
scales = ["minmax"]
model_labels = ['unsup_english',
                'sup_english',
                'random_model',
                'unsup_ac_scenes',
                'finetuned_english',
                'unsup_mandarin',
                'unsup_dutch',
                'unsup_french']

CONCATS = [True]
FEATURES = [f"{model_label}_{feature_type}_{scale}"
            for model_label in model_labels
            for feature_type in types for scale in scales]
LAYERS = None

# Mapping
X_PCA = None

def _job_compute_speech_brain_score(subject, feature_files, output_file,
                                    hemi="L",
                                    layers=None, to_rois=True, x_pca=0,
                                    concat_layers=False):
    score = get_brain_score_speech(
        feature_files,
        audio=True,
        subject=subject,
        layers=layers,
        # X
        x_pca=x_pca,
        # Y
        rois=to_rois,
        hemi=hemi,
        space="fsaverage6",
        y_pca=0,
        # Model
        fit_intercept=True,
        alpha_per_target=True,
        n_folds=5,
        # Output
        metric="correlate",
        average_folds=True,
        concat_layers=concat_layers,
        concat_conv_trick=CONCAT_CONV_TRICK,
        hierarch_concat=HIERARCH_CONCAT,
    )
    print(f"Saving score to {output_file}")
    Path(output_file).parent.mkdir(exist_ok=True, parents=True)
    np.save(output_file, score)
    return np.nanmean(score)


if __name__ == "__main__":

    if AVERAGE_BOLD:
        print("Average subject")
        subjects = ["avg"]
        tasks = get_task_df().audio_task.unique()
    else:
        assert N_SUBJECTS is None or N_SUBJECTS > 0

        task_df = get_task_df()

        # Filter subjects on valid tasks
        task_df["all_tasks"] = task_df.groupby(
            "subject")["audio_task"].transform(lambda x: "-".join(x))
        for task in task_df.audio_task.unique():
            print(task)
            task_df[f"col_{task}"] = task_df["all_tasks"].str.contains(task)
        for task in IGNORE_TASK:
            task_df = task_df.query(f"not col_{task}")

        # Set subjects and tasks
        subjects = task_df.subject.unique()[:N_SUBJECTS][::-1]
        tasks = task_df.query("subject in @subjects").audio_task.unique()

    # --------- Brain scores ---------
    feature_dir = paths.speech_embeddings / FEATURE_FOLDER
    output_dir = paths.scores / OUTPUT_NAME
    output_dir.mkdir(exist_ok=True, parents=True)
    df_output_file = output_dir / "brain_scores_df.csv"
    print(
        f"Computing brain scores and saving to {output_dir}, and {df_output_file}")

    params = []
    for concat_layers in CONCATS:
        for feature in FEATURES:
            feature_files = {}
            for task in tasks:
                ff = feature_dir / f"{task}_{feature}.pth"
                assert ff.is_file(), f"{ff} does not exists"
                feature_files[task] = str(ff)
            for hemi in HEMIS:
                for subject in subjects:
                    ext = "_concat" if concat_layers else ""
                    output_file = output_dir / feature / f"{subject}_{hemi}.npy"
                    params.append(
                        dict(
                            subject=subject,
                            feature_files=feature_files,
                            output_file=str(output_file),
                            hemi=hemi,
                            layers=LAYERS,
                            to_rois=TO_ROIS,
                            x_pca=X_PCA,
                            concat_layers=concat_layers,
                            to_run=OVERWRITE or (not output_file.is_file()),
                            feature=feature,
                        )
                    )

    df = pd.DataFrame(params)
    df.to_csv(df_output_file)
    df_to_run = df.query("to_run")
    keys = ["subject", "feature_files", "output_file",
            "hemi", "layers", "to_rois", "x_pca", "concat_layers"]

    print(f"{len(df_to_run)} jobs, {df_to_run.subject.nunique()} subjects")
    if LOCAL:
        for _, row in df_to_run.iterrows():
            score = _job_compute_speech_brain_score(*[row[k] for k in keys])
            print(f"{score:.2f}")

    else:
        name = "brainscore_r"
        executor = AutoExecutor(
            f"submitit_jobs/submitit_jobs/{name}")
        executor.update_parameters(
            slurm_partition=SLURM_PARTITION,
            slurm_array_parallelism=200,
            timeout_min=60*72,
            # cpus_per_tasks=3,
            name=name,
            cpus_per_task=3,
            gpus_per_node=0,
        )

        jobs = executor.map_array(_job_compute_speech_brain_score,
                                  *[df_to_run[k].values for k in keys])

    # Load average scores
    df = pd.read_csv(df_output_file)
    df["is_file"] = df["output_file"].apply(lambda x: Path(x).is_file())
    df["avg_r"] = np.nan
    tmp = df.query("is_file")
    scores = tmp["output_file"].apply(lambda x: np.nanmean(np.load(x))).values
    # scores = [np.nanmean(x) if x is not None else np.nan for x in scores]
    df.loc[df.query("is_file").index, "avg_r"] = scores
    print(f"Averaged scores saved to {df_output_file}")
    df.to_csv(df_output_file)
