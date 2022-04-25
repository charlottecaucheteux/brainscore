from pathlib import Path

import numpy as np
import pandas as pd
from submitit import AutoExecutor

from .brain.data import get_task_df
from .get_brain_score import get_brain_score


# STOP HERE UNTIL FINISH
def compute_brainscore(subject, feature_files, output_file,
                       layers=[8], to_rois=True, x_pca=0, concat_layers=False):
    score = get_brain_score(
        feature_files,
        subject=subject,
        layers=layers,
        # X
        x_pca=x_pca,
        # Y
        rois=to_rois,
        hemi="L",
        space="fsaverage6",
        TR=1.5,
        y_pca=0,
        # FIR
        n_delays=5,
        n_delays_start=0,
        # Model
        fit_intercept=True,
        alpha_per_target=True,
        n_folds=20 if subject == "avg" else 5,
        n_jobs=10,
        # Output
        metric="correlate",
        average_folds=(subject != "avg"),
        concat_layers=concat_layers,
        select_tasks=list(feature_files.keys()),
    )
    print(f"Saving score to {output_file}")
    Path(output_file).parent.mkdir(exist_ok=True, parents=True)
    np.save(output_file, score)
    return np.nanmean(score)


def run_eval_brainscore(feature_files,
                        output_dir,
                        layers=[8],
                        average_bold=False,
                        max_n_subjects=10,
                        subjects=None,
                        hemis=["L"],
                        to_rois=True,
                        x_pca=False,
                        slurm_partition="learnfair",
                        slurm_array_parallelism=200,
                        local=True,
                        overwrite=False,
                        feature_name=None,
                        concat_layers=False,
                        ):
    """
    feature_files is a dictionnary with {task: embedding_file} with embedding_file
    the model's activations. embeddings of shape [n_layers, n_samples, dim]
    """

    if average_bold:
        print("Average subject")
        subjects = ["avg"]
        tasks = get_task_df().audio_task.unique()
    else:
        assert max_n_subjects is None or max_n_subjects > 0
        if subjects is None:
            subjects = get_task_df().subject.unique()
        subjects = subjects[:max_n_subjects]
        tasks = get_task_df().query("subject in @subjects").audio_task.unique()
    for task in tasks:
        assert task in feature_files, f"provide embedding file for task {task}"

    # Compute brain scores
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    df_output_file = output_dir / "brain_scores_df.csv"
    print(
        f"Computing brain scores and saving to {output_dir}, and {df_output_file}")

    params = []
    for hemi in hemis:
        for subject in subjects:
            output_file = output_dir / f"{subject}_{hemi}.npy"
            params.append(
                dict(
                    subject=subject,
                    feature_files=feature_files,
                    output_file=str(output_file),
                    layers=layers,
                    to_rois=to_rois,
                    x_pca=x_pca,
                    feature_name=feature_name,
                    to_run=overwrite or (not output_file.is_file()),
                    concat_layers=concat_layers,
                )
            )

    df = pd.DataFrame(params)
    df.to_csv(df_output_file)
    df_to_run = df.query("to_run")
    keys = ["subject", "feature_files",
            "output_file", "layers", "to_rois", "x_pca", "concat_layers"]

    print(f"{len(df_to_run)} jobs, {df_to_run.subject.nunique()} subjects")
    if local:
        for _, row in df_to_run.iterrows():
            score = compute_brainscore(*[row[k] for k in keys])
            print(f"{score:.2f}")
        jobs = None
    else:
        name = "brainscore_r"
        executor = AutoExecutor(
            f"submitit_jobs/submitit_jobs/{name}")
        executor.update_parameters(
            slurm_partition="learnfair",
            slurm_array_parallelism=slurm_array_parallelism,
            timeout_min=60,
            # cpus_per_tasks=3,
            name=name,
            cpus_per_task=2,
            gpus_per_node=0,
        )

        jobs = executor.map_array(compute_brainscore,
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

    return df, jobs
