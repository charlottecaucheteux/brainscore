
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from submitit import AutoExecutor

from .brain.data import get_stimulus, get_task_df
from .deep_net.data import get_activations
from .get_brain_score import get_brain_score


def _wait_until_complete(jobs, max_time_to_wait=50, wait_step=10*60, job_names=None):
    if job_names is None:
        job_names = [j.job_id for j in jobs]
    time_counter = 0
    runs_left = [True]
    while len(runs_left) > 0:
        running = [k for (k, j) in enumerate(jobs) if j.state == "RUNNING"]
        failed = [k for (k, j) in enumerate(jobs) if j.state == "FAILED"]
        pending = [k for (k, j) in enumerate(jobs) if j.state == "PENDING"]
        runs_left = pending + running
        print(f"Failed tasks : {[job_names[k] for k in failed]}")
        print(f"Running tasks : {[job_names[k] for k in running]}")
        print(f"{len(pending)} pending tasks")
        time.sleep(wait_step)
        if time_counter > max_time_to_wait:
            break
        max_time_to_wait += 1

    completed = [k for (k, j) in enumerate(jobs) if j.state == "COMPLETED"]
    print(
        f"Failed tasks : {[(job_names[k], jobs[k].stderr()) for k in failed]}")
    return np.array(completed)

# Compute embeddings


def _job_compute_activations(model_file, task, output_file, max_len=1024):
    # With time window
    print(f"Computing the activations of {model_file} for task {task}")
    stimulus = get_stimulus(task, lower=False)
    activations = get_activations(stimulus,
                                  model_name_or_path=model_file,
                                  max_len=max_len,
                                  device="cuda")
    print(f"Savving activations of shape {activations.shape} to {output_file}")
    torch.save(activations, output_file)
    return True


# STOP HERE UNTIL FINISH
def _job_compute_brain_score(subject, feature_files, output_file,
                             layers=[8], to_rois=True, x_pca=0):
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
        trim_init=2,
        y_pca=0,
        # FIR
        n_delays=5,
        n_delays_start=0,
        # Model
        fit_intercept=False,
        alpha_per_target=True,
        n_folds=20 if subject == "avg" else 5,
        n_jobs=10,
        # Output
        metric="correlate",
        average_folds=(subject != "avg"),
    )
    print(f"Saving score to {output_file}")
    Path(output_file).parent.mkdir(exist_ok=True, parents=True)
    np.save(output_file, score)
    return np.nanmean(score)


def run_eval(model_name_or_path,
             output_path,
             layers=[8],
             average_bold=True,
             n_subjects=20,
             hemis=["L"],
             to_rois=True,
             x_pca=False,
             cache_path="/checkpoint/ccaucheteux/cache",
             delete_cache_after_run=True,
             slurm_partition="learnfair",
             local=False,
             model_max_len=128,
             overwrite=False,
             ):

    # cache_path.rmdir()
    # i = 1
    # while cache_path.exists():
    #     cache_path = Path(str(cache_path) + f"-{i}")
    #     i += 1
    # print(f"Saving to {cache_path}")

    if average_bold:
        print("Average subject")
        subjects = ["avg"]
        tasks = get_task_df().audio_task.unique()
    else:
        assert n_subjects is None or n_subjects > 0
        subjects = get_task_df().subject.unique()[:n_subjects]
        tasks = get_task_df().query("subject in @subjects").audio_task.unique()

    # Deep nets' activations
    name = Path(model_name_or_path).name
    feature_dir = Path(cache_path) / "embeddings" / name
    print(f"Computing deep networks activations to {feature_dir}")
    feature_dir.mkdir(exist_ok=True, parents=True)

    feature_files = {}
    to_run_files = []
    to_run_tasks = []
    for task in tasks:
        feature_file = feature_dir / f"{task}.pth"
        if not feature_file.is_file():
            to_run_files.append(str(feature_file))
            to_run_tasks.append(task)
        feature_files[task] = str(feature_file)

    if local:
        for run_task, run_file in zip(to_run_tasks, to_run_files):
            _job_compute_activations(
                model_name_or_path, run_task, run_file, max_len=model_max_len)
    else:
        name = "brainscore_embeddings"
        executor = AutoExecutor(
            f"submitit_jobs/submitit_jobs/{name}")
        executor.update_parameters(
            slurm_partition="learnfair",
            slurm_array_parallelism=100,
            timeout_min=60 * 72,
            # cpus_per_tasks=3,
            name=name,
            cpus_per_task=4,
            gpus_per_node=1,
        )
        jobs = executor.map_array(_job_compute_activations,
                                  [model_name_or_path]*len(to_run_files),
                                  to_run_tasks,
                                  to_run_files,
                                  [model_max_len]*len(to_run_files))

        # Check jobs done
        completed = _wait_until_complete(
            jobs, max_time_to_wait=50, wait_step=10*60, job_names=to_run_tasks)
        print(
            f"Done computing deep networks activations, \
                {len(completed)}/{len(jobs)} jobs completed")

    # Compute brain scores
    output_dir = Path(output_path)
    output_dir.mkdir(exist_ok=True, parents=True)
    df_output_file = output_dir / "brain_scores_df.csv"
    print(
        f"Computing brain scores and saving to {output_dir}, and {df_output_file}")

    params = []
    for hemi in hemis:
        for subject in subjects:
            output_file = output_dir / \
                "brain_score" / f"{subject}_{hemi}.npy"
            params.append(
                dict(
                    subject=subject,
                    feature_files=feature_files,
                    output_file=str(output_file),
                    layers=layers,
                    to_rois=to_rois,
                    x_pca=x_pca,
                    to_run=overwrite or (not output_file.is_file()),
                    model_name_or_path=str(model_name_or_path),
                )
            )

    df = pd.DataFrame(params)
    df.to_csv(df_output_file)
    df_to_run = df.query("to_run")
    keys = ["subject", "feature_files",
            "output_file", "layers", "to_rois", "x_pca"]

    print(f"{len(df_to_run)} jobs, {df_to_run.subject.nunique()} subjects")
    if local:
        for _, row in df_to_run.iterrows():
            score = _job_compute_brain_score(*[row[k] for k in keys])
            print(f"{score:.2f}")

    else:
        name = "brainscore_r"
        executor = AutoExecutor(
            f"submitit_jobs/submitit_jobs/{name}")
        executor.update_parameters(
            slurm_partition="learnfair",
            slurm_array_parallelism=200,
            timeout_min=60,
            # cpus_per_tasks=3,
            name=name,
            cpus_per_task=2,
            gpus_per_node=0,
        )

        jobs = executor.map_array(_job_compute_brain_score,
                                  *[df_to_run[k].values for k in keys])
        # Check jobs done
        completed = _wait_until_complete(
            jobs, max_time_to_wait=50, wait_step=10*60, job_names=df_to_run["subject"].values)
        print(
            f"""Done computing deep networks activations, \
                {len(completed)}/{len(jobs)} jobs completed""")

    # Load average scores
    df = pd.read_csv(df_output_file)
    df["is_file"] = df["output_file"].apply(lambda x: Path(x).is_file())
    df["avg_r"] = np.nan
    tmp = df.query("is_file")
    scores = tmp["output_file"].apply(lambda x: np.nanmean(np.load(x))).values
    # scores = [np.nanmean(x) if x is not None else np.nan for x in scores]
    df.loc[df["is_file"].index, "avg_r"] = scores
    print(f"Averaged scores saved to {df_output_file}")
    df.to_csv(df_output_file)

    if delete_cache_after_run:
        shutil.rmtree(feature_dir)
        Path(cache_path).rmdir()
