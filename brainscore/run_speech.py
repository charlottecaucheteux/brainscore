import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from submitit import AutoExecutor

from . import paths
from .brain.data import get_task_df
from .deep_net.data_speech import get_speech_activations
from .get_brain_score_speech import get_brain_score_speech


def _wait_until_complete(
        jobs, max_time_to_wait=50, wait_step=10 * 60, job_names=None):
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


def _job_compute_speech_activations(task, feature_type, output_file,
                                    hrf_model="glover",
                                    window=15,
                                    context=60):
    # With time window
    print("Computing the activations of Wav2Vec")
    wav_file = paths.stimuli / f"{task}_audio.wav"
    assert wav_file.is_file(), f"{wav_file} does not exist !!"
    activations, _ = get_speech_activations(
        wav_file,
        model_name_or_path="facebook/wav2vec2-base-960h",
        feature_type=feature_type,
        window=window,
        context=context,
        device="cpu",  # "cuda" if use_cuda else "cpu",
        TR=1.5,
        extra_scans=10,
        hrf_model="glover",
        flatten_hrf_cond=True,
    )
    print(f"Saving activations of shape {activations.shape} to {output_file}")
    torch.save(activations, output_file)
    return True


# STOP HERE UNTIL FINISH
def _job_compute_speech_brain_score(subject, feature_files, output_file,
                                    layers=None, to_rois=True, x_pca=0):
    score = get_brain_score_speech(
        feature_files,
        audio=True,
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


def run_eval_speech(output_path,
                    model_name="wav2vec2",
                    feature_type="conv",
                    layers=None,
                    average_bold=False,
                    hrf_model="glover",
                    n_subjects=10,
                    hemis=["L"],
                    to_rois=True,
                    x_pca=False,
                    cache_path="/checkpoint/ccaucheteux/cache",
                    delete_cache_after_run=True,
                    slurm_partition="learnfair",
                    local=False,
                    overwrite=False,
                    overwrite_feat=False,
                    ):

    if average_bold:
        print("Average subject")
        subjects = ["avg"]
        tasks = get_task_df().audio_task.unique()
    else:
        assert n_subjects is None or n_subjects > 0
        subjects = get_task_df().subject.unique()[:n_subjects]
        tasks = get_task_df().query("subject in @subjects").audio_task.unique()

    # --------- Deep nets' activations ---------
    feature_dir = Path(cache_path) / "embeddings" / \
        model_name / hrf_model / feature_type
    print(f"Computing deep networks activations to {feature_dir}")
    feature_dir.mkdir(exist_ok=True, parents=True)

    params = []
    for task in tasks:
        feature_file = feature_dir / f"{task}.pth"
        params.append(
            dict(
                task=task,
                feature_type=feature_type,
                output_file=str(feature_file),
                hrf_model=hrf_model,
                window=15,
                context=60,
                to_run=overwrite_feat or (not feature_file.is_file()),
            )
        )
    df = pd.DataFrame(params)
    df.to_csv(feature_dir / "params.csv")
    feature_files = {k: of for (k, of) in zip(
        df["task"], df["output_file"])}

    df_to_run = df.query("to_run")

    if local:
        for params_ in params:
            if params_["to_run"]:
                del params_["to_run"]
                _job_compute_speech_activations(**params_)
    elif len(df_to_run):
        print(f"{len(df_to_run)} jobs")

        name = "brainscore_embeddings"
        executor = AutoExecutor(
            f"submitit_jobs/submitit_jobs/{name}")
        executor.update_parameters(
            slurm_partition=slurm_partition,
            slurm_array_parallelism=100,
            timeout_min=60 * 72,
            # cpus_per_tasks=3,
            name=name,
            cpus_per_task=4,
            gpus_per_node=0,
        )

        keys = ["task", "feature_type", "output_file",
                "hrf_model",  "out_sr", "window", "context"]

        jobs = executor.map_array(
            _job_compute_speech_activations, *
            [df_to_run[k].values for k in keys])

        # Check jobs done
        completed = _wait_until_complete(
            jobs, max_time_to_wait=50, wait_step=10 * 60,
            job_names=df_to_run.task.values)
        print(
            f"Done computing deep networks activations, \
                {len(completed)}/{len(jobs)} jobs completed")

    # --------- Brain scores ---------
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
                    feature_type=feature_type,
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
            score = _job_compute_speech_brain_score(*[row[k] for k in keys])
            print(f"{score:.2f}")

    else:
        name = "brainscore_r"
        executor = AutoExecutor(
            f"submitit_jobs/submitit_jobs/{name}")
        executor.update_parameters(
            slurm_partition=slurm_partition,
            slurm_array_parallelism=200,
            timeout_min=60,
            # cpus_per_tasks=3,
            name=name,
            cpus_per_task=2,
            gpus_per_node=0,
        )

        jobs = executor.map_array(_job_compute_speech_brain_score,
                                  *[df_to_run[k].values for k in keys])
        # Check jobs done
        completed = _wait_until_complete(
            jobs, max_time_to_wait=50, wait_step=10 * 60,
            job_names=df_to_run["subject"].values)
        print(
            f"""Done computing deep networks activations, \
                {len(completed)}/{len(jobs)} jobs completed""")

    # Load average scores
    # df = pd.read_csv(df_output_file)
    # df["is_file"] = df["output_file"].apply(lambda x: Path(x).is_file())
    # df["avg_r"] = np.nan
    # tmp = df.query("is_file")
    # scores = tmp["output_file"].apply(lambda x: np.nanmean(np.load(x))).values
    # # scores = [np.nanmean(x) if x is not None else np.nan for x in scores]
    # df.loc[df.query("is_file").index, "avg_r"] = scores
    # print(f"Averaged scores saved to {df_output_file}")
    # df.to_csv(df_output_file)

    # if delete_cache_after_run:
    #     shutil.rmtree(feature_dir)
    #     Path(cache_path).rmdir()
