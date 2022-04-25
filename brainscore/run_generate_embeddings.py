from pathlib import Path

import pandas as pd
import torch
from submitit import AutoExecutor

from .brain.data import get_stimulus, get_task_df
from .deep_net.data import get_activations


def compute_embeddings(
        model_file, task, output_file, max_len=1024, bidir=False):
    # With time window
    print(f"Computing the activations of {model_file} for task {task}")
    use_cuda = torch.cuda.is_available()
    stimulus = get_stimulus(task, lower=False)
    activations = get_activations(stimulus,
                                  model_name_or_path=model_file,
                                  max_len=max_len,
                                  bidir=bidir,
                                  device="cuda" if use_cuda else "cpu",
                                  )
    print(f"Savving activations of shape {activations.shape} to {output_file}")
    torch.save(activations, output_file)
    return True


def run_generate_embeddings(
        model_name_or_path,
        output_dir="/checkpoint/ccaucheteux/brainscore/cache",
        n_subjects=None,
        slurm_partition="learnfair",
        slurm_array_parallelism=200,
        local=True,
        model_max_len=1024,
        bidir=False,
        overwrite=False, device="cpu", select_tasks=None,):
    """
    TODO: cache instead of folder
    """

    assert n_subjects is None or n_subjects > 0
    subjects = get_task_df().subject.unique()[:n_subjects]
    tasks = get_task_df().query("subject in @subjects").audio_task.unique()
    if select_tasks is not None:
        tasks = [t for t in tasks if t in select_tasks]

    # Deep nets' activations
    output_dir = Path(output_dir)
    print(f"Computing deep networks activations to {output_dir}")
    output_dir.mkdir(exist_ok=True, parents=True)

    params = []
    for task in tasks:
        feature_file = output_dir / f"{task}.pth"
        params.append(
            dict(
                model_file=str(model_name_or_path),
                task=task,
                output_file=str(feature_file),
                max_len=model_max_len,
                bidir=bidir,
                to_run=overwrite or (not feature_file.is_file()),
            )
        )
    df = pd.DataFrame(params)
    df.to_csv(output_dir / "params.csv")

    df_to_run = df.query("to_run")

    if local:
        print(
            f"Computing activations in local for model {model_name_or_path} to {output_dir}")
        for params_ in params:
            if params_["to_run"]:
                del params_["to_run"]
                compute_embeddings(**params_)
        jobs = None
    elif len(df_to_run):
        print(f"{len(df_to_run)} jobs")

        name = "embeddings"
        executor = AutoExecutor(
            f"submitit_jobs/submitit_jobs/{name}")
        executor.update_parameters(
            slurm_partition=slurm_partition,
            slurm_array_parallelism=slurm_array_parallelism,
            timeout_min=60 * 72,
            # cpus_per_tasks=3,
            name=name,
            cpus_per_task=3,
            gpus_per_node=1 if device == "cuda" else 0,
        )

        keys = ["model_file", "task", "output_file", "max_len", "bidir"]

        jobs = executor.map_array(
            compute_embeddings, *[df_to_run[k].values for k in keys])
    out = {task: out_file for (task, out_file)
           in zip(df.task, df.output_file)}
    return out, jobs
