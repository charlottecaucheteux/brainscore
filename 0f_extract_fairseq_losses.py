from pathlib import Path

import pandas as pd
import torch
from submitit import AutoExecutor

from brainscore import paths
from brainscore.brain.data import get_task_df
from brainscore.deep_net.data_speech import get_speech_activations

SELECTED_TASKS = []
LOCAL = True
FEATURE_FOLDER = "fairseq_losses_0429"
SLURM_PARTITION = "devlab"
DEVICE = "cpu"
OVERWRITE = False

WINDOW = .05
CONTEXT = 10


def _job_compute_loss_speech_activations(task, output_file,
                                         window=.1,
                                         context=10,
                                         ):
    # With time window
    print(f"Computing the activations of Wav2Vec to {output_file}")

    wav_file = paths.stimuli / f"{task}_audio.wav"
    assert wav_file.is_file(), f"{wav_file} does not exist !!"
    fname = paths.fairseq_models / f"checkpoint_unsup_english.pt"
    activations, _ = get_speech_activations(
        wav_file,
        model_name_or_path=str(fname),
        window=window,
        context=context,
        device=DEVICE,  # "cuda" if use_cuda else "cpu",
        TR=1.5,
        extra_scans=10,
        hrf_model="glover",
        flatten_hrf_cond=True,
        scale="minmax",
        fairseq=False,
        loss=True,
        fairseq_to_hugg=True,
    )
    print(f"Saving activations of shape {activations.shape} to {output_file}")
    torch.save(activations, output_file)
    return True


if __name__ == "__main__":

    tasks = get_task_df()
    if SELECTED_TASKS is not None and len(SELECTED_TASKS):
        tasks = tasks.query("audio_task in @SELECTED_TASKS")
    tasks = tasks.audio_task.unique()

    # --------- Deep nets' activations ---------
    feature_dir = paths.speech_embeddings / FEATURE_FOLDER
    print(f"Computing deep networks activations to {feature_dir}")
    feature_dir.mkdir(exist_ok=True, parents=True)

    params = []
    for task in tasks:
        feature_file = feature_dir / \
            f"{task}_fairseq_unsup_english_loss.pth"
        params.append(
            dict(
                task=task,
                output_file=str(feature_file),
                window=WINDOW,
                context=CONTEXT,
                to_run=OVERWRITE or (
                    not feature_file.is_file()),
            )
        )
    df = pd.DataFrame(params)
    df.to_csv(feature_dir / "params.csv")
    feature_files = {k: of for (k, of) in zip(
        df["task"], df["output_file"])}

    df_to_run = df.query("to_run")

    if LOCAL:
        for params_ in params:
            if params_["to_run"]:
                del params_["to_run"]
                _job_compute_loss_speech_activations(**params_)

    elif len(df_to_run):
        print(f"{len(df_to_run)} jobs")

        name = "wav2vec_embeddings"
        executor = AutoExecutor(
            f"submitit_jobs/submitit_jobs/{name}")
        executor.update_parameters(
            slurm_partition=SLURM_PARTITION,
            slurm_array_parallelism=220,
            timeout_min=60 * 72,
            # cpus_per_tasks=3,
            name=name,
            cpus_per_task=3,
            gpus_per_node=1 if DEVICE == "cuda" else 0,
        )

        keys = ["task",  "output_file", "window", "context"]

        jobs = executor.map_array(
            _job_compute_loss_speech_activations, *
            [df_to_run[k].values for k in keys])

        import pdb
        pdb.set_trace()
