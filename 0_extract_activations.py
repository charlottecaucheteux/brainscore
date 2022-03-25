
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from submitit import AutoExecutor

from brainscore import paths
from brainscore.brain.data import get_stimulus, get_task_df
from brainscore.deep_net.data_speech import get_speech_activations
from brainscore.get_brain_score_speech import get_brain_score_speech

SELECTED_TASKS = []
LOCAL = True
FEATURE_FOLDER = "wav2vec2_0304"
SLURM_PARTITION = "learnlab"
DEVICE = "cpu"
OVERWRITE = False

FEATURE_TYPES = ["tr", "conv"]
SCALE_TYPES = ["minmax"]
PRETRAINS = [False]

# Unchanged
HRF_LABELS = {
    "glov": "glover",
    "glov2": 'glover + derivative',
    'glov3': 'glover + derivative + dispersion',
    'spm': 'spm',
    'spm2': 'spm + derivative',
    'spm3': 'spm + derivative + dispersion',
}


def _job_compute_speech_activations(task, feature_type, output_file,
                                    hrf_model="glover",
                                    window=15,
                                    context=60,
                                    scale="minmax",
                                    pretrained=True,
                                    model_name="wav2vec2-base-960h",
                                    ):
    # With time window
    print(f"Computing the activations of Wav2Vec to {output_file}")
    # use_cuda = torch.cuda.is_available()
    wav_file = paths.stimuli / f"{task}_audio.wav"
    assert wav_file.is_file(), f"{wav_file} does not exist !!"
    activations, _ = get_speech_activations(
        wav_file,
        model_name_or_path=f"facebook/{model_name}",
        feature_type=feature_type,
        window=window,
        context=context,
        device=DEVICE,  # "cuda" if use_cuda else "cpu",
        TR=1.5,
        extra_scans=10,
        hrf_model="glover",
        flatten_hrf_cond=True,
        scale="minmax",
        pretrained=pretrained,
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
    for pretrain in PRETRAINS:
        for scaling in SCALE_TYPES:
            for feature_type in FEATURE_TYPES:
                for task in tasks:
                    if not pretrain:
                        ext = "_scratch"
                    else:
                        ext = ""
                    feature_file = feature_dir / \
                        f"{task}_{feature_type}_{scaling}{ext}.pth"
                    params.append(
                        dict(
                            task=task,
                            feature_type=feature_type,
                            output_file=str(feature_file),
                            hrf_model="glover",
                            out_sr=16000,
                            window=5,
                            context=30,
                            scale=scaling,
                            pretrained=pretrain,
                            to_run=OVERWRITE or (not feature_file.is_file()),
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
                _job_compute_speech_activations(**params_)

    elif len(df_to_run):
        print(f"{len(df_to_run)} jobs")

        name = "wav2vec_embeddings"
        executor = AutoExecutor(
            f"submitit_jobs/submitit_jobs/{name}")
        executor.update_parameters(
            slurm_partition=SLURM_PARTITION,
            slurm_array_parallelism=200,
            timeout_min=60 * 72,
            # cpus_per_tasks=3,
            name=name,
            cpus_per_task=3,
            gpus_per_node=1 if DEVICE == "cuda" else 0,
        )

        keys = ["task", "feature_type", "output_file",
                "hrf_model",  "out_sr", "window", "context", "scale", "pretrained"]

        jobs = executor.map_array(
            _job_compute_speech_activations, *[df_to_run[k].values for k in keys])

        import pdb
        pdb.set_trace()
