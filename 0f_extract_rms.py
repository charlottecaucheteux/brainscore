from pathlib import Path

import librosa
import pandas as pd
import torch
from submitit import AutoExecutor

from brainscore import paths
from brainscore.brain.data import get_task_df
from brainscore.deep_net.data_speech import apply_hrf_to_list

SELECTED_TASKS = []
LOCAL = True
FEATURE_FOLDER = "rms_0504"
SLURM_PARTITION = "learnlab"
DEVICE = "cpu"
OVERWRITE = False


def _job_compute_rms(task, output_file):
    # With time window
    print(f"Computing the activations of Wav2Vec to {output_file}")

    wav_file = paths.stimuli / f"{task}_audio.wav"
    assert wav_file.is_file(), f"{wav_file} does not exist !!"
    # tu load ton filename en utilisant librosa
    wav, wav_sr = librosa.load(wav_file, sr=None)
    total_duration = wav.size / wav_sr

    feature = librosa.feature.rms(
        y=wav, frame_length=int(0.025 * wav_sr), hop_length=int(0.010 * wav_sr)
    )
    feature = torch.Tensor(feature)

    # each feature [D, T]
    hrf_out = apply_hrf_to_list(
        [torch.Tensor(feature)],
        total_duration, scale="minmax", TR=1.5, extra_scans=10,
        hrf_model="glover",)

    # Transpose --> [L, N, D, C]
    hrf_out = hrf_out.transpose(1, 2)

    # Flatten --> [L, N, D*C]
    hrf_out = hrf_out.reshape((*hrf_out.shape[:2], -1))

    print(f"Saving activations of shape {hrf_out.shape} to {output_file}")
    torch.save(hrf_out, output_file)
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
            f"{task}_rms.pth"
        params.append(
            dict(
                task=task,
                output_file=str(feature_file),
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
                _job_compute_rms(**params_)

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

        keys = ["task",  "output_file"]

        jobs = executor.map_array(
            _job_compute_rms, *
            [df_to_run[k].values for k in keys])

        import pdb
        pdb.set_trace()
