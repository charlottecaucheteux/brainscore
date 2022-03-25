from pathlib import Path

import pandas as pd
import torch
from submitit import AutoExecutor

from brainscore import paths
from brainscore.brain.data import get_task_df
from brainscore.deep_net.data_speech import get_speech_activations

SELECTED_TASKS = []
LOCAL = True
FEATURE_FOLDER = "fairseq_0315"
FEATURE_FOLDER = "fairseq_0318_st5_ct10"
SLURM_PARTITION = "learnlab"
DEVICE = "cuda"
OVERWRITE = False

WINDOW = 5
CONTEXT = 10

FEATURE_TYPES = ["tr", "conv"]
SCALE_TYPES = ["minmax"]
MODEL_NAMES = ['checkpoint_unsup_english',
               'checkpoint_sup_english',
               'random_model',
               'checkpoint_unsup_ac_scenes',
               'checkpoint_finetuned_english',
               'checkpoint_unsup_mandarin',
               'checkpoint_unsup_dutch',
               'checkpoint_unsup_french']

SUPERVISED = [("_sup_" in k) or ("_finetuned_" in k) for k in MODEL_NAMES]
assert len(SUPERVISED) == len(MODEL_NAMES)


def _job_compute_speech_activations(model_name, task, feature_type, output_file,
                                    supervised=True,
                                    hrf_model="glover",
                                    window=15,
                                    context=60,
                                    scale="minmax",
                                    ):
    # With time window
    print(f"Computing the activations of Wav2Vec to {output_file}")

    wav_file = paths.stimuli / f"{task}_audio.wav"
    assert wav_file.is_file(), f"{wav_file} does not exist !!"
    fname = paths.fairseq_models / f"{model_name}.pt"
    activations, _ = get_speech_activations(
        wav_file,
        model_name_or_path=fname,
        feature_type=feature_type,
        window=window,
        context=context,
        device=DEVICE,  # "cuda" if use_cuda else "cpu",
        TR=1.5,
        extra_scans=10,
        hrf_model="glover",
        flatten_hrf_cond=True,
        scale="minmax",
        fairseq=True,
        supervised=supervised,
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
    for scaling in SCALE_TYPES:
        for feature_type in FEATURE_TYPES:
            for model_name, supervised in zip(MODEL_NAMES, SUPERVISED):
                for task in tasks:
                    label = model_name.replace("checkpoint_", "")
                    feature_file = feature_dir / \
                        f"{task}_{label}_{feature_type}_{scaling}.pth"
                    params.append(
                        dict(
                            model_name=model_name,
                            task=task,
                            feature_type=feature_type,
                            output_file=str(feature_file),
                            supervised=supervised,
                            hrf_model="glover",
                            window=WINDOW,
                            context=CONTEXT,
                            scale=scaling,
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
                _job_compute_speech_activations(**params_)

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

        keys = ["model_name", "task", "feature_type", "output_file",
                "supervised", "hrf_model", "window", "context", "scale"]

        jobs = executor.map_array(
            _job_compute_speech_activations, *
            [df_to_run[k].values for k in keys])

        import pdb
        pdb.set_trace()
