from pathlib import Path

import numpy as np
import torch

from brainscore import paths
from brainscore.brain.data import get_task_df
from brainscore.deep_net.data_speech import get_speech_activations
from brainscore.get_brain_score_speech import get_brain_score_speech


def _job_compute_speech_activations(task, output_file, feature_type="tr",
                                    hrf_model="glover",
                                    window=5,
                                    context=10,
                                    scale="minmax",
                                    pretrained=True,
                                    model_name="wav2vec2-base-960h",
                                    device="cpu",
                                    overwrite=False,
                                    ):
    # With time window
    print(f"Computing the activations of Wav2Vec to {output_file}")
    if output_file.exists() and not overwrite:
        return
    # use_cuda = torch.cuda.is_available()
    wav_file = paths.stimuli / f"{task}_audio.wav"
    assert wav_file.is_file(), f"{wav_file} does not exist !!"
    activations, _ = get_speech_activations(
        wav_file,
        model_name_or_path=f"facebook/{model_name}",
        feature_type=feature_type,  # either tr or conv
        window=window,  # stride
        context=context,  # context size
        pretrained=pretrained,  # whether to start from scratch or use pretrained
        device=device,  # "cuda" if use_cuda else "cpu",
        # Preprocessing params
        TR=1.5,  # HRF
        extra_scans=10,  # HRF
        hrf_model="glover",  # HRF
        scale="minmax",
    )
    print(f"Saving activations of shape {activations.shape} to {output_file}")
    torch.save(activations, output_file)


def _job_compute_speech_brain_score(feature_files,
                                    output_file,
                                    subject="avg",
                                    select_tasks=["pieman"],
                                    hemi="L",
                                    layers=None, to_rois=False, x_pca=0):
    score = get_brain_score_speech(
        feature_files,
        subject=subject,
        # X
        layers=layers,  # selected embedding layers
        concat_layers=False,  # whether to concatenate layers of run for each layer
        x_pca=x_pca,  # whether to apply pca on embeddings
        # Y
        rois=to_rois,  # whether to compute scores on brain ROIS
        hemi=hemi,
        y_pca=0,
        select_tasks=select_tasks,  # None or subselected selected audio tasks
        # Model
        metric="correlate",
        n_folds=20 if subject == "avg" else 5,
        average_folds=(subject != "avg"),
    )
    print(f"Saving score to {output_file}")
    Path(output_file).parent.mkdir(exist_ok=True, parents=True)
    np.save(output_file, score)
    return score


if __name__ == "__main__":

    SELECT_TASKS = ["pieman"]
    SELECT_TASKS = []

    # ---- Select audio tasks -----
    if len(SELECT_TASKS):
        tasks = SELECT_TASKS.copy()
    else:
        tasks = get_task_df().audio_task.unique()  # for all tasks

    # ---- Compute embeddings for each task ----
    embed_files = {}
    for task in tasks:
        feature_type = 'conv'
        embed_file = paths.speech_embeddings / \
            "minimal" / f"{task}_{feature_type}.pth"
        embed_file.parent.mkdir(exist_ok=True, parents=True)
        _job_compute_speech_activations(
            task, embed_file, feature_type=feature_type)  # either "tr" or "conv"
        embed_files[task] = embed_file

    # ------ Compute brain scores for average subject (left hemi) ----
    output_file = paths.scores / "minimal" / "avg_L.npy"
    score = _job_compute_speech_brain_score(embed_files,
                                            output_file,
                                            layers=(0, 8),
                                            to_rois=True,
                                            subject="avg",
                                            select_tasks=tasks)
