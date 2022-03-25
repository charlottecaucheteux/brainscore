"""
Get activations form Wav2Vec model (huggingface)
"""
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from nistats import hemodynamic_models
from tqdm import tqdm

VALID_N_CONDS = {
    "glover": 1,
    "glover + derivative": 2,
    "glover + derivative + dispersion": 3,
    "spm": 1,
    "spm + derivative": 2,
    "spm + derivative + dispersion": 3,
}
logger = logging.getLogger(__name__)


def extract_tr(model, input_values, sampling_rate=16000, window=15, context=60):
    """
    Extract Transformer activations using a rolling window
    window: cut inputs in shares of `window` seconds
    context: feed the networks with `context` seconds of context
    """

    # Init params
    assert window <= context
    window_len = window * sampling_rate
    context_len = context * sampling_rate
    story_len = input_values.size(-1)
    assert story_len >= context_len
    n_steps = story_len // window_len + 1

    # Build context inputs, with window at the end
    context_input = [
        torch.roll(input_values, -(k + 1) * window_len, dims=-1)[:, -context_len:]
        for k in range(n_steps)
    ]

    # Extract activations
    with torch.no_grad():
        context_output = []
        for k, x in enumerate(tqdm(context_input)):
            outs = model(x.to(model.device), output_hidden_states=True)
            # Batch size of one here
            outs = torch.stack([out.to("cpu")[0] for out in outs.hidden_states])

            # Extract window activations at the end
            seq_len = outs.size(1)
            cut = int(np.round(seq_len * (window_len / context_len)))
            outs = outs[:, -cut:]

            context_output.append(outs)

        # Concatenate each window activations
        context_output = torch.cat(context_output, dim=1)

    # Remove extra samples
    out_sr = context_output.size(1) / (n_steps * window_len / sampling_rate)
    cut = int(story_len / sampling_rate * out_sr)
    output = context_output[:, :cut]  # [L, B, To, D]
    output = output.transpose(1, 2)  # [L, D, To]
    return list(output)


def extract_conv(model, input_values):
    outs = []
    hidden_states = input_values[:, None].to(model.device)
    for conv_layer in model.feature_extractor.conv_layers:
        hidden_states = conv_layer(hidden_states)
        print(hidden_states.shape)
        outs.append(hidden_states.cpu()[0])  # Only batch size of 1 here
    return list(outs)


def apply_hrf(features, frame_times, sampling_rate, hrf_model="glover"):
    n_time = features.size(-1)
    n_scans = len(frame_times)
    n_cond = VALID_N_CONDS[hrf_model]

    # Setup HRF inputs that are stable
    onsets = np.arange(n_time) * sampling_rate
    durations = np.repeat(1 / sampling_rate, n_time)

    # Apply HRF model for each feature
    flatten = features.reshape(-1, n_time).numpy()
    hrf_out = np.zeros((len(flatten), n_scans, n_cond))
    logger.info(f"Applying HRF")
    for k, amplitudes in enumerate(tqdm(flatten)):
        exp_condition = np.stack([onsets, durations, amplitudes])
        signal, _ = hemodynamic_models.compute_regressor(
            exp_condition, hrf_model, frame_times
        )
        hrf_out[k] = signal
    hrf_out = hrf_out.reshape((*features.shape[:-1], n_scans, n_cond))
    hrf_out = torch.Tensor(hrf_out)
    return hrf_out


def apply_hrf_to_list(
        features, total_duration, scale="minmax", TR=1.5, extra_scans=10,
        hrf_model="glover"):

    n_scans = int(total_duration // TR + extra_scans)
    frame_times = np.arange(n_scans) * TR
    outputs = []

    for feat in features:
        n_time = feat.size(-1)

        # Scale
        feat = scale_features(feat, scale=scale)

        # To numpy
        feat = feat.numpy()

        # Setup HRF params
        onsets = np.linspace(0, total_duration, n_time)
        durations = np.repeat(total_duration / n_time, n_time)

        # Apply HRF model for each feature
        flatten = feat.reshape(-1, n_time)
        hrf_out = np.zeros((len(flatten), n_scans))
        logger.info(f"Applying HRF")
        for k, amplitudes in enumerate(tqdm(flatten)):
            exp_condition = np.stack([onsets, durations, amplitudes])
            signal, _ = hemodynamic_models.compute_regressor(
                exp_condition, hrf_model, frame_times
            )
            hrf_out[k] = signal.squeeze()
        hrf_out = hrf_out.reshape((*feat.shape[:-1], n_scans))
        hrf_out = torch.Tensor(hrf_out)

        outputs.append(hrf_out)

    # Back to torch
    outputs = torch.stack(outputs)

    return outputs


def scale_features(features, scale="minmax"):
    """
    Features of shape [L, D, T]
    """
    assert scale in ["minmax", "normal", "none"]
    if scale == "minmax":
        features -= features.min(-1, keepdim=True)[0]
        a = features.max(-1, keepdim=True)[0]
        a[a == 0.0] = 1.0
        features /= a
    elif scale == "normal":
        features -= features.mean(-1, keepdim=True)
        features /= features.std(-1, keepdim=True)
    return features


def get_speech_activations(
    wav_file,
    model_name_or_path="facebook/wav2vec2-base-960h",
    feature_type="tr",
    window=15,
    context=30,
    device="cpu",
    TR=1.5,
    extra_scans=10,
    hrf_model="glover",
    flatten_hrf_cond=True,
    scale="minmax",
    pretrained=True,
    supervised=False,
    fairseq=False,
):
    """
    From wav file to
    [L, B, D, N, C]
    with
         - L number of features
         - B batch size
         - D dimensionality, 768 or 512
         - N number of scans (one every TR + extra_scans)
         - C number of conditions in HRF model
    """
    assert feature_type in ["tr", "conv"]

    if fairseq:
        from .extract_fairseq_activations import EmbeddingDatasetWriter

        fname = Path(model_name_or_path)
        fairseq_model = EmbeddingDatasetWriter(
            # folder to checkpoint
            model_folder=str(fname.parent),
            model_fname=fname.name,  # path to the checkpoint
            gpu=0,  # do not change this
            # True if you are using a fine tuned model or a supervised model, False otherwise
            asr=supervised,
        )
        model_sr = 16000
    else:
        from transformers import AutoConfig, Wav2Vec2Model, Wav2Vec2Processor

        # Load model
        processor = Wav2Vec2Processor.from_pretrained(model_name_or_path)
        model_sr = processor.feature_extractor.sampling_rate
        if pretrained:
            model = Wav2Vec2Model.from_pretrained(model_name_or_path)
        else:
            logger.info("Starting from scratch")
            print("Starting from scratch")
            config = AutoConfig.from_pretrained(model_name_or_path)
            model = Wav2Vec2Model(config)

    # Load wav
    wav, wav_sr = sf.read(wav_file)

    # Average channels
    if wav.ndim > 1:
        assert wav.ndim == 2
        wav = wav.mean(-1)
        print(f"wav shape {wav.shape}")

    # Upsample to model sampling rate
    total_duration = wav.size / wav_sr
    wav = torch.Tensor(wav)[None, None]
    wav = F.interpolate(wav, int(total_duration * model_sr))
    wav = wav.squeeze()

    if fairseq:
        features = fairseq_model.transform_wav(
            wav.numpy(),
            window_stride=window,
            context_size=context,
            feature_type=feature_type,
        )
        feature_names = [f"{feature_type}_{i}" for i in range(len(features))]

    else:
        # Process inputs
        inputs = processor(wav, sampling_rate=model_sr, return_tensors="pt")

        # To device
        model = model.to(device)

        # Extract Wav2Vec features
        logger.info(
            f"Extracting {feature_type} features with window {window}s and context {context}s"
        )
        assert feature_type in ["conv", "tr"]
        with torch.no_grad():
            input_values = inputs.input_values
            if feature_type == "conv":
                features = extract_conv(model, input_values)
            else:
                features = extract_tr(
                    model,
                    input_values,
                    sampling_rate=model_sr,
                    window=window,
                    context=context,
                )
            feature_names = [f"{feature_type}_{i}" for i in range(
                len(features))]

    hrf_out = apply_hrf_to_list(
        features, total_duration, scale=scale, TR=TR, extra_scans=extra_scans
    )

    # Transpose --> [L, N, D, C]
    hrf_out = hrf_out.transpose(1, 2)

    # Flatten --> [L, N, D*C]
    if flatten_hrf_cond:
        hrf_out = hrf_out.reshape((*hrf_out.shape[:2], -1))
    return hrf_out, feature_names


if __name__ == "__main__":
    task = "pieman"
    wav_file = f"/datasets01/hasson_narratives/{task}_audio.wav"
    for supervised in [True, False]:
        for feature_type in ["tr", "conv"]:
            res, _ = get_speech_activations(
                wav_file,
                model_name_or_path="facebook/wav2vec2-base-960h",
                feature_type=feature_type,
                fairseq=True,
                supervised=supervised,
            )
            print(supervised, feature_type, res.shape)
