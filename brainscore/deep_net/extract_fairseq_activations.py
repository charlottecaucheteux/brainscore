import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm


def forward_for_conv_encoder(conv_encoder, x, nb=10000):

    # BxT -> BxCxT
    count = 0
    x = x.unsqueeze(1)

    for conv in conv_encoder.conv_layers:
        x = conv(x)
        if count == nb:
            return x
        count += 1

    return x


def forward_for_model_wav2vec2(
    model, source, padding_mask=None, mask=True, features_only="z"
):
    """This is a copy of the wav2vec2 forward function, modified to output feature ate different level
    model: the wav2vec2 model
    source: the source as input
    features_only : the feature you want as output, what is available : conv_0, conv_1, conv_2, conv_3, conv_4
    conv_5, conv_6, z, q, c, transf_0 to 11  warnings : q does not work for fine tuned models"""

    if model.feature_grad_mult > 0:
        if not "conv" in features_only:
            features = forward_for_conv_encoder(model.feature_extractor, source)
        else:
            features = forward_for_conv_encoder(
                model.feature_extractor, source,
                nb=int(features_only.split("_")[1]))
        if model.feature_grad_mult != 1.0:
            from fairseq.modules import GradMultiply

            features = GradMultiply.apply(features, model.feature_grad_mult)
    else:
        with torch.no_grad():
            if not "conv" in features_only:
                features = forward_for_conv_encoder(
                    model.feature_extractor, source)
            else:
                features = forward_for_conv_encoder(
                    model.feature_extractor, source,
                    nb=int(features_only.split("_")[1]))
    if "conv" in features_only:
        return {"x": features}

    features_pen = features.float().pow(2).mean()

    features = features.transpose(1, 2)
    features = model.layer_norm(features)
    unmasked_features = features.clone()

    if padding_mask is not None:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(
            padding_mask.size(0), features.size(1), -1)
        padding_mask = padding_mask.all(-1)

    if model.post_extract_proj is not None:
        features = model.post_extract_proj(features)

    features = model.dropout_input(features)
    unmasked_features = model.dropout_features(unmasked_features)

    num_vars = None
    code_ppl = None
    prob_ppl = None
    curr_temp = None

    if model.input_quantizer:
        q = model.input_quantizer(features, produce_targets=False)
        features = q["x"]
        num_vars = q["num_vars"]
        code_ppl = q["code_perplexity"]
        prob_ppl = q["prob_perplexity"]
        curr_temp = q["temp"]
        features = model.project_inp(features)

    if mask:
        x, mask_indices = model.apply_mask(features, padding_mask)
        if mask_indices is not None:
            y = unmasked_features[mask_indices].view(
                unmasked_features.size(0), -1, unmasked_features.size(-1)
            )
        else:
            y = unmasked_features
    else:
        x = features
        y = unmasked_features
        mask_indices = None

    if features_only == "z":
        return {"x": x, "padding_mask": padding_mask}

    if "transf" in features_only:
        block_nb = features_only.split("_")[-1]
        x, layer_results = model.encoder(
            x, padding_mask=padding_mask, layer=int(block_nb)
        )
        return {"x": x, "padding_mask": padding_mask}

    x = model.encoder(x, padding_mask=padding_mask)

    if features_only == "z_after_encoder":
        return {"x": x, "padding_mask": padding_mask}

    if model.quantizer:
        q = model.quantizer(y, produce_targets=False)
        y = q["x"]
        num_vars = q["num_vars"]
        code_ppl = q["code_perplexity"]
        prob_ppl = q["prob_perplexity"]
        curr_temp = q["temp"]

        y = model.project_q(y)

        if model.negatives_from_everywhere:
            neg_cands, *_ = model.quantizer(unmasked_features,
                                            produce_targets=False)
            negs, _ = model.sample_negatives(neg_cands, y.size(1))
            negs = model.project_q(negs)

        else:
            negs, _ = model.sample_negatives(y, y.size(1))

        if model.codebook_negatives > 0:
            cb_negs = model.quantizer.sample_from_codebook(
                y.size(0) * y.size(1), model.codebook_negatives
            )
            cb_negs = cb_negs.view(
                model.codebook_negatives, y.size(0), y.size(1), -1
            )  # order doesnt matter
            cb_negs = model.project_q(cb_negs)
            negs = torch.cat([negs, cb_negs], dim=0)
    elif model.project_q:
        y = model.project_q(y)

        if model.negatives_from_everywhere:
            negs, _ = model.sample_negatives(unmasked_features, y.size(1))
            negs = model.project_q(negs)
        else:
            negs, _ = model.sample_negatives(y, y.size(1))
    if features_only == "q":
        return {"x": y}

    if features_only == "c":
        x = model.final_proj(x[0])
        return {"x": x}

    x = x[mask_indices].view(x.size(0), -1, x.size(-1))

    if model.target_glu:
        y = model.target_glu(y)
        negs = model.target_glu(negs)

    x = model.final_proj(x)
    x = model.compute_preds(x, y, negs)

    result = {"x": x, "padding_mask": padding_mask,
              "features_pen": features_pen}

    if prob_ppl is not None:
        result["prob_perplexity"] = prob_ppl
        result["code_perplexity"] = code_ppl
        result["num_vars"] = num_vars
        result["temp"] = curr_temp

    return result


def read_audio(fname):
    """Load an audio file and return PCM along with the sample rate"""

    wav, sr = sf.read(fname)
    assert sr == 16e3

    return wav, 16e3


class PretrainedWav2Vec2Model(nn.Module):
    def __init__(self, folder, fname, asr=False):
        super().__init__()
        self.asr = asr  # this option needs to be true for fine tuned models
        print("formatting model")

        from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Model

        model = Wav2Vec2Model.from_pretrained(folder, fname).models
        model = model[0]
        model.eval()
        if self.asr:
            self.model_enc_all = model.w2v_encoder
            self.model = model.w2v_encoder.w2v_model
        else:
            self.model = model

    def forward(self, x, feat_only):
        with torch.no_grad():
            # print('in model', x.shape)
            if not self.asr:
                z = forward_for_model_wav2vec2(
                    self.model, x, mask=False, features_only=feat_only
                )
            else:
                if feat_only != "c":
                    z = forward_for_model_wav2vec2(
                        self.model, x, mask=False, features_only=feat_only
                    )
                else:
                    # we use the whole original function
                    z = self.model_enc_all.forward(x, padding_mask=None)
                    z["x"] = z["encoder_out"]
        return z["x"]


class Prediction:
    """Lightweight wrapper around a fairspeech embedding model"""

    def __init__(self, folder, fname, gpu=0, asr=False):
        print("initialising")
        self.gpu = gpu
        self.model = PretrainedWav2Vec2Model(folder, fname, asr=asr).cuda(gpu)

    def __call__(self, x, feat_only):
        x = torch.from_numpy(x).float().cuda(self.gpu)
        with torch.no_grad():
            # print('x shape', x.shape)
            # print('x uns', x.unsqueeze(0).shape)
            z = self.model(x.unsqueeze(0), feat_only)
        return z.squeeze(0).cpu().numpy()


class EmbeddingDatasetWriter(object):
    """Given a model and a wav2letter++ dataset, pre-compute and store embeddings"""

    def __init__(
        self,
        model_folder,  # the folder of your checkpoint
        model_fname,  # the complete path to your checkpoint
        gpu=0,
        verbose=False,
        asr=False,  # True if you used a fine tuned model
    ):
        assert (Path(model_folder) / model_fname).is_file()

        self.model_fname = model_fname
        self.model_folder = model_folder
        self.model = Prediction(
            self.model_folder, self.model_fname, asr=asr, gpu=gpu)
        self.verbose = verbose

    def _progress(self, iterable, **kwargs):
        if self.verbose:
            return tqdm.tqdm(iterable, **kwargs)
        return iterable

    def transform_wav(
            self, wav, window_stride, context_size, feature_type, sr=16000):
        """
        transform a wavfile using a pretrained model
        :param window_stride: size of the stride used for the transformation
        :param context_size: size of the context windows used for the transformation
        :param feature_type: "tr" or "conv"
        :return: a list of torch tensor, each element is the torch tensor production of one layer, each tensor is of shape [D,T] (D dimension, T time)
        """

        list_to_return = []

        size_win = context_size * sr
        stride_win = window_stride * sr

        print("size wav in second:", len(wav) / sr)

        size_wav_cut = len(wav) - size_win
        nb_cuts = int(size_wav_cut // stride_win)

        print("size_wav_cut", size_wav_cut, "nb_cuts", nb_cuts)

        if feature_type == "tr":
            features = ["transf_" + str(i) for i in range(12)]
        elif feature_type == "conv":
            features = ["conv_" + str(i) for i in range(7)]

        for feat_used in features:
            for i in range(nb_cuts):
                wav_to_transform = wav[
                    int(stride_win * i): int(stride_win * i + size_win)
                ]
                feat = self.model(wav_to_transform, feat_used)
                if "conv" in feat_used:  # we get time in first dimension
                    feat = feat.swapaxes(0, 1)

                if i == 0:
                    size_equival_win = int(feat.shape[0])
                    size_equival_stride = int(
                        size_equival_win * (stride_win / size_win)
                    )
                    size_all = int(size_equival_win * len(wav) / size_win)
                    print("size_equivalent_win", size_win / sr,
                          "s is", size_equival_win)
                    nb_times = np.asarray([0.0 for i in range(size_all)])
                    size_feat = feat.shape[1]
                    feat_all = np.zeros((size_all, size_feat))

                feat_all[
                    int(i * size_equival_stride): int(
                        i * size_equival_stride + size_equival_win
                    ),
                    :,
                ] += feat
                nb_times[
                    int(i * size_equival_stride): int(
                        i * size_equival_stride + size_equival_win
                    )
                ] += 1.0
                final = i

            # We get what is missing
            wav_to_transform = wav[int(stride_win * final + size_win):]
            feat = self.model(wav_to_transform, feat_used)
            if "conv" in feat_used:  # we get time in first dimension
                feat = feat.swapaxes(0, 1)
            feat_all[
                int(final * size_equival_stride + size_equival_win): int(
                    final * size_equival_stride + size_equival_win + feat.shape[0]
                ),
                :,
            ] += feat
            nb_times[
                int(final * size_equival_stride + size_equival_win): int(
                    final * size_equival_stride + size_equival_win + feat.shape[0]
                )
            ] += 1

            # get where the zeros are
            limit = 0
            for i in range(nb_times.shape[0]):
                if nb_times[i] == 0:
                    limit = i
                    break

            # now we average over strides
            divider = np.asarray([nb_times for i in range(size_feat)])
            divider = divider.swapaxes(0, 1)
            # dim = Time x Dimensions
            feat_all = feat_all[:limit] / divider[:limit]
            feat_all = feat_all.swapaxes(0, 1)  # dim = Dimension x Time
            # conversion to torch tensor
            feat_all = torch.from_numpy(feat_all).float()
            list_to_return.append(feat_all)  # we add it to the list

        return list_to_return

    def __repr__(self):

        return "EmbeddingDatasetWriter ({n_files} files)\n\tinput:\t{input_root}\n\toutput:\t{output_root}\n\tsplit:\t{split})".format(
            n_files=len(self), **self.__dict__
        )


def extract_fairseq_activations(
    wav_file,
    model_name_or_path="facebook/wav2vec2-base-960h.pt",
    feature_type="tr",  # or conv
    window=15,  # stride
    context=30,
    device="cpu",
    supervised=False,
):
    """
    wav_file: audio file, sampled at 16kHz
    model_name_or_path: path to model, need to be complete (example path/path/checkpoint.pt)
    feature_type: "tr" to extract transformer layers, "conv" to extract conv layers
    window: (or stride), splits the inputs in shares of `window` seconds
    context: context size, in seconds
    device: "cpu" or "cuda" # I do not use this, feel free to modify the code to use it
    supervised: True if you are using a supervised or finetuned model, False otherwise

    Returns list of L torch tensors of shape [D, T]
    with:
        - L the number of layers
        - D the dimensionality (e.g. 512 for conv, 768 for tr)
        - T the sequence length. T corresponds to a subsampling of the total_duration of the audio wav_file.
    """

    # Load wav
    wav, wav_sr = sf.read(wav_file)

    # Average channels
    if wav.ndim > 1:
        assert wav.ndim == 2
        wav = wav.mean(-1)
        print(f"wav shape {wav.shape}")

    # Upsample to model sampling rate
    model_sr = 16000
    total_duration = wav.size / wav_sr
    wav = torch.Tensor(wav)[None, None]
    wav = F.interpolate(wav, int(total_duration * model_sr)).squeeze()

    writer = EmbeddingDatasetWriter(
        model_folder=Path(model_name_or_path).parent,  # folder to checkpoint
        model_fname=model_name_or_path,  # path to the checkpoint
        gpu=0,  # do not change this
        # True if you are using a fine tuned model or a supervised model, False otherwise
        asr=supervised,
    )
    return writer.transform_wav(
        wav.numpy(),
        window_stride=window,
        context_size=context,
        feature_type=feature_type,
    )


if __name__ == "__main__":
    folder = "models/fairseq/fairseq_models_juliette/"
    fname = "checkpoint_unsup_english.pt"
    fname = "random_model.pt"
    model = EmbeddingDatasetWriter(
        model_folder=folder,  # folder to checkpoint
        model_fname=fname,  # path to the checkpoint
        gpu=0,  # do not change this
        # True if you are using a fine tuned model or a supervised model, False otherwise
        asr=False,
    )
    for supervised in [True, False]:
        for feature_type in ["tr", "conv"]:
            res = model.transform_wav(
                torch.rand(1000000).numpy(),
                window_stride=5,
                context_size=10,
                feature_type=feature_type,
            )
            print("supervised", supervised,
                  feature_type, [k.shape for k in res])
