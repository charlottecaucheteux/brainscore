import numpy as np
import torch


def get_model_weights_fromFairSeq(modelHF, model):
    modelHF.wav2vec2.feature_extractor.conv_layers[0].conv.weight = \
        model.feature_extractor.conv_layers[0][0].weight
    modelHF.wav2vec2.feature_extractor.conv_layers[0].layer_norm.weight = \
        model.feature_extractor.conv_layers[0][2].weight
    modelHF.wav2vec2.feature_extractor.conv_layers[0].layer_norm.bias = \
        model.feature_extractor.conv_layers[0][2].bias
    for i in np.arange(1, 7):
        modelHF.wav2vec2.feature_extractor.conv_layers[i].conv.weight = \
            model.feature_extractor.conv_layers[i][0].weight

    # projection post feature extraction
    modelHF.wav2vec2.feature_projection.layer_norm.weight = model.layer_norm.weight
    modelHF.wav2vec2.feature_projection.layer_norm.bias = model.layer_norm.bias
    modelHF.wav2vec2.feature_projection.weight = model.post_extract_proj.weight
    modelHF.wav2vec2.feature_projection.bias = model.post_extract_proj.bias

    # quantifier: weights and codevectors
    modelHF.quantizer.codevectors = model.quantizer.vars
    modelHF.quantizer.weight_proj.bias = model.quantizer.weight_proj.bias
    modelHF.quantizer.weight_proj.weight = model.quantizer.weight_proj.weight

    # Quantifier projections
    modelHF.project_q.weight = model.project_q.weight
    modelHF.project_q.bias = model.project_q.bias

    # encoder:
    # position embedding
    modelHF.wav2vec2.encoder.pos_conv_embed.conv.bias = model.encoder.pos_conv[0].bias
    modelHF.wav2vec2.encoder.pos_conv_embed.conv.weight_g = model.encoder.pos_conv[
        0].weight_g
    modelHF.wav2vec2.encoder.pos_conv_embed.conv.weight_v = model.encoder.pos_conv[
        0].weight_v

    # series of transformer layers:
    assert len(modelHF.wav2vec2.encoder.layers) == len(model.encoder.layers)
    for i in range(len(modelHF.wav2vec2.encoder.layers)):
        modelHF.wav2vec2.encoder.layers[i].attention.k_proj.weight = model.encoder.layers[i].self_attn.k_proj.weight
        modelHF.wav2vec2.encoder.layers[i].attention.k_proj.bias = model.encoder.layers[i].self_attn.k_proj.bias
        modelHF.wav2vec2.encoder.layers[i].attention.v_proj.weight = model.encoder.layers[i].self_attn.v_proj.weight
        modelHF.wav2vec2.encoder.layers[i].attention.v_proj.bias = model.encoder.layers[i].self_attn.v_proj.bias
        modelHF.wav2vec2.encoder.layers[i].attention.q_proj.weight = model.encoder.layers[i].self_attn.q_proj.weight
        modelHF.wav2vec2.encoder.layers[i].attention.q_proj.bias = model.encoder.layers[i].self_attn.q_proj.bias
        modelHF.wav2vec2.encoder.layers[i].attention.out_proj.weight = model.encoder.layers[i].self_attn.out_proj.weight
        modelHF.wav2vec2.encoder.layers[i].attention.out_proj.bias = model.encoder.layers[i].self_attn.out_proj.bias

        modelHF.wav2vec2.encoder.layers[i].layer_norm.weight = model.encoder.layers[i].self_attn_layer_norm.weight
        modelHF.wav2vec2.encoder.layers[i].layer_norm.bias = model.encoder.layers[i].self_attn_layer_norm.bias

        modelHF.wav2vec2.encoder.layers[i].feed_forward.intermediate_dense.weight = model.encoder.layers[i].fc1.weight
        modelHF.wav2vec2.encoder.layers[i].feed_forward.intermediate_dense.bias = model.encoder.layers[i].fc1.bias
        modelHF.wav2vec2.encoder.layers[i].feed_forward.output_dense.weight = model.encoder.layers[i].fc2.weight
        modelHF.wav2vec2.encoder.layers[i].feed_forward.output_dense.bias = model.encoder.layers[i].fc2.bias

        modelHF.wav2vec2.encoder.layers[i].final_layer_norm.weight = model.encoder.layers[i].final_layer_norm.weight
        modelHF.wav2vec2.encoder.layers[i].final_layer_norm.bias = model.encoder.layers[i].final_layer_norm.bias

    # final layer normalization in the encoder
    modelHF.wav2vec2.encoder.layer_norm.weight = model.encoder.layer_norm.weight
    modelHF.wav2vec2.encoder.layer_norm.bias = model.encoder.layer_norm.bias

    # last projection post encoder
    modelHF.project_hid.weight = model.final_proj.weight
    modelHF.project_hid.bias = model.final_proj.bias

    # mask embedding
    modelHF.wav2vec2.masked_spec_embed = model.mask_emb

    return modelHF


def verify_wav2vec2_weights_HFandFairSeq(modelHF, model):
    assert torch.all(
        modelHF.wav2vec2.feature_extractor.conv_layers[0].conv.weight == model.
        feature_extractor.conv_layers[0][0].weight)
    assert torch.all(
        modelHF.wav2vec2.feature_extractor.conv_layers[0].layer_norm.weight ==
        model.feature_extractor.conv_layers[0][2].weight)
    assert torch.all(
        modelHF.wav2vec2.feature_extractor.conv_layers[0].layer_norm.bias ==
        model.feature_extractor.conv_layers[0][2].bias)
    for i in np.arange(1, 7):
        assert torch.all(
            modelHF.wav2vec2.feature_extractor.conv_layers[i].conv.weight ==
            model.feature_extractor.conv_layers[i][0].weight)

    # projection post feature extraction
    assert torch.all(
        modelHF.wav2vec2.feature_projection.layer_norm.weight == model.layer_norm.weight)
    assert torch.all(
        modelHF.wav2vec2.feature_projection.layer_norm.bias == model.layer_norm.bias)
    assert torch.all(modelHF.wav2vec2.feature_projection.weight ==
                     model.post_extract_proj.weight)
    assert torch.all(modelHF.wav2vec2.feature_projection.bias ==
                     model.post_extract_proj.bias)

    # quantifier: weights and codevectors
    assert torch.all(modelHF.quantizer.codevectors == model.quantizer.vars)
    assert torch.all(modelHF.quantizer.weight_proj.bias ==
                     model.quantizer.weight_proj.bias)
    assert torch.all(modelHF.quantizer.weight_proj.weight ==
                     model.quantizer.weight_proj.weight)

    # Quantifier projections
    assert torch.all(modelHF.project_q.weight == model.project_q.weight)
    assert torch.all(modelHF.project_q.bias == model.project_q.bias)

    # encoder:
    # position embedding
    assert torch.all(modelHF.wav2vec2.encoder.pos_conv_embed.conv.bias ==
                     model.encoder.pos_conv[0].bias)
    assert torch.all(modelHF.wav2vec2.encoder.pos_conv_embed.conv.weight_g ==
                     model.encoder.pos_conv[0].weight_g)
    assert torch.all(modelHF.wav2vec2.encoder.pos_conv_embed.conv.weight_v ==
                     model.encoder.pos_conv[0].weight_v)

    # series of transformer layers:
    assert len(modelHF.wav2vec2.encoder.layers) == len(model.encoder.layers)
    for i in range(len(modelHF.wav2vec2.encoder.layers)):
        assert torch.all(
            modelHF.wav2vec2.encoder.layers[i].attention.k_proj.weight == model.
            encoder.layers[i].self_attn.k_proj.weight)
        assert torch.all(
            modelHF.wav2vec2.encoder.layers[i].attention.k_proj.bias == model.
            encoder.layers[i].self_attn.k_proj.bias)
        assert torch.all(
            modelHF.wav2vec2.encoder.layers[i].attention.v_proj.weight == model.
            encoder.layers[i].self_attn.v_proj.weight)
        assert torch.all(
            modelHF.wav2vec2.encoder.layers[i].attention.v_proj.bias == model.
            encoder.layers[i].self_attn.v_proj.bias)
        assert torch.all(
            modelHF.wav2vec2.encoder.layers[i].attention.q_proj.weight == model.
            encoder.layers[i].self_attn.q_proj.weight)
        assert torch.all(
            modelHF.wav2vec2.encoder.layers[i].attention.q_proj.bias == model.
            encoder.layers[i].self_attn.q_proj.bias)
        assert torch.all(
            modelHF.wav2vec2.encoder.layers[i].attention.out_proj.weight ==
            model.encoder.layers[i].self_attn.out_proj.weight)
        assert torch.all(
            modelHF.wav2vec2.encoder.layers[i].attention.out_proj.bias == model.
            encoder.layers[i].self_attn.out_proj.bias)

        assert torch.all(
            modelHF.wav2vec2.encoder.layers[i].layer_norm.weight == model.
            encoder.layers[i].self_attn_layer_norm.weight)
        assert torch.all(
            modelHF.wav2vec2.encoder.layers[i].layer_norm.bias == model.encoder.
            layers[i].self_attn_layer_norm.bias)

        assert torch.all(modelHF.wav2vec2.encoder.
                         layers[i].feed_forward.intermediate_dense.weight ==
                         model.encoder.layers[i].fc1.weight)
        assert torch.all(
            modelHF.wav2vec2.encoder.layers[i].feed_forward.intermediate_dense.bias == model.encoder.layers[i].fc1.bias)
        assert torch.all(
            modelHF.wav2vec2.encoder.layers[i].feed_forward.output_dense.weight
            == model.encoder.layers[i].fc2.weight)
        assert torch.all(
            modelHF.wav2vec2.encoder.layers[i].feed_forward.output_dense.bias ==
            model.encoder.layers[i].fc2.bias)

        assert torch.all(
            modelHF.wav2vec2.encoder.layers[i].final_layer_norm.weight == model.
            encoder.layers[i].final_layer_norm.weight)
        assert torch.all(
            modelHF.wav2vec2.encoder.layers[i].final_layer_norm.bias == model.
            encoder.layers[i].final_layer_norm.bias)

    # final layer normalization in the encoder
    assert torch.all(modelHF.wav2vec2.encoder.layer_norm.weight ==
                     model.encoder.layer_norm.weight)
    assert torch.all(modelHF.wav2vec2.encoder.layer_norm.bias ==
                     model.encoder.layer_norm.bias)

    # last projection post encoder
    assert torch.all(modelHF.project_hid.weight == model.final_proj.weight)
    assert torch.all(modelHF.project_hid.bias == model.final_proj.bias)

    # mask embedding
    assert torch.all(modelHF.wav2vec2.masked_spec_embed == model.mask_emb)

    return modelHF


def get_config_fairseq_to_hf(modelHF, model):
    # next we want to check for the config parameters:

    modelHF.config.activation_dropout = model.encoder.layers[0].dropout1.p
    for i in range(len(modelHF.wav2vec2.encoder.layers)):
        modelHF.wav2vec2.encoder.layers[i].dropout.p = model.encoder.layers[i].dropout1.p
        modelHF.wav2vec2.encoder.layers[i].feed_forward.intermediate_dropout.p = model.encoder.layers[i].dropout2.p
        modelHF.wav2vec2.encoder.layers[i].feed_forward.output_dropout.p = model.encoder.layers[i].dropout3.p
    assert modelHF.config.hidden_act == model.args.activation_fn
    # otherwise one has to reinstentiate the layers...
    modelHF.config.hidden_dropout = model.args.dropout

    modelHF.config.mask_time_selection = model.mask_selection
    modelHF.config.contrastive_logits_temperature = model.logit_temp
    modelHF.config.do_stable_layer_norm = model.args.layer_norm_first
    modelHF.config.feat_extract_activation = model.args.activation_fn
    # we found no equivalent of dropout_input
    # model.dropout_input
    # model.args.dropout_input
    modelHF.dropout_features.p = model.dropout_features.p
    # modelHF.dropout_features.p = model.dropout_features.p

    # not sure about that:
    modelHF.wav2vec2.feature_projection.dropout.p = model.dropout_features.p
    modelHF.config.feat_proj_dropout = model.args.dropout_features

    # layer drop
    modelHF.config.layerdrop = model.args.encoder_layerdrop

    # masking parameters:
    modelHF.config.mask_channel_length = model.args.mask_channel_length
    modelHF.config.mask_channel_min_space = model.args.mask_channel_min_space
    modelHF.config.mask_channel_other = model.args.mask_channel_other
    modelHF.config.mask_channel_prob = model.args.mask_channel_prob
    modelHF.config.mask_channel_selection = model.args.mask_channel_selection
    model.no_mask_channel_overlap = model.args.no_mask_channel_overlap

    modelHF.config.mask_time_prob = model.args.mask_prob
    modelHF.config.mask_time_length = model.args.mask_length
    modelHF.config.mask_time_min_space = model.args.mask_min_space
    modelHF.config.mask_time_other = model.args.mask_other
    modelHF.config.mask_time_selection = model.args.mask_selection
    modelHF.config.no_mask_time_overlap = model.args.no_mask_overlap

    modelHF.config.mask_feature_prob = model.args.mask_prob
    modelHF.config.mask_feature_length = model.args.mask_length
    modelHF.config.mask_feature_min_space = model.args.mask_min_space
    modelHF.config.mask_feature_other = model.args.mask_other
    modelHF.config.mask_feature_selection = model.args.mask_selection
    modelHF.config.no_mask_feature_overlap = model.args.no_mask_overlap

    # remain to check:

    # "_name_or_path": "facebook/wav2vec2-base",
    #   "apply_spec_augment": true,
    #   "attention_dropout": 0.1,
    #   "bos_token_id": 1,
    #   "eos_token_id": 2,
    #   "diversity_loss_weight": 0.1,
    #   "feat_extract_norm": "group",
    #   "feat_proj_dropout": 0.1,

    #   "freeze_feat_extract_train": true,
    #   "gradient_checkpointing": true,
    #   "initializer_range": 0.02,
    #   "layer_norm_eps": 1e-05,
    #   "pad_token_id": 0,
    #   "proj_codevector_dim": 256,
    #   "torch_dtype": "float32",
    #   "transformers_version": "4.10.2",
    #   "use_weighted_layer_sum": false,
    #   "vocab_size": 32
    return modelHF
