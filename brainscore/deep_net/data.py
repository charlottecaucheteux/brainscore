"""TODO"""
import logging

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def get_activations(
    stimulus,
    model_name_or_path="gpt2",
    max_len=1024,
    col="word_raw",
    device="cpu",
    bidir=False,  # whether to include future context
):
    stimulus["word_index"] = np.arange(len(stimulus))

    if bidir:
        future_context = max_len//2
    else:
        future_context = 0

    # Align sequences with fixed context
    sequences = []
    for i, row in stimulus.iterrows():
        wp = row.word_index
        # put current word at the end
        words = np.roll(stimulus[col].values, -(wp + 1 + future_context))
        fwd = " ".join(words)
        sequences.append(fwd)
    assert len(sequences) == len(stimulus)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, add_special_token=False)
    model = AutoModel.from_pretrained(model_name_or_path)
    if model.config.model_type == "gpt2":
        fix_embeddings = True
        logging.warning("Fix word embedding for gpt2 model")
    else:
        fix_embeddings = False
    if torch.cuda.is_available():
        model.to(device)

    # Extract embeddings
    outputs = []
    with torch.no_grad():
        for seq in tqdm(sequences):
            inpt = tokenizer.encode(seq, return_tensors="pt")[:, -max_len:]
            inpt = inpt.to(model.device)
            out = model(inpt, output_hidden_states=True)
            out = torch.stack(out.hidden_states)
            if fix_embeddings:
                out[0] = model.base_model.wte.forward(inpt)[None]
            out = out[:, 0, -(1+future_context)]
            out = out.cpu()
            outputs.append(out)
    outputs = torch.stack(outputs, dim=1)
    assert outputs.size(1) == len(sequences) == len(stimulus)
    return outputs
