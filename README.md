# brainscore

## Goal

Computes the alignment, or "brainscore" between the fMRI signal of participants of the Narratives dataset, and one speech model (e.g. Wav2Vec2) or language model (e.g. GPT-2), when both the participants and the models process the same language input. 

e.g. `main.py` computes the activations of pretrained Wav2Vec2 from huggingface in response to the story called "pieman", and computes the brainscore on the fMRI signal, averaged across participants and some regions of interests in the brain.


## Core steps

**Extracting deep nets' activations and brain activity**
- `src/brain/data.py` handles preprocessing and loading of fMRI data (from the Narratives dataset, participants listen to language stimulus).
- `src/deep_net/data.py` handles computing the activations of a language model (GPT-2/BERT), given text stimulus
- `src/deep_net/data_speech.py` handles computing the activations of a speech model (Wav2Vec2), given audio stimulus

**Compute the alignment between both**
- `src/mapping.py` computes the linear mapping between X and Y

## Others: paths, exp, metrics

- `src/paths.py` handles paths.
    + To change Narrative dir: setup `base_dir` in `brainscore/paths.py` (default to `"/datasets01/hasson_narratives"`)
    + To change the mean fMRI dir: setup `mean_bolds` in `brainscore/paths.py` (default to `"root/data/bold/mean_bolds_concat_tasks_L.npy"`)
- `src/get_brain_score.py` and `src/get_brain_score_speech.py` handles the exp (extracting data, features, setting up splits, model and eval depending on the exp)
- `src/metrics.py` metrics to use to evaluate the alignment (e.g. R, R2, V2V etc). 
