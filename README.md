# brainscore
Repo to compute the brain score of a transformer model (huggingface) on the Hasson dataset
Compatible with Hugginface models

```
pip install -r requirements.txt
```

# Generate fairseq embeddings

Need a separate environment for fairseq

modify `fairseq_models` in brainscore/paths
run `0f_extract_fairseq_activations.py` with LOCAL=True (to test in local)
This will save embeddings in `data/speech_embeddings/fairseq_st5_ct10/TASK_FEATURE.pth`

# Run brain scores
modify `base_dir` in brainscore/paths
run `1f_brainscore_fairseq.py` with LOCAL=True (to test in local). 
This will save brainscores in `scores/fairseq/fairseq_st5_ct10/FEATURE/sub_hemi.npy`

To run for all layers:
```
`1f_brainscore_fairseq.py`
```

To run for concatenated layers:
```
`2f_brainscore_fairseq_concat.py`
```


To run for concatenated layers (hierarchically):
```
`3f_brainscore_fairseq_hierarch_concat.py`
```


