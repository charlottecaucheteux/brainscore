# brainscore

Repo to compute the brainscore of the mean subject or single subjects on the Narrative dataset.

```
pip install -r requirements.txt
```

## Setup paths

* To change Narrative dir: setup `base_dir` in `brainscore/paths.py` (default to `"/datasets01/hasson_narratives"`)
* To change the mean bold dir: setup `mean_bolds` in `brainscore/paths.py` (default to `"root/data/bold/mean_bolds_concat_tasks_L.npy"`)

## Compute Wav2Vec activations and compute brainscore on average bold

To compute the activations of pretrained *Wav2Vec2* from huggingface for the narrative "pieman", on *rois* and compute the brainscore on the *average subject*, run:

```
python main.py
```




## Run evaluation on single subjects and voxels

In `main.py`, setup: 

```
subject_name = "sub-004"
score = _job_compute_speech_brain_score(embed_files, output_file,
        subject=subject_name,
        layers=(0, 8),
        to_rois=False, # True for rois, False for voxels
        select_tasks=["pieman"]) # set to None if all tasks
```
