# brainscore
Repo to compute the brain score of a transformer model (huggingface) on the Hasson dataset
Compatible with Hugginface models

```
pip install -r requirements.txt
```

```
pip install -e
```

## Run evaluation on the average subject
```
from brainscore import run_eval
model_name_or_path = "gpt2" # Huggingface model 
local = True # If False, run on the cluster with submitit
run_eval(model_name_or_path, average_bold=True)
```

## Run evaluation on single subjects

```
from brainscore import run_eval
model_name_or_path = "gpt2" # Huggingface model 
n_subjects = 2
local = True # If False, run on the cluster with submitit
run_eval(model_name_or_path, n_subjects=n_subjects, average_bold=False)
```
