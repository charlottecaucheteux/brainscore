from pathlib import Path

# Narrative dir
base_dir = Path("/datasets01/hasson_narratives")

# Brain data
deriv_dir = base_dir / "derivatives"
afni_dir = deriv_dir / "afni-smooth"
afni_dir_nosmooth = deriv_dir / "afni-nosmooth"
fmriprep_dir = deriv_dir / "fmriprep"

# Stimuli
stimuli = base_dir / "stimuli"
gentle_path = base_dir / "stimuli" / "gentle"

# Events
event_meta_path = base_dir / "code" / "event_meta.json"
task_meta_path = base_dir / "code" / "task_meta.json"
scan_exclude_path = base_dir / "code" / "scan_exclude.json"

# Brain map
surf_dir = deriv_dir / "freesurfer" / "fsaverage6" / "surf"
sulc_left = str(surf_dir / "lh.sulc")
sulc_right = str(surf_dir / "rh.sulc")
inf_left = str(surf_dir / "lh.inflated")
inf_right = str(surf_dir / "rh.inflated")

# Repo
root = Path("")

# Fairseq Models (TO CHANGE HERE)
# e.g. : models/fairseq/fairseq_models_juliette/checkpoint_finetuned_english.pt
fairseq_models = root / "models" / "fairseq" / "fairseq_models_juliette"

# Outputs
data = root / "data"
speech_embeddings = data / "speech_embeddings"
scores = root / "scores"

# Other
checked_gentle_path = root / "stimuli" / "gentle_checked"

# Mean bold (TO CHANGE)
mean_bolds = data / "bold" / "mean_bolds_concat_tasks_%s.npy"
