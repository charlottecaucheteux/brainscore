from pathlib import Path

# Data to load
# base_dir = Path("/private/home/ccaucheteux/narratives")

fmriprep_dir = Path("/datasets01/hasson_narratives") / \
    "derivatives" / "fmriprep"

# base_dir = Path("/checkpoint/ccaucheteux/narratives")
base_dir = Path("/datasets01/hasson_narratives")
deriv_dir = base_dir / "derivatives"
afni_dir = base_dir / "derivatives" / "afni-smooth"
afni_dir_nosmooth = base_dir / "derivatives" / "afni-nosmooth"

event_meta_path = base_dir / "code" / "event_meta.json"
task_meta_path = base_dir / "code" / "task_meta.json"
scan_exclude_path = base_dir / "code" / "scan_exclude.json"

# probe_path = Path("/private/home/ccaucheteux/structural-probes")
# pos_equiv_file = (
#     "/private/home/ccaucheteux/drafts/data/english_equivalence_pos_test.npy"
# )
# posdep_equiv_file = (
#     "/private/home/ccaucheteux/drafts/data/english_equivalence_pos_dep_1M.npy"
# )
# wiki_dir = Path("/checkpoint/ccaucheteux/train-xlm-models/XLM/data/wiki/txt/")
# wiki_100m_path = wiki_dir / "en.100m"
# wiki_20m_path = wiki_dir / "en.20m"
# wiki_20m_splits = [wiki_dir / f"en.5m.{i}" for i in [1, 2, 3, 4]]
# wiki_20m_1_path = wiki_dir / "en.20m.1"
# wiki_20m_2_path = wiki_dir / "en.20m.2"

# Brain map
surf_dir = base_dir / "derivatives/freesurfer/fsaverage6/surf/"
sulc_left = str(surf_dir / "lh.sulc")
sulc_right = str(surf_dir / "rh.sulc")
inf_left = str(surf_dir / "lh.inflated")
inf_right = str(surf_dir / "rh.inflated")

# Repo
root = Path("")

stimuli = base_dir / "stimuli"
gentle_path = base_dir / "stimuli" / "gentle"
checked_gentle_path = root / "stimuli" / "gentle_checked"

# Data to generate
data = root / "data"
phone_dic = data / "phone_dic.npy"
phone0_dic = data / "phone0_dic.npy"
phone1_dic = data / "phone1_dic.npy"
pos_dic = data / "pos_dic.npy"
gpt2_vocab_spacy_feats = data / "spacy_feats" / "gpt2_vocab_spacy_features.npy"
future_predictions = data / "future_predictions"


wiki_seq_len_dic = data / "wiki_seq_len_dic.npy"
syn_equiv_dir = data / "syntactic_equivalences"
syn_equiv_file = str(
    data
    / "syntactic_equivalences"
    / "0201_wiki_valid"
    / "%s"
    / "%s"
    / "equival_story.npy"
)
embeddings = data / "embeddings"
speech_embeddings = data / "speech_embeddings"
mean_bolds = data / "bold" / "mean_bolds_concat_tasks_%s.npy"
mean_bolds_rois = data / "bold" / "mean_bolds_concat_tasks_rois_%s.npy"
mean_bolds_mapping_rois = data / "bold" / "rois_%s.npy"
median_bolds = data / "bold" / "median_bold_concat_tasks_%s.npy"
slice_bolds = data / "bold" / "sliced_voxels"
dict_bolds = data / "bold" / "dict_tasks"

# Results
scores = root / "scores"

# Wiki syntactic embeddings (no brain)
# wiki_bar_embeddings = data / "wiki_bar_embeddings"
# wiki_gpt2_embeddings = data / "wiki_gpt2_embeddings"
# wiki_gpt2_pca = data / "wiki_gpt2_pca"
# wiki_gpt2_pca_folder_name = "0727"  # "0706"


fairseq_models = root / "models" / "fairseq" / "fairseq_models_juliette"
