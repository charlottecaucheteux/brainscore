from brainscore.run import run_eval

if __name__ == "__main__":
    run_eval("gpt2",
             "test_gpt2_brain_score",
             layers=[8],
             average_bold=False,
             n_subjects=10,
             hemis=["L"],
             to_rois=True,
             x_pca=20,
             cache_path="/checkpoint/ccaucheteux/cache",
             slurm_partition="dev",
             local=True,
             delete_cache_after_run=False,
             model_max_len=128,
             overwrite=True,
             )
