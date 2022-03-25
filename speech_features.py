from brainscore.run_speech import run_eval_speech

if __name__ == "__main__":
    # for hrf in ['glover', 'glover + derivative', 'glover + derivative + dispersion', 'spm', 'spm + derivative', 'spm + derivative + dispersion']:
    run_eval_speech(
        output_path=f"outputs/wav2vec_glover/test_x_pca_rois/",
        model_name="wav2vec2",
        feature_type="conv",
        layers=None,
        hrf_model="glover",
        average_bold=False,
        n_subjects=300,
        hemis=["L"],
        to_rois=True,
        x_pca=20,
        cache_path="/checkpoint/ccaucheteux/cache",
        slurm_partition="devlab",
        local=True,
        delete_cache_after_run=False,
        overwrite=True,
    )
