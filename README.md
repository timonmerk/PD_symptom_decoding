### Code repository - Multitarget brain implants enable generalized decoding of Parkinson’s disease symptoms from chronic home recordings

Contact information: timon.merk95@gmail.com

<u>Data pre-processing</u>
1. [analysis-rcs-data](https://github.com/openmind-consortium/Analysis-rcs-data) toolbox to convert Medtronic RC+S .json data
2. rewrite data in parquet format: [change_data_to_parquet.py](change_data_to_parquet.py)
3. compute *py-neuromodulation* features: [run_ucsf_features.py](run_ucsf_features.py)
4. merge data across patients: [merge_ucsf_features.py](merge_ucsf_features.py)

<u>Anaylses</u>
 - comparison of different ML time-windows: [how_much_data_is_needed.py](how_much_data_is_needed.py)
 - decoding of cross-validation within channels: [figure_27_get_ch_ind_per_all_ch.py](figure_27_get_ch_ind_per_all_ch.py)
 - decoding include / exlude hour and hour only decoding: [run_decoding_ucsf_across_patients_hour.py](run_decoding_ucsf_across_patients_hour.py)
 - decoding include / exclude nighttime analysis: [run_decoding_ucsf_across_patients_nighttime.py](run_decoding_ucsf_across_patients_nighttime.py)
 - decoding different ML models: [run_decoding_ucsf_different_ML_methods.py](run_decoding_ucsf_different_ML_methods.py)
 - decoding different ML features: [run_decoding_ucsf_across_patients_diff_features.py](run_decoding_ucsf_across_patients_diff_features.py)
 - decoding leave one subject vs hemisphere decoding cross-validation: [leave_one_subject_out_not_hemisphere_validation.py](leave_one_subject_out_not_hemisphere_validation.py)

<u>Figures</u>
 - Figure 1 PKG wearable correlation with UPDRS scores: [pkg_corr_figure.py](pkg_corr_figure.py)
 - Figure 2 Power spectra all patients: [plot_psd.py](plot_psd.py)
 - Figure 2 Beta peak frequencies across patients: [figure_48_beta_plts.py](figure_48_beta_plts.py)
 - Figure 2 Comparison beta power vs ML decoding [beta_ml_comp.py](beta_ml_comp.py)
 - Figure 2 Time-series bet and ML example time-traces: [time_series_prediction_examples.py](time_series_prediction_examples.py)
 - Figure 2 ML performances when beta fails: [figure_54_comp_nonsigbeta_patients_all.py](figure_54_comp_nonsigbeta_patients_all.py)
 - Figure 2 ML performances during tremor or dyskinesia: [figure_53_per_during_tremor.py](figure_53_per_during_tremor.py)
 - Figure 3: ML analysis different features, different time windows, feature importances: [figure_49_joint_plot_2.py](figure_49_joint_plot_2.py)
 - Figure 4 Symptom vs activated fiber plot: [connectomic_DBS/activated_cleartune_fiber_correlation.m](connectomic_DBS/activated_cleartune_fiber_correlation.m)
 - Figure 4 Examplary tract activattion: [connectomic_DBS/Visualizations.m](connectomic_DBS/Visualizations.m)
 - Supplementary Figure 1 Data duration per patient: [figure_24_data_duration.py](figure_24_data_duration.py)
 - Supplementary Figure 1 Dyskinesia PKG example: [figure_1_example_pkg.py](figure_1_example_pkg.py)
 - Supplementary Figure 2 ML architecture comparison: [figure_49_joint_plot_2.py](figure_49_joint_plot_2.py)
 - Supplmentary Figure 3 ML feature comparison, model comparison, train duration analysis, feature importance: [figure_50_figure_3_paper.py](figure_50_figure_3_paper.py)
 - Supplementary Figure 4 ML performance comparison recording sites DBS + ECOG: [figure_54_compare_ML_STN_ECOG.py](figure_54_compare_ML_STN_ECOG.py) 
 - Supplementary Video 1: [connectomic_DBS/figure_54_create_video.py](connectomic_DBS/figure_54_create_video.py)

<u>Neural decoding models</u>: https://github.com/neuromodulation/AcrossPatientDecodingModel 

Setup:
[uv](https://docs.astral.sh/uv/getting-started/installation/) install:

```
uv venv
uv sync
```