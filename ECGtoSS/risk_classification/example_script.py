"""
Filename: example_script.py

Description: This file contains example scripts for implementing the MESA and MIMIC pipelines
"""

import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from imblearn.over_sampling import SMOTE

# python script imports
import mimic_diagnoses
import preprocessing
import models
import visualizations

# reload the files every time
import importlib
importlib.reload(mimic_diagnoses)
importlib.reload(preprocessing)
importlib.reload(models)
importlib.reload(visualizations)


# MIMIC EXAMPLE SCRIPT

# read clinical data from .csv files
patients = pd.read_csv('mimic_data/PATIENTS.csv')
admissions = pd.read_csv('mimic_data/ADMISSIONS.csv')
diagnoses = pd.read_csv('mimic_data/DIAGNOSES_ICD.csv')

# get subset of clinical data for patients with ECG lead ii and ABP waveform data
admissions_leadii, patients_leadii, diagnoses_leadii = mimic_diagnoses.get_leadii_dataframes(patients, admissions, diagnoses)

# get ethnicity from admissions ETHNICITY column and get its umbrella race
ethnicities_leadii = admissions_leadii.drop_duplicates(subset='SUBJECT_ID')
ethnicities_leadii['ETHNICITY_COND'] = ethnicities_leadii.ETHNICITY.apply(lambda x: mimic_diagnoses.get_race(x))

# add icd 10 column to diagnoses_leadii using conversion from icd 9 codes
mimic_diagnoses.add_icd_10_code_to_diagnoses(diagnoses_leadii)

# get aligned sbp, dbp, and sleep stages from files
data_dictionary = preprocessing.get_aligned_ss_and_bp('MIMIC', admissions_leadii)
# get patient ids
patient_ids = np.array([tup[0] for tup in list(data_dictionary.keys())])
# finds sleep starts for each instance and returns an array that starts at the sleep start
# the integer corresponding to awake in MIMIC is 3
start_before_sleep_arrays = [preprocessing.get_start_before_sleep(bp_ss, 3) for bp_ss in list(data_dictionary.values())]

# rolling means over 15 minute windows
start_before_sleep_arrays_bp_meaned = []
for arr in start_before_sleep_arrays:
    sbp = arr[0]
    dbp = arr[1]
    ss = arr[2]
    try:
        sbp_means = np.pad(np.apply_along_axis(lambda x: np.mean(x), 1, sliding_window_view(sbp, 30)), (15, 14), 'edge')
        dbp_means = np.pad(np.apply_along_axis(lambda x: np.mean(x), 1, sliding_window_view(dbp, 30)), (15, 14), 'edge')
    except:
        sbp_means = sbp
        dbp_means = dbp

    arr = np.concatenate((sbp_means.reshape((1,-1)),dbp_means.reshape((1,-1)),ss.reshape((1,-1))),axis=0)
    start_before_sleep_arrays_bp_meaned += [arr]

# mean zero for both raw bp signals
start_before_sleep_arrays_m0 = preprocessing.mean_zero(start_before_sleep_arrays)

# summary statistics features for fixed 6 hour blocks
X_sum, y_sum, patient_ids_sum = preprocessing.get_features(patient_ids, start_before_sleep_arrays_m0, 'MIMIC', True, True, False, patients_leadii, admissions_leadii, diagnoses_leadii, 6)
# summary statistics + age + sex features for fixed 6 hour blocks
X_sum_dem, y_sum_dem, patient_ids_sum_dem = preprocessing.get_features(patient_ids, start_before_sleep_arrays_m0, 'MIMIC', True, True, True, patients_leadii, admissions_leadii, diagnoses_leadii, 6)
# fixed block 6 hours
X_ts, y_ts, patient_ids_ts = preprocessing.get_features(patient_ids, start_before_sleep_arrays_m0, 'MIMIC', False, True, False, patients_leadii, admissions_leadii, diagnoses_leadii, 6, 6)

# feature extraction using tsfresh package
X_ts_feats = models.calculate_features(X_ts)

# visualize FFT magnitude_phase_plot from feats
# index of instance to visualize
instance_ind = 10
# which signal to visualize
sbp_ind = 0
dbp_ind = 1
# get magnitude and phase columns
abs_cols = [col for col in X_ts_feats.columns if 'fft_coefficient' in col and 'abs' in col and 'single__' + str(sbp_ind) in col]
angle_cols = [col for col in X_ts_feats.columns if 'fft_coefficient' in col and 'angle' in col and 'single__' + str(sbp_ind) in col]
# get magnitudes and phases
magnitudes = X_ts_feats.loc[instance_ind][abs_cols]
phases = X_ts_feats.loc[instance_ind][angle_cols]
# visualize
N = 720 # for a 6 hour fixed block
dt = 30 # samples every 30 second
fft_magnitude_phase_plot(N, dt, magnitudes, phases)

# SMOTE oversampling
sm = SMOTE(sampling_strategy=1.0)
# drops columns with nans
X_smote, y_smote = sm.fit_resample(X_ts_feats.dropna(axis=1, inplace=False), y_ts)

# for different levels of supervision
# 10 runs of each combination of model type and transform type, stores features selected from best performing models
unsupervised_accuracies = models.compare_averaged_unsupervised_clustering(X_smote, y_smote, 10)
supervised_accuracies_20, all_models_dict_20 = models.get_selected_features_and_scores_over_n_runs(10, X_smote, y_smote, 0.2)
supervised_accuracies_50, all_models_dict_50 = models.get_selected_features_and_scores_over_n_runs(10, X_smote, y_smote, 0.5)
supervised_accuracies_80, all_models_dict_80 = models.get_selected_features_and_scores_over_n_runs(10, X_smote, y_smote, 0.8)



# MESA EXAMPLE SCRIPT

# get aligned sbp, dbp, and sleep stages from files
data_dictionary = preprocessing.get_aligned_ss_and_bp()
# get patient ids
patient_ids = np.array(list(data_dictionary.keys()))
# finds sleep starts for each instance and returns an array that starts at the sleep start
# the integer corresponding to awake in MESA is 0
start_before_sleep_arrays = [preprocessing.get_start_before_sleep(bp_ss,0) for bp_ss in list(data_dictionary.values())]

# cap out unrealistic blood pressure values
start_before_sleep_arrays_cap = []
for arr in start_before_sleep_arrays:
    sbp = arr[0]
    dbp = arr[1]
    ss = arr[2]

    # cap SBP
    sbp_capped = np.array([val if val<=180 else 180 for val in sbp])
    sbp_capped = np.array([val if val >= 80 else 80 for val in sbp_capped])
    # cap DBP
    dbp_capped = np.array([val if val >= 40 else 40 for val in dbp])
    dbp_capped = np.array([val if val <= 120 else 120 for val in dbp_capped])

    # reconcatenate
    arr = np.concatenate((sbp_capped.reshape((1, -1)), dbp_capped.reshape((1, -1)), ss.reshape((1, -1))), axis=0)
    start_before_sleep_arrays_cap += [arr]

# get 6 hour fixed blocks
X_ts_6, patient_ids_ts_6 = preprocessing.get_features(patient_ids, start_before_sleep_arrays_cap, 'MESA', False, False, False, None, None, None, 6, 6)

# take rolling median of blood pressure signals
X_ts_6_rmed = []
for arr in X_ts_6:
    sbp_rmed = np.pad(np.apply_along_axis(lambda x: np.median(x), 1, sliding_window_view(arr[0], 3)), (1, 1), 'edge')
    dbp_rmed = np.pad(np.apply_along_axis(lambda x: np.median(x), 1, sliding_window_view(arr[1], 3)), (1, 1), 'edge')
    ss = arr[2]
    # reconcatenate
    arr = np.concatenate((sbp_rmed.reshape((1, -1)), dbp_rmed.reshape((1, -1)), ss.reshape((1, -1))), axis=0)
    X_ts_6_rmed += [arr]
X_ts_6_rmed = np.array(X_ts_6_rmed)

# mean zero the blood pressure signals
X_ts_6_m0 = np.array(preprocessing.mean_zero(X_ts_6_rmed))

# extract features using tsfresh package
X_ts_6_m0_feats = models.calculate_features(X_ts_6_m0)

# get cluster predictions for different combinations of model and transform types
n_clusters = 2
y_preds = models.compare_unsupervised_clustering_MESA(X_ts_6_m0_feats, n_clusters)

# visualize clusters
# reduce dimensionality for visualization
X_reduced = visualizations.reduce_dimensions(X_ts_6_m0_feats.dropna(axis=1, inplace=False))
for i in range(9):
    # get model type
    if i in [0,1,2]:
        model_type = 'Hierarchical'
    elif i in [3,4,5]:
        model_type = 'KMeans'
    else:
        model_type = 'Spectral'

    # get transform type
    if i % 3 == 0:
        transform_type = 'std'
    elif i % 3 == 1:
        transform_type = 'minmax'
    else:
        transform_type = 'robust'

    # visualize
    visualizations.clustering_visualization(X_reduced, y_preds[i], 2, model_type, transform_type)