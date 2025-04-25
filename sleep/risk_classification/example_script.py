import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from imblearn.over_sampling import SMOTE

# python script imports
import mimic_diagnoses
import preprocessing
import models

# reload the files every time
import importlib
importlib.reload(mimic_diagnoses)
importlib.reload(preprocessing)
importlib.reload(models)

patients = pd.read_csv('mimic_data/PATIENTS.csv')
admissions = pd.read_csv('mimic_data/ADMISSIONS.csv')
diagnoses = pd.read_csv('mimic_data/DIAGNOSES_ICD.csv')

admissions_leadii, patients_leadii, diagnoses_leadii = mimic_diagnoses.get_leadii_dataframes(patients, admissions, diagnoses)
ethnicities_leadii = admissions_leadii.drop_duplicates(subset='SUBJECT_ID')
ethnicities_leadii['ETHNICITY_COND'] = ethnicities_leadii.ETHNICITY.apply(lambda x: mimic_diagnoses.get_race(x))

mimic_diagnoses.add_icd_10_code_to_diagnoses(diagnoses_leadii)

# getting features and labels for three different scenarios
data_dictionary = preprocessing.get_aligned_ss_and_bp(admissions_leadii)
patient_ids = np.array([tup[0] for tup in list(data_dictionary.keys())])
start_before_sleep_arrays = [preprocessing.get_start_before_sleep(bp_ss) for bp_ss in list(data_dictionary.values())]

start_before_sleep_arrays_bp_meaned = []
for arr in start_before_sleep_arrays:
    sbp = arr[0]
    dbp = arr[1]
    ss = arr[2]
    try:
        sbp_means = np.pad(np.apply_along_axis(lambda x: np.mean(x), 1, sliding_window_view(sbp, 60)), (30, 29), 'edge')
        dbp_means = np.pad(np.apply_along_axis(lambda x: np.mean(x), 1, sliding_window_view(dbp, 60)), (30, 29), 'edge')
    except:
        sbp_means = sbp
        dbp_means = dbp

    arr = np.concatenate((sbp_means.reshape((1,-1)),dbp_means.reshape((1,-1)),ss.reshape((1,-1))),axis=0)
    start_before_sleep_arrays_bp_meaned += [arr]

start_before_sleep_arrays_m0 = preprocessing.mean_zero(start_before_sleep_arrays)
start_before_sleep_arrays_bp_meaned_m0 = preprocessing.mean_zero(start_before_sleep_arrays_bp_meaned)

print("got sleep starts")
X_sum, y_sum, patient_ids_sum = preprocessing.get_features(patient_ids, start_before_sleep_arrays_m0, True, True, False, patients, admissions, diagnoses_leadii, 8)
print("sum")
X_sum_dem, y_sum_dem, patient_ids_sum_dem = preprocessing.get_features(patient_ids, start_before_sleep_arrays_m0, True, True, True, patients, admissions, diagnoses_leadii, 8)
print("sum dem")
X_ts, y_ts, patient_ids_ts = preprocessing.get_features(patient_ids, start_before_sleep_arrays_m0, False, True, False, patients, admissions, diagnoses_leadii, 6, 6)
print("ts")

# means
X_sum_meaned, y_sum_meaned, patient_ids_sum_meaned = preprocessing.get_features(patient_ids, start_before_sleep_arrays_bp_meaned_m0, True, True, False, patients, admissions, diagnoses_leadii, 8)
print("sum")
X_sum_dem_meaned, y_sum_dem_meaned, patient_ids_sum_dem_meaned = preprocessing.get_features(patient_ids, start_before_sleep_arrays_bp_meaned_m0, True, True, True, patients, admissions, diagnoses_leadii, 8)
print("sum dem")
X_ts_meaned, y_ts_meaned, patient_ids_ts_meaned = preprocessing.get_features(patient_ids, start_before_sleep_arrays_bp_meaned_m0, False, True, False, patients, admissions, diagnoses_leadii, 6, 6)
print("ts")

# calculate features
X_ts_feats = models.calculate_features(X_ts)
X_ts_meaned_feats = models.calculate_features(X_ts_meaned)

# SMOTE APPLICATION
sm = SMOTE(sampling_strategy=1.0)
X_smote, y_smote = sm.fit_resample(X_ts_feats.dropna(axis=1, inplace=False), y_ts)
X_smote_m, y_smote_m = sm.fit_resample(X_ts_meaned_feats.dropna(axis=1, inplace=False), y_ts_meaned)

supervised_accuracies_50_ts, all_models_dict_50_ts = models.get_selected_features_and_scores_over_n_runs(10, X_ts_feats, y_ts, 0.5)
supervised_accuracies_80_ts, all_models_dict_80_ts = models.get_selected_features_and_scores_over_n_runs(10, X_ts_feats, y_ts, 0.8)

# supervised_accuracies_20, all_models_dict_20 = models.get_selected_features_and_scores_over_n_runs(10, X_smote, y_smote, 0.2)
supervised_accuracies_50, all_models_dict_50 = models.get_selected_features_and_scores_over_n_runs(10, X_smote, y_smote, 0.5)
supervised_accuracies_80, all_models_dict_80 = models.get_selected_features_and_scores_over_n_runs(10, X_smote, y_smote, 0.8)

supervised_accuracies_m_20, all_models_dict_m_20 = models.get_selected_features_and_scores_over_n_runs(10, X_smote_m, y_smote_m, 0.2)
supervised_accuracies_m_50, all_models_dict_m_50 = models.get_selected_features_and_scores_over_n_runs(10, X_smote_m, y_smote_m, 0.5)
supervised_accuracies_m_80, all_models_dict_m_80 = models.get_selected_features_and_scores_over_n_runs(10, X_smote_m, y_smote_m, 0.8)

unsupervised_accuracies_abs = compare_averaged_unsupervised_clustering(X_smote_abs, y_smote_abs, 10)
supervised_accuracies_abs_50, all_models_dict_abs_50 = models.get_selected_features_and_scores_over_n_runs(10, X_smote_abs, y_smote_abs, 0.5)



supervised_accuracies, all_models_dict = models.get_selected_features_and_scores_over_n_runs(10, X_ts_feats, y_ts, 0.5)


# MESA
data_dictionary = preprocessing.get_aligned_ss_and_bp()
patient_ids = np.array(list(data_dictionary.keys()))
start_before_sleep_arrays = [preprocessing.get_start_before_sleep(bp_ss,0) for bp_ss in list(data_dictionary.values())]

# fix out unrealistic blood pressure values
start_before_sleep_arrays_cap = []
for arr in start_before_sleep_arrays:
    sbp = arr[0]
    dbp = arr[1]
    ss = arr[2]
    
    sbp_capped = np.array([val if val<=180 else 180 for val in sbp])
    sbp_capped = np.array([val if val >= 80 else 80 for val in sbp_capped])
    dbp_capped = np.array([val if val >= 40 else 40 for val in dbp])
    dbp_capped = np.array([val if val <= 120 else 120 for val in dbp_capped])

    arr = np.concatenate((sbp_capped.reshape((1, -1)), dbp_capped.reshape((1, -1)), ss.reshape((1, -1))), axis=0)
    start_before_sleep_arrays_cap += [arr]

X_ts_6, patient_ids_ts_6 = preprocessing.get_features(patient_ids, start_before_sleep_arrays_cap, 'MESA', False, False, False, None, None, None, 6, 6)

X_ts_6_m0 = []
for arr in X_ts_6:
    s_mean = np.mean(arr[0],keepdims=True)
    d_mean = np.mean(arr[1],keepdims=True)

    arr = np.concatenate(((arr[0] - s_mean).reshape((1, -1)), (arr[1] - d_mean).reshape((1, -1)), arr[2].reshape((1, -1))), axis=0)
    X_ts_6_m0 += [arr]
X_ts_6_m0 = np.array(X_ts_6_m0)

X_ts_6_m0_no = np.concatenate((X_ts_6_m0[:43],X_ts_6_m0[44:]))

X_ts_6_m0_no_feats = models.calculate_features(X_ts_6_m0_no)

n_clusters = 2
# known_dif = []
model_type = 'KMeans'
transform_type = 'robust'
y_preds = []
# for i in range(10):
for model_type in ['Hierarchical', 'KMeans', 'Spectral']:
    for transform_type in ['std', 'minmax', 'robust']:
        # Feature selection
        context = {'model_type': model_type, 'transform_type': transform_type}
        top_feats = feature_selection(X_ts_6_m0_rmed_feats, labels={}, context=context)
        print("selected features")
        print(len(top_feats))
        df_feats = X_ts_6_m0_rmed_feats[top_feats]
        
        # Clustering
        model = ClusterWrapper(n_clusters=n_clusters,model_type=model_type, transform_type=transform_type)
        y_pred = model.fit_predict(df_feats)
        # mode = stats.mode(y_pred)[0]
        # known_dif += [mode!=y_pred[43]]
        print(y_pred)
        y_preds.append(y_pred)

y_preds = np.array(y_preds)

# print(np.mean(known_dif))

for i in range(20):
    plt.plot(X_ts_6_m0[i][0])
    plt.plot(X_ts_6_m0[i][0])
    plt.show()