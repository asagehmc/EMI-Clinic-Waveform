import pandas as pd
import numpy as np

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
print("got sleep starts")
X_sum, y_sum, patient_ids_sum = preprocessing.get_features(patient_ids, start_before_sleep_arrays, True, True, False, patients, admissions, diagnoses_leadii, 8)
print("sum")
X_sum_dem, y_sum_dem, patient_ids_sum_dem = preprocessing.get_features(patient_ids, start_before_sleep_arrays, True, True, True, patients, admissions, diagnoses_leadii, 8)
print("sum dem")
X_ts, y_ts, patient_ids_ts = preprocessing.get_features(patient_ids, start_before_sleep_arrays, False, True, False, patients, admissions, diagnoses_leadii, 6, 6)
print("ts")

# compare averages over
models.compare_averages_summary_models(X_sum,y_sum)
