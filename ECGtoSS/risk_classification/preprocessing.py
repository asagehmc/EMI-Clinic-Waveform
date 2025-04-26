"""
Filename: preprocessing.py

Description: This file contains functions for getting the data formatted into features to be inputted into models
"""

import os
import json
import numpy as np
from dask.array import piecewise
from jinja2.filters import sync_do_sum
from numpy.lib.stride_tricks import sliding_window_view
from requests.compat import JSONDecodeError
from scipy import stats
import pandas as pd
import importlib

import mimic_diagnoses
importlib.reload(mimic_diagnoses)

def get_aligned_ss_and_bp_one_instance(patient_data_path, pipeline='MESA',patient_id=None,admissions=None):
    """
    :param patient_data_path: str, path to patient data
    :param pipeline: string, default 'MESA', either 'MESA' or 'MIMIC'
    :param patient_id: str, default None, patient id, only necessary if pipeline='MIMIC'
    :param admissions: pd df, default None, admissions.csv pd df, only necessary if pipeline='MIMIC'
    :return: np array, 3 by number of samples, first row is SBP, second row is DBP, third is sleep stages
    """
    # loads data
    try:
        patient_data = json.load(open(patient_data_path))
    except JSONDecodeError:
        # no data in file
        return []
    except FileNotFoundError:
        # file doesn't exist
        return []

    if pipeline=='MIMIC':
        # gets datetime str of the data measurement and cuts off decimal time
        date = patient_data['date'][:19]
        # checks if patient died during stay where data was taken
        died = mimic_diagnoses.check_if_died_during_admission(int(patient_id), date, admissions)
        if died:
            return []

    # no data samples
    if len(np.array(patient_data['sleep_stages'])) == 0:
        return []

    if pipeline=='MIMIC':
        # calculate sleep stages from probabilities
        ss = np.argmax(np.array(patient_data['sleep_stages']), axis=1)
    elif pipeline=='MESA':
        ss = np.array(patient_data['sleep_stages'])

    # sbp and dbp arrays
    sbp = np.array(patient_data['systolic'])
    dbp = np.array(patient_data['diastolic'])

    # concatenate the three signals together
    try:
        bp_ss = np.concatenate((sbp.reshape((1, -1)), dbp.reshape((1, -1)), ss.reshape((1, -1))), axis=0)
    # if they are not the same length
    except ValueError:
        new_length = min(len(sbp), len(dbp), len(ss))
        sbp = sbp[:new_length]
        dbp = dbp[:new_length]
        ss = ss[:new_length]

        bp_ss = np.concatenate((sbp.reshape((1, -1)), dbp.reshape((1, -1)), ss.reshape((1, -1))), axis=0)

    return bp_ss

def get_aligned_ss_and_bp(pipeline='MESA', admissions=None):
    """
    :param pipeline: str, default 'MESA', either 'MESA' or 'MIMIC'
    :param admissions: pd df, default None, admissions.csv pd df, only necessary if pipeline='MIMIC'
    :return: dictionary, for each patient (key), gets the aligned SS and BPs (value)
    """
    """
    for each patient, gets the aligned SS and BP by taking the highest probability
    of sleep stage
    :return:
    """
    data_dictionary = {}
    if pipeline=="MIMIC":
        sleep_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(sleep_dir_path, 'data', 'mimic_aligned_data')

        # find all patient data files
        for patient_subset in os.listdir(data_path):
            subset_path = os.path.join(data_path, patient_subset)
            if not os.path.isdir(subset_path):
                continue
            for patient in os.listdir(subset_path):
                patient_path = os.path.join(subset_path, patient)
                if not os.path.isdir(patient_path) or patient[0] != 'p':
                    continue
                patient_id = patient.split('p')[1]

                i = 0
                for ts in os.listdir(patient_path):
                    patient_data_path = os.path.join(patient_path, ts)

                    # gets the aligned bp_ss array for that instance
                    bp_ss = get_aligned_ss_and_bp_one_instance(patient_data_path, 'MIMIC', patient_id, admissions)
                    if len(bp_ss) == 0:
                        # doesn't save if no data there
                        continue
                    else:
                        # because a patient can have more than one instance, keeps track of which one
                        data_dictionary[(patient_id, i)] = bp_ss
                        i += 1

    elif pipeline=="MESA":
        sleep_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(sleep_dir_path, 'data', 'mesa_aligned_data')

        # gets all patient files
        for patient in os.listdir(data_path):
            patient_data_path = os.path.join(data_path, patient)
            patient_id = patient.split('.')[0][-4:]

            # gets the aligned bp_ss array for that instance
            bp_ss = get_aligned_ss_and_bp_one_instance(patient_data_path, 'MESA')
            if len(bp_ss) == 0:
                # doesn't save if no data there
                continue
            else:
                # only has one instance per patient, so just records the id
                data_dictionary[patient_id] = bp_ss

    return data_dictionary

def get_summary_stats_for_instance(bp_ss, demographics, patient_id, min_num_hours, patients, admissions):
    """
    :param bp_ss: np array, 3 by number of samples, first row is SBP, second row is DBP, third is sleep stages
    :param demographics: bool, True if including age + sex in features, False otherwise
    :param patient_id: str, patient id
    :param min_num_hours: int, minimum number of hours of data to include as a feature
    :param patients: pd df, patients.csv pd df
    :param admissions: pd df, admissions.csv pd df
    :return: np array, array of nans if not enough data, or array summary stats with or without demographics
    """
    # checks length of data
    if len(bp_ss[0]) < min_num_hours*60*2:
        if demographics:
            return np.array([np.nan, np.nan, np.nan, np.nan,np.nan,np.nan,np.nan,np.nan])
        else:
            return np.array([np.nan,np.nan,np.nan,np.nan,np.nan,np.nan])

    sbp = bp_ss[0]
    dbp = bp_ss[1]
    ss = bp_ss[2]

    # bp summary stats
    sbp_mean = float(np.nanmean(sbp))
    dbp_mean = float(np.nanmean(dbp))
    bp_range = int(np.nanmax(sbp) - np.nanmin(sbp))

    # sleep stage summary stats
    uniques, counts = np.unique(ss, return_counts=True)
    percentages = dict(zip(uniques, counts / len(ss)))
    if 1 in percentages.keys():
        ss_1 = percentages[1]
    else:
        ss_1 = 0
    if 2 in percentages.keys():
        ss_2 = percentages[2]
    else:
        ss_2 = 0
    if 3 in percentages.keys():
        ss_3 = percentages[3]
    else:
        ss_3 = 0

    # demographics handling
    if demographics:
        age, sex = mimic_diagnoses.basic_info(int(patient_id),patients,admissions)
        if 'F' in sex:
            sex = 0
        else:
            sex = 1
        return np.array([sbp_mean, dbp_mean, bp_range, ss_1, ss_2, ss_3, age, sex])
    else:
        return np.array([sbp_mean, dbp_mean, bp_range, ss_1, ss_2, ss_3])


def get_summary_features(patient_ids, start_before_sleep_arrays, labels, demographics, min_num_hours, patients, admissions, diagnoses):
    """
    gets summary statistics features for all data
    :param start_before_sleep_arrays: list of np arrays, bp_ss starting after sleep start times
    :param labels: bool, if getting labels or not
    :param demographics: bool, if getting demographics or not
    :param min_num_hours: int, minimum number of hours of data to include as a feature
    :param patients: pd df, patients.csv pd df
    :param admissions: pd df, admissions.csv pd df
    :param diagnoses: pd df, diagnoses.csv pd df
    :return: X, np array, summary statistics features
             new_patient_ids, list of strings, list of patient ids corresponding to instances in features
    """
    X = np.array([get_summary_stats_for_instance(bp_ss,demographics,id,min_num_hours,patients,admissions) for id,bp_ss in zip(patient_ids,start_before_sleep_arrays)])
    nan_mask = np.isnan(X).any(axis=1)
    X = X[~nan_mask]

    new_patient_ids = np.array(patient_ids)[~nan_mask]
    if labels:
        y = mimic_diagnoses.get_patient_labels(patient_ids,diagnoses)
        y = y[~nan_mask]
        return X,y, new_patient_ids
    else:
        return X, new_patient_ids


def pad_list_of_arrays(list_of_arrays, max_length, padding_value=0):
    """
    pads a list of 2d np arrays to a length
    :param list_of_arrays: list of 2d arrays
    :param max_length: int, length to pad to
    :param padding_value: int, value to pad with
    :return: 3d npy array, array of padded to same length arrays
    """
    padded_arrays = [
        np.pad(arr, ((0,0),(0,max_length-arr.shape[1])), 'constant', constant_values=padding_value) if max_length-arr.shape[1] >=0 else arr[:,:max_length]
        for arr in list_of_arrays
    ]
    return np.array(padded_arrays)


def get_start_before_sleep(bp_ss, awake_int):
    """
    gets sleep start time and then returns arrays values starting then
    :param bp_ss: 2d np array, 3 by number of samples, first row is SBP, second row is DBP
    :param awake_int: int, integer that corresponds to awake, 0 for MESA, 3 for MIMIC
    :return:
    """
    sbp = bp_ss[0]
    dbp = bp_ss[1]
    ss = bp_ss[2]

    rolling_mode = np.pad(np.apply_along_axis(lambda x: stats.mode(x)[0], 1, sliding_window_view(ss, 30)),(15,14) , 'edge')

    try:
        # first place where mode over 15 minutes goes from awake to sleep
        sleep_start = np.where(rolling_mode!=awake_int)[0][0] - 15

        if len(ss[sleep_start:])>=6*60*2:
            end = sleep_start+6*60*2
        else:
            end = sleep_start + len(ss[sleep_start:])

        # checks for mostly sleep
        if np.sum(ss[sleep_start:end] == awake_int) / len(ss[sleep_start:end]) >= 0.5:
            #print("not sleep")
            return np.array([[], [], []])
        #print(sleep_start)
    except IndexError:
        # no sleep in data
        #print("no sleep start")
        return np.array([[],[],[]])

    after_sleep_bp_ss = np.concatenate((sbp[sleep_start:].reshape(1,-1), dbp[sleep_start:].reshape(1,-1), ss[sleep_start:].reshape(1,-1)), axis=0)

    return after_sleep_bp_ss

def mean_zero(start_before_sleep_arrays):
    """
    :param start_before_sleep_arrays: list of np arrays, bp_ss's starting after sleep start times
    :return: list of np arrays, bp_ss's mean zeroed starting after sleep start times
    """
    # find means of sbp and dbp and subtract from values
    start_before_sleep_arrays_m0 = []
    for arr in start_before_sleep_arrays:
        sbp_mean = np.nanmean(arr[0])
        dbp_mean = np.nanmean(arr[1])
        start_before_sleep_arrays_m0 += [np.array([[val - sbp_mean for val in arr[0]],
                                                   [val - dbp_mean for val in arr[1]],
                                                   arr[2]])]

    return start_before_sleep_arrays_m0


def get_time_series_features(patient_ids, start_before_sleep_arrays, labels, min_num_hours, fixed_block_hours,diagnoses):
    """
    :param patient_ids: list of strs, patient_ids
    :param start_before_sleep_arrays: list of np arrays, bp_ss's starting after sleep start times
    :param labels: bool, whether to include labels or not
    :param min_num_hours: int, minimum number of hours to include
    :param fixed_block_hours: int, number of hours to zero pad to, must be greater than or equal to min_num_hours
    :param diagnoses: pd df, diagnoses.csv pd df
    :return: 3d np array, bp_ss's time series features'
    """
    # checks constraint
    if min_num_hours > fixed_block_hours:
        return "fixed_block_hours must be greater than or equal to min_num_hours"

    
    has_min_hours = np.array([len(arr[0]) for arr in start_before_sleep_arrays]) > min_num_hours*60*2
    # zero pad / cut off for a minimum hour block
    X = pad_list_of_arrays(start_before_sleep_arrays, fixed_block_hours*60*2)
    X = X[has_min_hours]

    # drops instances with nan values
    nan_mask = np.isnan(X).any(axis=(1,2))
    X = X[~nan_mask]

    new_patient_ids = np.array(patient_ids)[has_min_hours][~nan_mask]

    # labels handling
    if labels:
        y = mimic_diagnoses.get_patient_labels(patient_ids,diagnoses)
        return X,y[has_min_hours][~nan_mask], new_patient_ids
    else:
        return X, new_patient_ids

def get_features(patient_ids, start_before_sleep_arrays, pipeline, summary, labels, demographics, patients=None, admissions=None, diagnoses=None, min_num_hours=8, fixed_block_hours=8):
    """
    :param patient_ids: list of strs, list of patient ids
    :param start_before_sleep_arrays: list of np arrays, bp_ss's starting after sleep start times
    :param pipeline: str, either 'MESA' or 'MIMIC'
    :param summary: bool, whether to do summary statistics or not
    :param labels: bool, whether to include labels or not
    :param demographics: bool, whether to include demographics or not for summary statistics
    :param patients: pd df, patients.csv pd df
    :param admissions: pd df, admissions.csv pd df
    :param diagnoses: pd df, diagnoses.csv pd df
    :param min_num_hours: int, minimum number of hours to include
    :param fixed_block_hours: int, number of hours to zero pad to, must be greater than or equal to min_num_hours
    :return: 2d or 3d np arrays, features to input into a model
    """
    if summary:
        features = get_summary_features(patient_ids, start_before_sleep_arrays, labels, demographics, min_num_hours, patients, admissions, diagnoses)
    else:
        if pipeline == 'MIMIC':
            features = get_time_series_features(patient_ids, start_before_sleep_arrays, labels, min_num_hours, fixed_block_hours, diagnoses)
        elif pipeline == 'MESA':
            features = get_time_series_features(patient_ids, start_before_sleep_arrays, False, min_num_hours,fixed_block_hours, None)

    if labels:
        # features, labels, corresponding patient ids
        return features[0], features[1], features[2]
    else:
        # features, corresponding patient ids
        return features[0], features[1]
    
def load_preprocessing_data():
    """
    Loads and returns preprocessed data.

    This function:
      - Loads admissions data (filtered for lead II) via load_admissions_leadii().
      - Loads additional DataFrames: patients and diagnoses.
      - Scans the patient directories to obtain aligned BP/SS arrays.
      - Computes summary statistic features (with and without demographics) and
        uniformly padded time-series features.
      - Produces binary risk labels using diagnoses information.

    Returns:
        Tuple: (X_sum, y_sum, X_sum_dem, y_sum_dem, X_ts, y_ts)
    """
    from mimic_diagnoses import load_admissions_leadii, add_icd_10_code_to_diagnoses
    admissions_leadii = load_admissions_leadii()

    # Load the additional necessary DataFrames.
    patients = pd.read_csv(
        '/Users/lydiastone/PycharmProjects/EIT-Clinic-Waveform/sleep/risk_classification/mimic_data/PATIENTS.csv')
    admissions = pd.read_csv(
        '/Users/lydiastone/PycharmProjects/EIT-Clinic-Waveform/sleep/risk_classification/mimic_data/ADMISSIONS.csv')
    diagnoses = pd.read_csv(
        '/Users/lydiastone/PycharmProjects/EIT-Clinic-Waveform/sleep/risk_classification/mimic_data/DIAGNOSES_ICD.csv')

    # Ensure that the diagnoses DataFrame gets the ICD10_CODE column.
    add_icd_10_code_to_diagnoses(diagnoses)

    # Scan directories for aligned BP/SS data.
    data_dictionary = get_aligned_ss_and_bp(admissions_leadii)
    patient_ids = [tup[0] for tup in list(data_dictionary.keys())]
    start_before_sleep_arrays = [get_start_before_sleep(bp_ss) for bp_ss in list(data_dictionary.values())]

    # Compute features.
    # For summary features with labels, get_features() returns three items (features, labels, and patient IDs)
    X_sum, y_sum, _ = get_features(patient_ids, start_before_sleep_arrays, True, True, False, patients, admissions,
                                   diagnoses)
    X_sum_dem, y_sum_dem, _ = get_features(patient_ids, start_before_sleep_arrays, True, True, True, patients,
                                           admissions, diagnoses)
    # For time-series features (assuming labels=True), we also expect three items or if you want only features and labels.
    X_ts, y_ts, _ = get_features(patient_ids, start_before_sleep_arrays, False, True, False, patients, admissions,
                                 diagnoses)

    return X_sum, y_sum, X_sum_dem, y_sum_dem, X_ts, y_ts