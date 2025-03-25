import os
import json
import numpy as np
from jinja2.filters import sync_do_sum
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats

from mimic_diagnoses import get_patient_labels, basic_info

def get_aligned_ss_and_bp():
    """
    for each patient, gets the aligned SS and BP by taking the highest probability
    of sleep stage
    :return:
    """
    data_path = "/Users/lydiastone/PycharmProjects/EIT-Clinic-Waveform/sleep/data/patients"

    data_dictionary = {}
    for patient_subset in os.listdir(data_path):
        subset_path = os.path.join(data_path, patient_subset)
        for patient in os.listdir(subset_path):
            patient_path = os.path.join(subset_path, patient)
            patient_data_path = os.path.join(patient_path,os.listdir(patient_path)[0])

            patient_data = json.load(open(patient_data_path))

            bp = np.array(patient_data['blood_pressure'])
            ss = np.argmax(np.array(patient_data['sleep_stages']),axis=1)
            try:
                bp_ss = np.concatenate((bp.reshape((1, -1)), ss.reshape((1, -1))), axis=0)

            # if they are off by one
            except ValueError:
                new_length = min(len(bp), len(ss))
                bp = bp[:new_length]
                ss = ss[:new_length]

                bp_ss = np.concatenate((bp.reshape((1, -1)), ss.reshape((1, -1))), axis=0)

            patient_id = patient.split('p')[1]

            data_dictionary[patient_id] = bp_ss

    return data_dictionary

def get_summary_stats_for_instance(bp_ss, demographics, patient_id=None):
    """
    returns summary statistics features for one instance
    :param bp_ss:  2d np array [[bp values],[ss values]]
    :param demographics:
    :param patient_id:
    :return:
    """
    bp = bp_ss[0]
    ss = bp_ss[1]

    bp_mean = float(np.mean(bp))
    bp_range = max(bp) - min(bp)

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

    if demographics:
        age, sex = basic_info(int(patient_id))
        sex = 0 if sex == 'F' else 1
        return np.array([bp_mean, bp_range, ss_1, ss_2, ss_3, age, sex])
    else:
        return np.array([bp_mean, bp_range, ss_1, ss_2, ss_3])


def get_summary_features(data_dictionary, labels, demographics):
    """
    gets summary statistics features for all data
    :param data_dictionary: str patient_ids keys and 2d np array [[bp values],[ss values]] values
    :param labels: bool, if getting labels or not
    :param demographics: bool if getting demographics or not
    :return:
    """
    patient_ids = list(data_dictionary.keys())
    patient_data = list(data_dictionary.values())

    X = np.array([get_summary_stats_for_instance(bp_ss,demographics,id) for id,bp_ss in zip(patient_ids,patient_data)])

    if labels:
        y = get_patient_labels(patient_ids)
        return X,y
    else:
        return X


def pad_list_of_arrays(list_of_arrays, max_length, padding_value=0):
    """
    pads a list of 2d np arrays to a length
    :param list_of_arrays: list of 2d arrays
    :param max_length: length to pad to
    :param padding_value:
    :return:
    """
    """Pads a list of numpy arrays to the same length.

    Args:
        list_of_arrays: A list of numpy arrays.
        padding_value: The value to use for padding (default is 0).

    Returns:
        A numpy array containing the padded arrays.
    """
    padded_arrays = [
        np.pad(arr, ((0,0),(0,max_length-arr.shape[1])), 'constant', constant_values=padding_value) if max_length-arr.shape[1] >=0 else arr[:,:max_length]
        for arr in list_of_arrays
    ]
    return np.array(padded_arrays)


def get_start_before_sleep(bp_ss):
    """
    gets sleep start time and then returns arrays values starting then
    :param bp_ss:
    :return:
    """
    bp = bp_ss[0]
    ss = bp_ss[1]

    awake = np.pad(np.apply_along_axis(lambda x: stats.mode(x)[0]==3, 1, sliding_window_view(ss, 30)),(15,14) , 'edge')

    try:
        sleep_start = np.where(awake==False)[0][0] - 15
    except IndexError:
        # no sleep in data
        return np.array([[],[]])

    after_sleep_bp_ss = np.concatenate((bp[sleep_start:].reshape(1,-1), ss[sleep_start:].reshape(1,-1)), axis=0)

    return after_sleep_bp_ss


def get_time_series_features(data_dictionary, labels):
    """
    gets time series features for all data
    :param data_dictionary:
    :param labels:
    :return:
    """
    patient_ids = list(data_dictionary.keys())
    patient_data = list(data_dictionary.values())

    start_before_sleep = [get_start_before_sleep(bp_ss) for bp_ss in patient_data]

    # zero pad for an 8 hour block
    X = pad_list_of_arrays(start_before_sleep, 8*60*2)

    has_sleep = np.array([False if np.sum(arr == 0)==2*8*60*2 else True for arr in X])

    if labels:
        y = get_patient_labels(patient_ids)
        return X[has_sleep],y[has_sleep]
    else:
        return X[has_sleep]


def get_features(data_dictionary, summary, labels, demographics):
    """
    gets features for all data
    :param data_dictionary:
    :param summary: bool, true if summary statistics, false if time series
    :param labels: if returning labels or not
    :param demographics: if returning demographics or not
    :return:
    """
    if summary:
        features = get_summary_features(data_dictionary, labels, demographics)
    else:
        features = get_time_series_features(data_dictionary, labels)

    if labels:
        return features[0], features[1]
    else:
        return features

# getting features and labels for three different scenarios
data_dictionary = get_aligned_ss_and_bp()
X_sum, y_sum = get_features(data_dictionary, True, True, False)
X_sum_dem, y_sum_dem = get_features(data_dictionary, True, True, True)
X_ts, y_ts = get_features(data_dictionary, False, True, False)