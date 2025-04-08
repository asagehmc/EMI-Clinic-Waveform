import os
import json
import numpy as np
from jinja2.filters import sync_do_sum
from numpy.lib.stride_tricks import sliding_window_view
from requests.compat import JSONDecodeError
from scipy import stats

from mimic_diagnoses import get_patient_labels, basic_info, check_if_died_during_admission

def get_aligned_ss_and_bp_one_instance(patient_data_path,patient_id):
    """
    :param patient_data_path: str, path to patient data
    :param patient_id: str, patient id
    :return: bp_ss array for the given patient's data path if there is data and they didn't die during the stay
             else return []
    """
    try:
        patient_data = json.load(open(patient_data_path))
    except JSONDecodeError:
        # no data in file
        return []

    # gets datetime str of the data measurement and cuts off decimal time
    date = patient_data['date'][:19]
    # checks if patient died during stay where data was taken
    died = check_if_died_during_admission(int(patient_id), date)
    if died:
        return []

    bp = np.array(patient_data['blood_pressure'])
    ss = np.argmax(np.array(patient_data['sleep_stages']), axis=1)

    try:
        bp_ss = np.concatenate((bp.reshape((1, -1)), ss.reshape((1, -1))), axis=0)
    # if they are off by one
    except ValueError:
        new_length = min(len(bp), len(ss))
        bp = bp[:new_length]
        ss = ss[:new_length]

        bp_ss = np.concatenate((bp.reshape((1, -1)), ss.reshape((1, -1))), axis=0)

    return bp_ss

def get_aligned_ss_and_bp():
    """
    for each patient, gets the aligned SS and BP by taking the highest probability
    of sleep stage
    :return:
    """
    data_paths = ["/Users/lydiastone/PycharmProjects/EIT-Clinic-Waveform/sleep/data/patients","/Users/lydiastone/PycharmProjects/EIT-Clinic-Waveform/sleep/data/patients_cvd"]

    data_dictionary = {}
    for data_path in data_paths:
        for patient_subset in os.listdir(data_path):
            subset_path = os.path.join(data_path, patient_subset)
            if not os.path.isdir(subset_path):
                continue
            for patient in os.listdir(subset_path):
                patient_path = os.path.join(subset_path, patient)
                if not os.path.isdir(patient_path):
                    continue
                patient_id = patient.split('p')[1]

                i = 0
                for ts in os.listdir(patient_path):
                    patient_data_path = os.path.join(patient_path, ts)

                    bp_ss = get_aligned_ss_and_bp_one_instance(patient_data_path, patient_id)
                    if len(bp_ss) == 0:
                        continue
                    else:
                        data_dictionary[(patient_id,i)] = bp_ss
                        i += 1

    return data_dictionary

def get_summary_stats_for_instance(bp_ss, demographics, patient_id=None):
    """
    returns summary statistics features for one instance
    :param bp_ss:  2d np array [[bp values],[ss values]]
    :param demographics:
    :param patient_id:
    :return:
    """
    if len(bp_ss[0]) < 8*60*2:
        if demographics:
            return np.array([np.nan, np.nan, np.nan, np.nan,np.nan,np.nan,np.nan])
        else:
            return np.array([np.nan,np.nan,np.nan,np.nan,np.nan])

    bp = bp_ss[0]
    ss = bp_ss[1]

    bp_mean = float(np.nanmean(bp))
    bp_range = int(np.nanmax(bp) - np.nanmin(bp))

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
        if 'F' in sex:
            sex = 0
        else:
            sex = 1
        return np.array([bp_mean, bp_range, ss_1, ss_2, ss_3, age, sex])
    else:
        return np.array([bp_mean, bp_range, ss_1, ss_2, ss_3])


def get_summary_features(patient_ids, start_before_sleep_arrays, labels, demographics):
    """
    gets summary statistics features for all data
    :param data_dictionary: str patient_ids keys and 2d np array [[bp values],[ss values]] values
    :param labels: bool, if getting labels or not
    :param demographics: bool if getting demographics or not
    :return:
    """
    X = np.array([get_summary_stats_for_instance(bp_ss,demographics,id) for id,bp_ss in zip(patient_ids,start_before_sleep_arrays)])
    # mostly_sleep_mask = X[:, 4] != 1
    # X = X[mostly_sleep_mask]
    nan_mask = np.isnan(X).any(axis=1)
    X = X[~nan_mask]

    new_patient_ids = patient_ids[~nan_mask]

    if labels:
        y = get_patient_labels(patient_ids)
        # y = y[mostly_sleep_mask]
        y = y[~nan_mask]
        return X,y, new_patient_ids
    else:
        return X, new_patient_ids


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

    # np.array([np.pad(np.apply_along_axis(lambda x: np.mean(x), 1, sliding_window_view(bp, 10)), (5, 4), 'edge') for bp in bp_arrays])

    try:
        sleep_start = np.where(awake==False)[0][0] - 15
        print("found sleep start")
    except IndexError:
        # no sleep in data
        print("not sleep start")
        return np.array([[],[]])

    after_sleep_bp_ss = np.concatenate((bp[sleep_start:].reshape(1,-1), ss[sleep_start:].reshape(1,-1)), axis=0)

    return after_sleep_bp_ss


def get_time_series_features(patient_ids, start_before_sleep_arrays, labels):
    """
    gets time series features for all data
    :param data_dictionary:
    :param labels:
    :return:
    """

    has_eight_hours = np.array([len(arr[0]) for arr in start_before_sleep_arrays]) > 8*60*2

    # zero pad / cut off for an 8 hour block
    X = pad_list_of_arrays(start_before_sleep_arrays, 8*60*2)
    X = X[has_eight_hours]

    # has_sleep = np.array([False if np.sum(arr == 0)==2*8*60*2 else True for arr in X])
    # X = X[has_sleep]

    nan_mask = np.isnan(X).any(axis=(1,2))
    X = X[~nan_mask]

    new_patient_ids = patient_ids[has_eight_hours][~nan_mask]

    if labels:
        y = get_patient_labels(patient_ids)
        return X,y[has_eight_hours][~nan_mask], new_patient_ids
    else:
        return X, new_patient_ids


def get_features(patient_ids, start_before_sleep_arrays, summary, labels, demographics):
    """
    gets features for all data
    :param data_dictionary:
    :param summary: bool, true if summary statistics, false if time series
    :param labels: if returning labels or not
    :param demographics: if returning demographics or not
    :return:
    """
    if summary:
        features = get_summary_features(patient_ids, start_before_sleep_arrays, labels, demographics)
    else:
        features = get_time_series_features(patient_ids, start_before_sleep_arrays, labels)

    if labels:
        # features, labels, corresponding patient ids
        return features[0], features[1], features[2]
    else:
        # features, corresponding patient ids
        return features[0], features[1]

if __name__ == '__main__':
    # getting features and labels for three different scenarios
    data_dictionary = get_aligned_ss_and_bp()
    patient_ids = np.array([tup[0] for tup in list(data_dictionary.keys())])
    start_before_sleep_arrays = [get_start_before_sleep(bp_ss) for bp_ss in list(data_dictionary.values())]
    print("got sleep starts")
    X_sum, y_sum, patient_ids_sum = get_features(patient_ids, start_before_sleep_arrays, True, True, False)
    print("sum")
    X_sum_dem, y_sum_dem, patient_ids_sum_dem = get_features(patient_ids, start_before_sleep_arrays, True, True, True)
    print("sum dem")
    X_ts, y_ts, patient_ids_ts = get_features(patient_ids, start_before_sleep_arrays, False, True, False)
    print("ts")