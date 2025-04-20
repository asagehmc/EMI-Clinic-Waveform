import pickle
import numpy as np
import os

from PPGtoBP.PPG_model.bloodPressureModel.codes.helper_functions import *
from PPGtoBP.PPG_model.bloodPressureModel.codes.models import UNetDS64, MultiResUNet1D
import PPGtoBP.PPG_model.calibration as cf
import PPGtoBP.PPG_model.evaluation_functions as ef

# load the approximate model
approxModel = UNetDS64(1024)
approxModel.load_weights(os.path.join('PPG_model/bloodPressureModel/models', 'ApproximateNetwork.h5'))

# load the refinement model
refineModel = MultiResUNet1D(1024)
refineModel.load_weights(os.path.join('PPG_model/bloodPressureModel/models', 'RefinementNetwork.h5'))


def normalize_min_max(x):
    """
    Normalizes the ppg data so it is ready to be fed into the model
    :param x: a ppg waveform reading
    :return: a normalized ppg waveform reading
    """

    with open(f"PPG_model/bloodPressureModel/codes/data/meta9.p", "rb") as f:
        meta = pickle.load(f)
    return (x - meta["min_ppg"]) / (meta["max_ppg"] - meta["min_ppg"] + 1e-6)


def unnormalize_min_max(x):
    """
    Un-Normalizes the ppg data so it is actually helpful data
    :param x: the normalized ppg waveform reading
    :return: the un-normalized ppg waveform reading
    """

    with open(f"PPG_model/bloodPressureModel/codes/data/meta9.p", "rb") as f:
        meta = pickle.load(f)
    return x * (meta["max_abp"] - meta["min_abp"]) + meta["min_abp"]


def predict_bp_from_ppg(normalized_ppg_waveform):
    """
    Predicts the blood pressure from a ppg waveform
    :param normalized_ppg_waveform: the normalized ppg waveform we want to process
    :return predicted_bp: the predicted blood pressure waveform from the model
    """

    predicted_bp = []

    for i in range(0, len(normalized_ppg_waveform), 1024):
        chunk = normalized_ppg_waveform[i:i + 1024]
        if len(chunk) < 1024:
            continue

        input_ppg = np.expand_dims(chunk, axis=(0, 2))
        approx_output = approxModel.predict(input_ppg, verbose=0)[0].flatten()
        input_approx = np.expand_dims(approx_output, axis=(0, 2))
        refined_output = refineModel.predict(input_approx, verbose=0)[0].flatten()
        predicted_bp.append(refined_output)

    predicted_bp = np.concatenate(predicted_bp)
    predicted_bp = unnormalize_min_max(predicted_bp)

    return predicted_bp


def trim_waveform(x):
    length = len(x)
    new_length = (length // 1024) * 1024
    return x[:new_length]

def get_patient_records(patient_id):
    """
    Loads and returns the patient records corresponding to the given patient id
    :param patient_id: the id of the patient whose records should be returned
    :return: the patient records corresponding to the given patient id as a dictionary
    """

    with open(f'PPG_model/MIMIC-III-Database-Management/processed_data/{patient_id}/records.p', 'rb') as f:
        patientRecords = pickle.load(f)

    return patientRecords


def is_signal_valid(signal, flat_threshold=0.001, isBP=False):
    """
    TODO: write a better docstring here
    Returns True if the given signal is valid, False otherwise
    """

    signal = np.asarray(signal)
    if(isBP):
        if(np.min(signal) < 40) or (np.max(signal) > 200):
            return False

    if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
        return False

    for i in range(0, len(signal) - 100 + 1, 100 // 2):  # overlapping windows
        window = signal[i:i + 100]
        if np.max(window) - np.min(window) < flat_threshold:
            return False  # flat window found

    return True  # no flat segments found


def get_patient_episode_data(patient_id, episode_count):
    """
    This function gets the patient waveform data for a given patient and a given number of episodes. It automatically
    filters out episodes that contain flat line waveforms for significant periods of time and additionally filters out
    episodes that contain impossible values (e.g. negative blood pressure readings)
    :param patient_id: the id of the patient whose records we want to process
    :param episode_count: the number of episodes we want for a given patient
    :return ppg_signals, abp_signals, ecg_signals: for a specific patient if the patient has enough valid episodes to
        be processed, otherwise return None
    """
    patient_records = get_patient_records(patient_id)
    if patient_records['BP_EP_COUNT'] < episode_count:
        return None

    ppg_signals = []
    abp_signals = []
    ecg_signals = []

    episodes_processed = 0
    i = 0
    while episodes_processed < episode_count:
        with open(f'PPG_model/MIMIC-III-Database-Management/processed_data/{patient_id}/episodes/BP/episode_{i}.p', 'rb') as f:
            episode = pickle.load(f)

            if is_signal_valid(episode['ppg']) and is_signal_valid(episode['abp'], flat_threshold=20, isBP=True) and is_signal_valid(episode['ecg']):
                if len(ppg_signals) <= 0:
                    ppg_signals.append(episode['ppg'])
                    abp_signals.append(episode['abp'])
                    ecg_signals.append(episode['ecg'])

                else:
                    ppg_signals.append(episode['ppg'][625:])
                    abp_signals.append(episode['abp'][625:])
                    ecg_signals.append(episode['ecg'][625:])

                episodes_processed += 1
        i += 1

        if i >= patient_records["BP_EP_COUNT"]:
            break

    if len(ppg_signals) < episode_count:
        return None

    ppg_signals = normalize_min_max(np.concatenate(ppg_signals))
    abp_signals = np.concatenate(abp_signals)
    ecg_signals = np.concatenate(ecg_signals)

    return ppg_signals, abp_signals, ecg_signals

def get_good_bad_patient_split(patient_list, episode_count=10):
    """
    Runs the model on a list of patients for a specified episode count and in turn returns two lists of patients. 1 list
    contains the patients that received a good bhs grade and the other list contains the patients that received a bad bhs
    grade
    :param patient_list: the list of patients we want to process
    :param episode_count: the number of episodes we want to process for each patient
    :return bad_pred_patient_list: a list of patients that received a bad bhs grade (MBP BHS grade of C or D)
    :return good_pred_patient_list: a list of patients that received a good bhs grade (MBP BHS grade of A or B)
    """
    bad_pred_patient_list = []
    good_pred_patient_list = []

    subset_length = 1024

    for patient in patient_list:
        try:
            norm_ppg, actual_abp, ecg = get_patient_episode_data(patient, episode_count)
            pred_bp = predict_bp_from_ppg(norm_ppg)
            actual_bp = trim_waveform(actual_abp)

            a, b = cf.get_bp_correction(pred_bp[:subset_length], actual_bp[:subset_length], ecg)
            pred_bp_calibrated = cf.correct_bp(pred_bp, a, b)

            bhs_results = ef.eval_bhs_standard_dict(actual_bp, pred_bp_calibrated, ecg)
            if bhs_results['MBP']['BHS Grade'] == 'C' or bhs_results['MBP']['BHS Grade'] == 'D':
                bad_pred_patient_list.append(patient)
            else:
                good_pred_patient_list.append(patient)

        except Exception as e:
            continue

    return bad_pred_patient_list, good_pred_patient_list
