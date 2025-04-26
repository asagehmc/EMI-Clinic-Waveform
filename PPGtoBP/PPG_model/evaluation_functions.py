import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import PPGtoBP.PPG_model.calibration as cf
import pandas as pd
import PPGtoBP.PPG_model.helper_functions_bp_model as hf
import neurokit2 as nk
from scipy.signal import find_peaks, correlate
import math


def truncate_to_match(*arrays):
    """
    Truncates episodes down to the same length for comparison and visualization purposes
    :param arrays: the arrays of waveforms we want to process
    :return: the arrays truncated to the same length
    """

    min_len = min(len(arr) for arr in arrays)
    return [arr[:min_len] for arr in arrays]


def bhs_grading(actual, pred):
    """
    Calculates the bhs grade for a given set of true and predicted waveform peaks
    :param actual: the actual waveform peaks
    :param pred: the predicted waveform peaks
    :return: the BHS metrics for the given set of actual and predicted peaks as a dictionary
    """

    errors = np.abs(np.array(actual) - np.array(pred))
    total = len(errors)

    pct_within_5 = np.sum(errors <= 5) / total * 100
    pct_within_10 = np.sum(errors <= 10) / total * 100
    pct_within_15 = np.sum(errors <= 15) / total * 100

    if pct_within_5 >= 60 and pct_within_10 >= 85 and pct_within_15 >= 95:
        grade = 'A'
    elif pct_within_5 >= 50 and pct_within_10 >= 75 and pct_within_15 >= 90:
        grade = 'B'
    elif pct_within_5 >= 40 and pct_within_10 >= 65 and pct_within_15 >= 85:
        grade = 'C'
    else:
        grade = 'D'

    return {
        '≤5 mmHg': pct_within_5,
        '≤10 mmHg': pct_within_10,
        '≤15 mmHg': pct_within_15,
        'BHS Grade': grade
    }

def evaluate_all_bp(actual_sbp, pred_sbp, actual_dbp, pred_dbp):
    """
    Calculates the BHS metrics for all portions of the waveform; DBP, SBP and MBP
    :param actual_sbp: the actual systolic blood pressure waveform peaks
    :param pred_sbp: the predicted systolic blood pressure waveform peaks
    :param actual_dbp: the actual diastolic blood pressure waveform peaks
    :param pred_dbp: the predicted diastolic blood pressure waveform peaks
    :return: a dictionary containing all the BHS grades and metrics
    """

    actual_sbp, pred_sbp, actual_dbp, pred_dbp = truncate_to_match(actual_sbp, pred_sbp,
                                                                       actual_dbp, pred_dbp)

    # Calculate MBP
    actual_mbp = np.array(actual_dbp) + (np.array(actual_sbp) - np.array(actual_dbp)) / 3
    pred_mbp = np.array(pred_dbp) + (np.array(pred_sbp) - np.array(pred_dbp)) / 3

    return {
        'SBP': bhs_grading(actual_sbp, pred_sbp),
        'DBP': bhs_grading(actual_dbp, pred_dbp),
        'MBP': bhs_grading(actual_mbp, pred_mbp)
    }


def format_bhs_results_table(bhs_results):
    """
    Formats the given BHS results to a printable table
    :param bhs_results: the dictionary of BHS results
    """

    # Create DataFrame from nested dict
    df = pd.DataFrame(bhs_results).T  # Transpose so SBP/DBP/MBP are rows
    df = df[['≤5 mmHg', '≤10 mmHg', '≤15 mmHg', 'BHS Grade']]  # Order columns

    # Round the percentage values to 2 decimal places
    df[['≤5 mmHg', '≤10 mmHg', '≤15 mmHg']] = df[['≤5 mmHg', '≤10 mmHg', '≤15 mmHg']].round(2)

    print(df.to_markdown())  # Pretty-print as a markdown table


def eval_bhs_standard_dict(bp_actual, bp_predicted, ecg_signal, fs=125):
    """
    Calculates the BHS metrics for a given set of true and predicted waveform data and returns the BHS metrics data as a
    dictionary
    :param bp_actual: the ground truth BP waveform
    :param bp_predicted: the predicted BP waveform
    :param ecg_signal: the ECG signal (needed to getting heart rate)
    :param fs: sampling frequency (get this from the dataset, for MIMIC it is 125)
    :return: the dictionary containing the BHS metrics data
    """
    _, rr_ints, _, _ = cf.compute_heart_rate_from_ecg(ecg_signal, fs=fs)
    sys_peaks_actual, dias_peaks_actual = cf.get_peaks(bp_actual, distance=int(np.mean(rr_ints) * fs))
    sys_peaks_pred, dias_peaks_pred = cf.get_peaks(bp_predicted, distance=int(np.mean(rr_ints) * fs))

    sys_peaks_actual = bp_actual[sys_peaks_actual]
    sys_peaks_pred = bp_predicted[sys_peaks_pred]
    dias_peaks_actual = bp_actual[dias_peaks_actual]
    dias_peaks_pred = bp_predicted[dias_peaks_pred]
    return evaluate_all_bp(sys_peaks_actual, sys_peaks_pred, dias_peaks_actual, dias_peaks_pred)


def eval_bhs_standard(bp_actual, bp_predicted, ecg_signal, fs=125):
    """
    Calculates the BHS metrics for a given set of true and predicted waveform data and prints the markdown containing
    the BHS metric data for the prediction
    :param bp_actual: the ground truth BP waveform
    :param bp_predicted: the predicted BP waveform
    :param ecg_signal: the ECG signal (needed to getting heart rate)
    :param fs: sampling frequency (get this from the dataset, for MIMIC it is 125)
    """

    format_bhs_results_table(eval_bhs_standard_dict(bp_actual, bp_predicted, ecg_signal, fs))


def get_signal_quality_metrics(patient_list, episode_count=10):
    ppg_qualities = []  # use a regular list for collecting values

    for patient in patient_list:
        try:
            patient_ppg, _, _ = hf.get_patient_episode_data(patient, episode_count=episode_count)

            ppg_quality = nk.ppg_quality(patient_ppg, sampling_rate=125, method="templatematch")
            ppg_quality_score = np.round(np.mean(ppg_quality) * 100, 2)  # mean quality score per patient
            ppg_qualities.append(ppg_quality_score)
        except Exception as e:
            continue

    ppg_qualities = np.array(ppg_qualities)  # convert to NumPy array for analysis

    ppg_mean = np.round(np.mean(ppg_qualities), 2)
    ppg_median = np.round(np.median(ppg_qualities), 2)
    ppg_std = np.round(np.std(ppg_qualities), 2)
    ppg_min = np.round(np.min(ppg_qualities), 2)
    ppg_max = np.round(np.max(ppg_qualities), 2)
    ppg_range = np.round(ppg_max - ppg_min, 2)

    ppg_grade_counts = {
        "A": int(np.sum(ppg_qualities >= 90)),
        "B": int(np.sum((ppg_qualities >= 75) & (ppg_qualities < 90))),
        "C": int(np.sum((ppg_qualities >= 60) & (ppg_qualities < 75))),
        "D": int(np.sum((ppg_qualities >= 40) & (ppg_qualities < 60))),
        "F": int(np.sum(ppg_qualities < 40)),
    }

    return ppg_mean, ppg_median, ppg_std, ppg_min, ppg_max, ppg_range, ppg_grade_counts


