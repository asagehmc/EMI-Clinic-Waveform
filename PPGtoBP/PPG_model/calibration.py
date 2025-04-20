from scipy.signal import find_peaks, peak_widths
from scipy.interpolate import interp1d
import numpy as np
from scipy import stats


def compute_heart_rate_from_ecg(ecg_signal, fs=125):
    """
    Calculates heart rate (BPM) from raw ECG signal.
    :param ecg_signal: 1D numpy array of ECG waveform
    :param fs: sampling frequency in Hz (default 125 Hz for MIMIC)

    :return heart_rates: array of instantaneous heart rate values in BPM
    :return rr_intervals: array of R-R intervals in seconds
    :return r_peaks: array of indices where R-peaks were detected
    :return avg_hr: average heart rate over the segment
    """

    r_peaks, _ = find_peaks(ecg_signal, distance=int(fs * 0.4), height=np.mean(ecg_signal))
    rr_intervals = np.diff(r_peaks) / fs
    heart_rates = 60 / rr_intervals
    avg_hr = np.mean(heart_rates)

    return heart_rates, rr_intervals, r_peaks, avg_hr


def get_peaks(signal, distance=30):
    """
    Gets the systolic and diastolic peaks from a blood pressure signal
    :param signal: the blood pressure signal we want to process
    :param distance: the minimum distance between the peaks  (to avoid detecting noise as a peak)
    :return sys_peaks: an array of the locations of the systolic peaks in the waveform
    :return dias_peaks: an array of the locations of the diastolic peaks in the waveform
    """

    sys_peaks, _ = find_peaks(signal, distance=distance)
    dias_peaks, _ = find_peaks(-signal, distance=distance)
    return sys_peaks, dias_peaks


def compute_gain_offset_from_peak_heights(predicted_bp_signal, actual_bp_signal, sys_peaks_pred,
                                          dias_peaks_pred, sys_peaks_actual, dias_peaks_actual):
    """
    Computes and returns the amplitude and gain offset for a specific blood pressure signal
    :param predicted_bp_signal: the predicted blood pressure signal we want to process for calibration
    :param actual_bp_signal: the actual blood pressure signal we want to process for calibration
    :param sys_peaks_pred: the predicted systolic peak locations
    :param dias_peaks_pred: the predicted diastolic peak locations
    :param sys_peaks_actual: the actual systolic peak locations
    :param dias_peaks_actual: the actual diastolic peak locations
    :return a: the amplitude shift for calibrating the amplitude offset
    :return b: the gain shift for calibrating the gain offset
    """

    # Grab the signal values at the peak points
    pred_sys_vals = predicted_bp_signal[sys_peaks_pred]
    pred_dia_vals = predicted_bp_signal[dias_peaks_pred]
    act_sys_vals = actual_bp_signal[sys_peaks_actual]
    act_dia_vals = actual_bp_signal[dias_peaks_actual]


    # Compute average peak values
    pred_sys = np.mean(pred_sys_vals)
    pred_dia = np.mean(pred_dia_vals)
    act_sys = np.mean(act_sys_vals)
    act_dia = np.mean(act_dia_vals)

    # Compute gain and offset
    if (pred_sys - pred_dia) == 0: # if the signal is the same do nothing
        return 1.0, 0.0

    a = (act_sys - act_dia) / (pred_sys - pred_dia)
    b = act_dia - a * pred_dia

    return a, b


def get_bp_correction(pred_bp_signal, actual_bp_signal, ecg_signal, fs=125):
    """
    combines all of the necessary functions for getting the a and b values into one function
    :param pred_bp_signal: the predicted bp signal we want to process
    :param actual_bp_signal: the actual bp signal we want to process
    :param ecg_signal: the ecg signal of the patient that we need for getting the heart rate
    :param fs: the sampling frequency of the signal (125 for mimic)
    :return a: the amplitude shift
    :return b: the gain shift
    """
    _, rr_ints, _, _ = compute_heart_rate_from_ecg(ecg_signal, fs=fs)

    sys_peaks, dias_peaks = get_peaks(pred_bp_signal, distance=int(np.mean(rr_ints) * fs))
    sys_peaks_actual, dias_peaks_actual = get_peaks(actual_bp_signal, distance=int(np.mean(rr_ints) * fs))
    a, b = compute_gain_offset_from_peak_heights(pred_bp_signal, actual_bp_signal, sys_peaks,
                                                 dias_peaks, sys_peaks_actual, dias_peaks_actual)

    return a, b


def correct_bp(pred_bp_signal, a, b):
    """
    Applies the amplitude and gain shift to a whole bp waveform and returns it
    :param pred_bp_signal: the predicted bp signal
    :param a: the calculated amplitude shift
    :param b: the calculated gain shift
    :return: the corrected bp signal
    """
    return a * pred_bp_signal + b