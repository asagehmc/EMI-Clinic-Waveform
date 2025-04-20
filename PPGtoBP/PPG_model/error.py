import neurokit2 as nk
import helper_functions_bp_model as hf
import calibration as cf
import evaluation_functions as ef
import numpy as np

def bootstrap_confidence_interval(data, num_bootstrap=1000, ci=95):
    boot_means = []
    n = len(data)
    for _ in range(num_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        boot_means.append(np.mean(sample))
    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return lower, upper


def prediction_interval(data, ci=95):
    if len(data) == 0:
        return (np.nan, np.nan)

    lower_percentile = (100 - ci) / 2
    upper_percentile = 100 - lower_percentile
    lower = np.percentile(data, lower_percentile)
    upper = np.percentile(data, upper_percentile)
    return lower, upper


def get_patient_ppg_quality_bins(patient_list, episode_count=10):
    """
    Calculates the ppg quality for the list of given patients and returns lists of patients according to their ppg
    quality score.
    :param patient_list: the list of patients we want to process
    :param episode_count: the number of episodes we want to process for each patient
    :return: 10 lists of patients where the ppg quality score is within a specific threshold
    """

    one, ten, twenty, thirty, forty, fifty, sixty, seventy, eighty, ninety = [], [], [], [], [], [], [], [], [], []
    for patient in patient_list:
        try:
            norm_ppg, _, ecg = hf.get_patient_episode_data(patient, episode_count)
            quality_score = np.mean(nk.ppg_quality(norm_ppg, sampling_rate=125, method="templatematch")) * 100
            if quality_score >= 90:
                ninety.append(patient)
            elif quality_score >= 80:
                eighty.append(patient)
            elif quality_score >= 70:
                seventy.append(patient)
            elif quality_score >= 60:
                sixty.append(patient)
            elif quality_score >= 50:
                fifty.append(patient)
            elif quality_score >= 40:
                forty.append(patient)
            elif quality_score >= 30:
                thirty.append(patient)
            elif quality_score >= 20:
                twenty.append(patient)
            elif quality_score >= 10:
                ten.append(patient)
            else:
                one.append(patient)

        except Exception as e:
            continue

    return {
            0: one, 10: ten,
            20: twenty, 30: thirty,
            40: forty, 50: fifty,
            60: sixty, 70: seventy,
            80: eighty, 90: ninety
            }


def get_error_bounds_for_ppg_quality_metrics(bins, episode_count=10, fs=125):
    error_bounds = {}

    for name, bin in bins.items():
        bin_sys_errors = []
        bin_dias_errors = []
        bin_mean_errors = []
        for patient in bin:
            # get prediction
            ppg, bp, ecg = hf.get_patient_episode_data(patient, episode_count)
            pred_bp = hf.predict_bp_from_ppg(ppg)
            a, b = cf.get_bp_correction(pred_bp, bp, ecg)
            pred_bp = cf.correct_bp(pred_bp, a, b)
            _, rr_ints, _, _ = cf.compute_heart_rate_from_ecg(ecg, fs=125)

            pred_sys_peaks, pred_dias_peaks = cf.get_peaks(pred_bp, int(np.mean(rr_ints) * fs))
            actual_sys_peaks, actual_dias_peaks = cf.get_peaks(bp, int(np.mean(rr_ints) * fs))
            pred_sys_peaks, actual_sys_peaks = ef.truncate_to_match(pred_sys_peaks, actual_sys_peaks)
            pred_dias_peaks, actual_dias_peaks = ef.truncate_to_match(pred_dias_peaks, actual_dias_peaks)

            # get systolic error
            sys_errors = bp[actual_sys_peaks] - pred_bp[pred_sys_peaks]
            bin_sys_errors.append(sys_errors)

            # get diastolic error
            dias_errors = bp[actual_dias_peaks] - pred_bp[pred_dias_peaks]
            bin_dias_errors.append(dias_errors)

            # get mean error
            pred_sys_peaks, actual_sys_peaks, pred_dias_peaks, actual_dias_peaks = (
                ef.truncate_to_match(pred_sys_peaks, actual_sys_peaks, pred_dias_peaks, actual_dias_peaks))
            actual_mbp = bp[actual_dias_peaks] + (bp[actual_sys_peaks] - bp[actual_dias_peaks]) / 3
            pred_mbp = pred_bp[pred_dias_peaks] + (pred_bp[pred_sys_peaks] - pred_bp[pred_dias_peaks]) / 3
            mean_errors = actual_mbp - pred_mbp
            bin_mean_errors.append(mean_errors)


        if len(bin_sys_errors) > 0:
            bin_sys_errors = np.concatenate(bin_sys_errors)
        if len(bin_dias_errors) > 0:
            bin_dias_errors = np.concatenate(bin_dias_errors)
        if len(bin_mean_errors) > 0:
            bin_mean_errors = np.concatenate(bin_mean_errors)


        error_bounds[name] = {
            "sys errors" : prediction_interval(bin_sys_errors),
            "dias errors" : prediction_interval(bin_dias_errors),
            "mean errors" : prediction_interval(bin_mean_errors)
        }

    return error_bounds





