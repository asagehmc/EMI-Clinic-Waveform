import helper_functions_bp_model as hf
import evaluation_functions as ef
import calibration as cf
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
import neurokit2 as nk

def get_grade(five, ten, fifteen):
    """
    Gets the BHS letter grade for the given bhs metric data
    :param five: the percentage of values that fall within 5mmHg
    :param ten: the percentage of values that fall within 10mmHg
    :param fifteen: the percentage of values that fall within 15mmHg
    :return: the BHS letter grade as a character
    """

    if five >= 60 and ten >= 85 and fifteen >= 95:
        grade = 'A'
    elif five >= 50 and ten >= 75 and fifteen >= 90:
        grade = 'B'
    elif five >= 40 and ten >= 65 and fifteen >= 85:
        grade = 'C'
    else:
        grade = 'D'

    return grade


def eval_patient_bulk(patient_list, episode_count=10, calibrated=False, filtered=False):
    """
    Evaluates a bulk of patients given and returns the average BHS metric data for the group of patients
    :param patient_list: the list of patients we want to evaluate
    :param episode_count: the number of episodes we want to evaluate for each patient
    :param calibrated: a boolean indicating whether we want to calibrate the patients for evaluation
    :param filtered: a boolean indicating whether we want to filter the patients for evaluation
    :return: the BHS metric data and grades for the group of patients
    """

    subset_length = 1024

    sbp_5, sbp_10, sbp_15, sbp_grades = [], [], [], []
    dbp_5, dbp_10, dbp_15, dbp_grades = [], [], [], []
    mbp_5, mbp_10, mbp_15, mbp_grades = [], [], [], []

    for patient in patient_list:
        try:
            norm_ppg, actual_abp, ecg = hf.get_patient_episode_data(patient, episode_count)

            if ((np.mean(nk.ppg_quality(norm_ppg, sampling_rate=125, method="templatematch")) * 100) < 90) and filtered:
                continue

            pred_bp = hf.predict_bp_from_ppg(norm_ppg)
            actual_bp = hf.trim_waveform(actual_abp)

            if calibrated:
                a, b = cf.get_bp_correction(pred_bp[:subset_length], actual_bp[:subset_length], ecg)
                pred_bp_calibrated = cf.correct_bp(pred_bp, a, b)
                bhs_results = ef.eval_bhs_standard_dict(actual_bp, pred_bp_calibrated, ecg)
            else:
                bhs_results = ef.eval_bhs_standard_dict(actual_bp, pred_bp, ecg)

            sbp_grades.append(bhs_results['SBP']['BHS Grade'])
            sbp_5.append(bhs_results['SBP']['≤5 mmHg'])
            sbp_10.append(bhs_results['SBP']['≤10 mmHg'])
            sbp_15.append(bhs_results['SBP']['≤15 mmHg'])

            dbp_grades.append(bhs_results['DBP']['BHS Grade'])
            dbp_5.append(bhs_results['DBP']['≤5 mmHg'])
            dbp_10.append(bhs_results['DBP']['≤10 mmHg'])
            dbp_15.append(bhs_results['DBP']['≤15 mmHg'])

            mbp_grades.append(bhs_results['MBP']['BHS Grade'])
            mbp_5.append(bhs_results['MBP']['≤5 mmHg'])
            mbp_10.append(bhs_results['MBP']['≤10 mmHg'])
            mbp_15.append(bhs_results['MBP']['≤15 mmHg'])

        except Exception as e:
            continue

    sbp_5, sbp_10, sbp_15 = np.mean(sbp_5), np.mean(sbp_10), np.mean(sbp_15)
    sbp_grade = get_grade(sbp_5, sbp_10, sbp_15)

    dbp_5, dbp_10, dbp_15 = np.mean(dbp_5), np.mean(dbp_10), np.mean(dbp_15)
    dbp_grade = get_grade(dbp_5, dbp_10, dbp_15)

    mbp_5, mbp_10, mbp_15 = np.mean(mbp_5), np.mean(mbp_10), np.mean(mbp_15)
    mbp_grade = get_grade(mbp_5, mbp_10, mbp_15)

    ef.format_bhs_results_table(
        {
            'SBP' : {'≤5 mmHg': sbp_5, '≤10 mmHg': sbp_10, '≤15 mmHg': sbp_15, 'BHS Grade': sbp_grade},
            'DBP' : {'≤5 mmHg': dbp_5, '≤10 mmHg': dbp_10, '≤15 mmHg': dbp_15, 'BHS Grade': dbp_grade},
            'MBP' : {'≤5 mmHg': mbp_5, '≤10 mmHg': mbp_10, '≤15 mmHg': mbp_15, 'BHS Grade': mbp_grade},
        }
    )

    return sbp_grades, dbp_grades, mbp_grades


def get_grades_plot(sbp_grades, dbp_grades, mbp_grades, plot_title):
    sbp_values, sbp_counts = np.unique(sbp_grades, return_counts=True)
    dbp_values, dbp_counts = np.unique(dbp_grades, return_counts=True)
    mbp_values, mbp_counts = np.unique(mbp_grades, return_counts=True)

    grades = set()
    grades.update(sbp_values)
    grades.update(dbp_values)
    grades.update(mbp_values)
    grades = sorted(grades)

    sbp_dict = dict(zip(sbp_values, sbp_counts))
    dbp_dict = dict(zip(dbp_values, dbp_counts))
    mbp_dict = dict(zip(mbp_values, mbp_counts))

    sbp = [sbp_dict.get(g, 0) for g in grades]
    dbp = [dbp_dict.get(g, 0) for g in grades]
    mbp = [mbp_dict.get(g, 0) for g in grades]

    x = np.arange(len(grades))
    width = 0.25

    fig, ax = plt.subplots()

    # Grouped bars
    ax.bar(x - width, sbp, width, label='SBP')
    ax.bar(x, dbp, width, label='DBP')
    ax.bar(x + width, mbp, width, label='MBP')

    # Labels and legend
    ax.set_xlabel('Risk Grade')
    ax.set_ylabel('Count')
    ax.set_title(plot_title)
    ax.set_xticks(x)
    ax.set_xticklabels(grades)
    ax.legend()
    plt.show()

