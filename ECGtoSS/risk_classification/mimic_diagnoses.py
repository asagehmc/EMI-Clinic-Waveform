"""
Filename: mimic_diagnoses.py

Description: This file contains functions for accessing MIMIC clinical data in relation to
            the waveform data using patient ids
"""

import os
import pandas as pd
import numpy as np
import json
import datetime

def load_admissions_leadii():
    """
    :return: pandas DataFrame, admissions data for patients
    """
    rc_dir_path = os.path.dirname(os.path.abspath(__file__))
    admissions_file = os.path.join(rc_dir_path, 'mimic_data', 'ADMISSIONS.csv')
    admissions = pd.read_csv(admissions_file)
    return admissions

def get_leadii_dataframes(patients,admissions,diagnoses):
    """
    :param patients: patient pd df
    :param admissions: admissions pd df
    :param diagnoses: diagnoses pd df
    :return: pd df subset of each pd df with ecg lead ii data
    """
    # gets ids with desired data
    file_path = "mimic_data/filtered_ABP_II.txt"
    with open(file_path, 'r') as file:
        python_dict_from_file = json.load(file)
    leadii_sub_ids = [int(x.split('/')[1][1:]) for x in python_dict_from_file.keys()]

    # gets patient indices for each df if id in leadii_sub_ids
    admissions_leadii_patient_index = admissions.SUBJECT_ID.apply(lambda x: int(x) in leadii_sub_ids)
    patients_leadii_patient_index = patients.SUBJECT_ID.apply(lambda x: int(x) in leadii_sub_ids)
    diagnoses_leadii_patient_index = diagnoses.SUBJECT_ID.apply(lambda x: int(x) in leadii_sub_ids)

    # gets the subsets of those dfs
    admissions_leadii = admissions[admissions_leadii_patient_index]
    patients_leadii = patients[patients_leadii_patient_index]
    diagnoses_leadii = diagnoses[diagnoses_leadii_patient_index]

    return admissions_leadii, patients_leadii, diagnoses_leadii

def icd_9_to_10_from_code(code, icd_9_to_10_d):
    """
    :param code: str, icd9 code
    :param icd_9_to_10_d: dictionary of icd9 keys and corresponding icd10 values
    :return: str, icd10 code
    """
    try:
        if (code[0]=='0' or len(code)==5):
            icd10 = icd_9_to_10_d[code]
            return icd10
        else:
            icd10 = icd_9_to_10_d[code.zfill(len(code)+1)]
            return icd10
    except:
        try:
            icd10 = icd_9_to_10_d[code]
            return icd10
        except:
            icd10 = 'unknown'
            return icd10

def add_icd_10_code_to_diagnoses(diagnoses):
    """
    adds icd10 code column under name 'ICD10_CODE' to diagnoses dataframe
    :param diagnoses: diagnoses pd df
    """
    # get icd9 to icd10 dictionary
    icd_9_to_10 = pd.read_csv('mimic_data/icd9toicd10cmgem.csv')
    icd_9_to_10_d = dict(zip(icd_9_to_10.icd9cm.values, icd_9_to_10.icd10cm.values))

    # add corresponding icd 10 diagnoses
    diagnoses['ICD10_CODE'] = diagnoses.ICD9_CODE.apply(lambda x: icd_9_to_10_from_code(x, icd_9_to_10_d))

    return

def get_patient_labels(patient_ids,diagnoses):
    """
    :param patient_ids: list of patient ids in string or int form
    :param diagnoses: pd df, diagnoses pd df
    :return: lists of string patient ids for any risk, cad, cvd, and pad
    """
    # ICD 10 codes corresponding to the different predictive of MACE diseases
    CAD_codes = ['I200', 'I2109', 'I2111', 'I2119', 'I2129', 'I213', 'I214', 'I240', 'Z9861', 'Z955',
                 'I21', 'I210', 'I2101', 'I2102', 'I2109', 'I211', 'I2111', 'I2119', 'I212', 'I2121', 'I2129',
                 'I213', 'I214', 'I22', 'I220', 'I221', 'I222', 'I228', 'I229', 'I23', 'I230', 'I231', 'I232', 'I233',
                 'I234', 'I235', 'I236', 'I237', 'I238', 'I252']
    CVD_codes = ['I63', 'G450', 'G451', 'G452', 'G454', 'G458', 'G459']
    PAD_codes = ['I70201', 'I70202', 'I70203', 'I70208', 'I70209', 'I70211', 'I70212', 'I70213', 'I70218', 'I70219',
                 'I70221', 'I70222', 'I70223', 'I70228', 'I70229', 'I70231', 'I70231', 'I70232', 'I70232', 'I70233',
                 'I70233', 'I70234', 'I70234', 'I70235', 'I70235', 'I70238', 'I70238', 'I70239', 'I70239', 'I70241',
                 'I70241', 'I70242', 'I70242', 'I70243', 'I70243', 'I70244', 'I70244', 'I70245', 'I70245', 'I70248',
                 'I70248', 'I70249', 'I70249', 'I7025', 'I7025', 'I70261', 'I70262', 'I70263', 'I70268', 'I70269',
                 'I70291', 'I70292', 'I70293', 'I70298', 'I70299']

    # gets integer patient ids to match the pandas dfs
    patient_ids = [int(pid) for pid in patient_ids]

    # gets diagnoses for the patients
    model_pat_diags = diagnoses.loc[diagnoses.SUBJECT_ID.isin(patient_ids)]

    # indices of disease patients
    model_CAD_patients_id = model_pat_diags.loc[
        model_pat_diags.ICD10_CODE.apply(lambda x: x in CAD_codes)].SUBJECT_ID.unique()
    model_CVD_patients_id = model_pat_diags.loc[
        model_pat_diags.ICD10_CODE.apply(lambda x: x in CVD_codes)].SUBJECT_ID.unique()
    model_PAD_patients_id = model_pat_diags.loc[
        model_pat_diags.ICD10_CODE.apply(lambda x: x in PAD_codes)].SUBJECT_ID.unique()

    # str patient ids if they have any of the diseases
    risk_patients = [str(pid).zfill(6) for pid in
                     np.unique(np.concatenate((model_PAD_patients_id, model_CVD_patients_id, model_CAD_patients_id)))]

    # specific disease string patient ids if needed
    cad_patients = [str(pid).zfill(6) for pid in model_CAD_patients_id]
    cvd_patients = [str(pid).zfill(6) for pid in model_CVD_patients_id]
    pad_patients = [str(pid).zfill(6) for pid in model_PAD_patients_id]

    # labels - 0 if no diseases, 1 if any of the three disease types
    y = np.array([1 if str(pid).zfill(6) in risk_patients else 0 for pid in patient_ids])

    return y

def age_of_subject(subject_id,patients,admissions):
    """
    :param subject_id: int, patient id
    :param patients: pd df, patient.csv pandas df
    :param admissions: pd df, admissions.csv pandas df
    :return: age, int, age of patient at the time of their first stay
    """
    # get the patient's row
    patient = patients.loc[patients['SUBJECT_ID'] == subject_id]
    patient = patient.loc[patient.index.values[0]]

    # date of birth and admission time to get age at admission
    dob = datetime.datetime.strptime(patient.DOB, "%Y-%m-%d %H:%M:%S").date()
    patient_admissions = admissions.loc[admissions['SUBJECT_ID'] == subject_id]
    admittime = datetime.datetime.strptime(patient_admissions.ADMITTIME.values[0], "%Y-%m-%d %H:%M:%S").date()
    age = ((admittime - dob).days) // 365

    # the database autosets dob to 300 years before admission for patients 89 and above
    if age >= 299:
        age = 90

    return age


def basic_info(subject_id,patients,admissions):
    """
    :param subject_id: int, patient id
    :param patients: pd df, patient.csv pd df
    :param admissions: pd df, admissions.csv pd df
    :return: age, int, age of patient at the time of their first stay
             sex, string, 'F' or 'M'
    """
    patient = patients.loc[patients['SUBJECT_ID'] == subject_id]
    patient = patient.loc[patient.index.values[0]]

    sex = patient['GENDER']
    age = age_of_subject(subject_id,patients,admissions)

    return age, sex

def check_if_died_during_admission(patient_id, datetime_str, admissions):
    """
    :param patient_id: int, patient id
    :param datetime_str: str, instance start time datetime string from the data
    :param admissions: pd df, admissions.csv pd df
    :return: True if the patient died during the admission, False otherwise
    """
    # gets datetime object
    format_string = "%Y-%m-%d %H:%M:%S"
    data_time = datetime.datetime.strptime(datetime_str, format_string)

    # gets all patient's admissions
    patient_admissions = admissions.loc[admissions['SUBJECT_ID'] == patient_id]

    # goes through all admit and discharge times
    for i in range(len(patient_admissions)):
        atime = datetime.datetime.strptime(patient_admissions.ADMITTIME.values[i], "%Y-%m-%d %H:%M:%S")
        dtime = datetime.datetime.strptime(patient_admissions.DISCHTIME.values[i], "%Y-%m-%d %H:%M:%S")

        # if instance from a stay and the patient died during that stay
        if data_time <= dtime and data_time >= atime and patient_admissions.HOSPITAL_EXPIRE_FLAG.values[i] == 1:
            return True

    return False

def ethnicities_for_pids(patient_ids,ethnicities):
    """
    :param patient_ids: list of ints
    :param ethnicities: pd df, admissions.csv pd df with ETHNICITY_COND column
    :return: list of strings, list of ethnicities corresponding to the list of patient ids
    """
    ethnicities_for_model_patients = [ethnicities[ethnicities.SUBJECT_ID == pid].ETHNICITY_COND.values[0]
                                      if len(ethnicities[ethnicities.SUBJECT_ID == pid].ETHNICITY_COND.values) == 1
                                      else 'UNKNOWN' for pid in patient_ids]

    return ethnicities_for_model_patients

def get_race(ethn):
    """
    :param ethn: str, ethnicity directly from admissions.csv
    :return: str, condensed ethnicity, one of 'ASIAN', 'BLACK', 'WHITE',
                  'HISPANIC/LATINO', 'PACIFIC ISLANDER', or 'UNKNOWN'
    """
    if 'ASIAN' in ethn:
        return 'ASIAN'
    elif 'BLACK' in ethn:
        return 'BLACK'
    elif 'WHITE' in ethn:
        return 'WHITE'
    elif 'HISPANIC' in ethn:
        return 'HISPANIC/LATINO'
    elif ethn=='NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER':
        return 'PACIFIC ISLANDER'
    elif ethn=='MIDDLE EASTERN':
        return 'WHITE'
    elif ethn=='PORTUGUESE':
        return 'WHITE'
    else:
        return 'UNKNOWN'

def make_testing_dataframe(patient_ids, y_pred, y_true,patients,admissions,ethnicities):
    """
    :param patient_ids: list of strs, list of patient ids
    :param y_pred: list of ints, list of predicted risk values
    :param y_true: list of ints, list of true risk values
    :param patients: pd df, patients.csv pd df
    :param admissions: pd df, admissions.csv pd df
    :param ethnicities: pd df, admissions.csv pd df
    :return: pd df, with patient_ids, y_pred, y_true, and other demographic information for stratified accuracies
    """
    # get int versions of patient ids
    pids_ints = [int(pid) for pid in patient_ids]

    # get demographic information
    ethnicities_for_model_patients = ethnicities_for_pids(pids_ints,ethnicities)
    age_sex = np.array([basic_info(pid,patients,admissions) for pid in pids_ints])
    age = age_sex[:,0]
    sex = age_sex[:,1]

    # make pd df
    df = pd.DataFrame({'PATIENT_ID':pids_ints, 'Y_PRED':y_pred, 'Y_TRUE':y_true,
                       'ETHNICITY':ethnicities_for_model_patients, 'AGE':age, 'SEX':sex})
    return df


