import pandas as pd
import numpy as np
import json
import datetime

def get_leadii_dataframes(patients,admissions,diagnoses):
    """
    :param patients: patient pd df
    :param admissions: admissions pd df
    :param diagnoses: diagnoses pd df
    :return: subset of each pd df with ecg lead ii data
    """
    with open('/Users/lydiastone/PycharmProjects/EIT-Clinic-Waveform/filtered_ABP_II.txt', 'r') as file:
        python_dict_from_file = json.load(file)
    leadii_sub_ids = [int(x.split('/')[1][1:]) for x in python_dict_from_file.keys()]

    admissions_leadii_patient_index = admissions.SUBJECT_ID.apply(lambda x: int(x) in leadii_sub_ids)
    patients_leadii_patient_index = patients.SUBJECT_ID.apply(lambda x: int(x) in leadii_sub_ids)
    diagnoses_leadii_patient_index = diagnoses.SUBJECT_ID.apply(lambda x: int(x) in leadii_sub_ids)

    admissions_leadii = admissions[admissions_leadii_patient_index]
    patients_leadii = patients[patients_leadii_patient_index]
    diagnoses_leadii = diagnoses[diagnoses_leadii_patient_index]

    return admissions_leadii, patients_leadii, diagnoses_leadii

def icd_9_to_10_from_code(code, icd_9_to_10_d):
    """
    :param code: icd9 code
    :param icd_9_to_10_d: dictionary of icd9 keys and corresponding icd10 values
    :return: icd10 code
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
    adds icd10 code to diagnoses
    :param diagnoses: diagnoses pd df
    """
    # get icd9 to icd10 dictionary
    icd_9_to_10 = pd.read_csv('mimic_data/icd9toicd10cmgem.csv')
    icd_9_to_10_d = dict(zip(icd_9_to_10.icd9cm.values, icd_9_to_10.icd10cm.values))

    # add corresponding icd 10 diagnoses
    diagnoses['ICD10_CODE'] = diagnoses.ICD9_CODE.apply(lambda x: icd_9_to_10_from_code(x, icd_9_to_10_d))

    return

def get_patient_labels(patient_ids):
    """
    :param patient_ids: list of patient ids in string or int form
    :return: lists of string patient ids for any risk, cad, cvd, and pad
    """
    CAD_codes = ['I200', 'I2109', 'I2111', 'I2119', 'I2129', 'I213', 'I214', 'I240', 'Z9861', 'Z955',
                 'I21', 'I210', 'I2101', 'I2102', 'I2109', 'I211', 'I2111', 'I2119', 'I212', 'I2121', 'I2129',
                 'I213', 'I214', 'I22', 'I220', 'I221', 'I222', 'I228', 'I229', 'I23', 'I230', 'I231', 'I232', 'I233',
                 'I234', 'I235', 'I236', 'I237', 'I238', 'I252']
    CVD_codes = ['I63', 'G450', 'G451', 'G452', 'G454', 'G458', 'G459']
    PAD_codes = 'I70201, I70202, I70203, I70208, I70209, I70211, I70212, I70213, I70218, I70219, I70221, I70222, I70223, I70228, I70229, I70231, I70231, I70232, I70232, I70233, I70233, I70234, I70234, I70235, I70235, I70238, I70238, I70239, I70239, I70241, I70241, I70242, I70242, I70243, I70243, I70244, I70244, I70245, I70245, I70248, I70248, I70249, I70249, I7025, I7025, I70261, I70262, I70263, I70268, I70269, I70291, I70292, I70293, I70298, I70299'.split(
        ', ')

    patient_ids = [int(pid) for pid in patient_ids]

    model_pat_diags = diagnoses_leadii.loc[diagnoses_leadii.SUBJECT_ID.isin(patient_ids)]

    model_CAD_patients_id = model_pat_diags.loc[
        model_pat_diags.ICD10_CODE.apply(lambda x: x in CAD_codes)].SUBJECT_ID.unique()
    model_CVD_patients_id = model_pat_diags.loc[
        model_pat_diags.ICD10_CODE.apply(lambda x: x in CVD_codes)].SUBJECT_ID.unique()
    model_PAD_patients_id = model_pat_diags.loc[
        model_pat_diags.ICD10_CODE.apply(lambda x: x in PAD_codes)].SUBJECT_ID.unique()

    risk_patients = [str(pid).zfill(6) for pid in
                     np.unique(np.concatenate((model_PAD_patients_id, model_CVD_patients_id, model_CAD_patients_id)))]

    cad_patients = [str(pid).zfill(6) for pid in model_CAD_patients_id]
    cvd_patients = [str(pid).zfill(6) for pid in model_CVD_patients_id]
    pad_patients = [str(pid).zfill(6) for pid in model_PAD_patients_id]

    y = np.array([1 if str(pid).zfill(6) in risk_patients else 0 for pid in patient_ids])

    return y

def age_of_subject(subject_id):
    """
    returns age for a particular subject_id
    input: subject_id, int
    output: age, int, age of patient at the time of their first stay
    """
    patient = patients.loc[patients['SUBJECT_ID'] == subject_id]
    patient = patient.loc[patient.index.values[0]]

    # date of birth and admission time to get age at admission
    dob = datetime.datetime.strptime(patient.DOB, "%Y-%m-%d %H:%M:%S").date()
    patient_admissions = admissions.loc[admissions['SUBJECT_ID'] == subject_id]
    admittime = datetime.datetime.strptime(patient_admissions.ADMITTIME.values[0], "%Y-%m-%d %H:%M:%S").date()
    age = ((admittime - dob).days) // 365

    # the database autosets dob to 300 years before admission for patients 89 and above
    if age >= 299:
        age = 89

    return age


def basic_info(subject_id):
    """
    returns age and sex for a particular subject_id
    input: subject_id, int
    output: age, int, age of patient at the time of their first stay
            sex, string, 'F' or 'M'
    """
    patient = patients.loc[patients['SUBJECT_ID'] == subject_id]
    patient = patient.loc[patient.index.values[0]]

    sex = patient['GENDER']
    age = age_of_subject(subject_id)

    return age, sex

def check_if_died_during_admission(patient_id, datetime_str):
    """
    :param patient_id: int, patient id
    :param datetime_str: str, datetime string from the data
    :return: True if the patient died during the admission, False otherwise
    """
    format_string = "%Y-%m-%d %H:%M:%S"
    data_time = datetime.datetime.strptime(datetime_str, format_string)

    patient_admissions = admissions_leadii.loc[admissions_leadii['SUBJECT_ID'] == patient_id]

    for i in range(len(patient_admissions)):
        atime = datetime.datetime.strptime(patient_admissions.ADMITTIME.values[i], "%Y-%m-%d %H:%M:%S")
        dtime = datetime.datetime.strptime(patient_admissions.DISCHTIME.values[i], "%Y-%m-%d %H:%M:%S")

        if data_time <= dtime and data_time >= atime and patient_admissions.HOSPITAL_EXPIRE_FLAG.values[i] == 1:
            return True

    return False

patients = pd.read_csv('mimic_data/PATIENTS.csv')
admissions = pd.read_csv('mimic_data/ADMISSIONS.csv')
diagnoses = pd.read_csv('mimic_data/DIAGNOSES_ICD.csv')

admissions_leadii, patients_leadii, diagnoses_leadii = get_leadii_dataframes(patients, admissions, diagnoses)

add_icd_10_code_to_diagnoses(diagnoses_leadii)
