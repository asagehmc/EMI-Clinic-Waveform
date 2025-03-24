import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wfdb
import csv
import itertools as IT
import datetime
import json
import pickle

patients = pd.read_csv('mimic_data/PATIENTS.csv')
admissions = pd.read_csv('mimic_data/ADMISSIONS.csv')
diagnoses = pd.read_csv('mimic_data/DIAGNOSES_ICD.csv')

with open('/Users/lydiastone/PycharmProjects/EIT-Clinic-Waveform/filtered_ABP_II.txt', 'r') as file:
    python_dict_from_file = json.load(file)
leadii_sub_ids = [int(x.split('/')[1][1:]) for x in python_dict_from_file.keys()]

admissions_leadii_patient_index = admissions.SUBJECT_ID.apply(lambda x: int(x) in leadii_sub_ids)
patients_leadii_patient_index =  patients.SUBJECT_ID.apply(lambda x: int(x) in leadii_sub_ids)
diagnoses_leadii_patient_index =  diagnoses.SUBJECT_ID.apply(lambda x: int(x) in leadii_sub_ids)

admissions_leadii = admissions[admissions_leadii_patient_index]
patients_leadii = patients[patients_leadii_patient_index]
diagnoses_leadii = diagnoses[diagnoses_leadii_patient_index]

def icd_9_to_10_from_code(code, icd_9_to_10_d):
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

# add corresponding icd 10 diagnoses
icd_9_to_10 = pd.read_csv('mimic_data/icd9toicd10cmgem.csv')
icd_9_to_10_d = dict(zip(icd_9_to_10.icd9cm.values,icd_9_to_10.icd10cm.values))

diagnoses_leadii['ICD10_CODE'] = diagnoses_leadii.ICD9_CODE.apply(icd_9_to_10)
