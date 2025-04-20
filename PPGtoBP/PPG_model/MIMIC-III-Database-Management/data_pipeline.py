import wfdb
import numpy as np
import requests
import os
import pickle
import time
import pandas as pd

PATIENTS_DB = None
ADMISSIONS_DB = None
DIAGNOSES_DB = None
ICDLOOKUP_DB = None
CHARTEVENTS_DB = None


def check_available_signals(available_signals):
    """
    Checks to see if the signals we are looking for are available for a specific patient
    :param available_signals: the list of available signals
    :return: boolean representing if the desired patient is available
    """

    if "II" in available_signals and "ABP" in available_signals and "PLETH" in available_signals:
        return True
    return False


def extract_waveform_episodes(record, fs=125, episode_len=10, step_len=5):
    """
    extracts waveform episodes from a record and saves them as dictionary objects "episodes"
    :param record: the record we want to process
    :return: a list of dictionaries representing the waveform episodes
    """

    FS = fs
    EPISODE_LEN = episode_len
    STEP_LEN = step_len

    signal_map = {name.upper(): i for i, name in enumerate(record.sig_name)}
    ppg = record.p_signal[:, signal_map['PLETH']]
    abp = record.p_signal[:, signal_map['ABP']]
    ecg = record.p_signal[:, signal_map['II']]

    samples_per_episode = FS * EPISODE_LEN
    step_size = FS * STEP_LEN

    total_samples = min(len(ppg), len(abp), len(ecg))

    episodes = []

    for start in range(0, total_samples - samples_per_episode, step_size):
        ppg_seg = ppg[start:start + samples_per_episode]
        abp_seg = abp[start:start + samples_per_episode]
        ecg_seg = ecg[start:start + samples_per_episode]

        # Skip bad data
        if np.any(np.isnan(ppg_seg)) or np.any(np.isnan(abp_seg)) or np.any(np.isnan(ecg_seg)):
            continue

        # Compute labels from ABP
        sbp = np.max(abp_seg)
        dbp = np.min(abp_seg)

        episodes.append({
            'ppg': ppg_seg,
            'abp': abp_seg,
            'ecg': ecg_seg,
            'sbp': sbp,
            'dbp': dbp,
            'start_idx': start
        })

    return episodes

def get_patient_record_data(patient_id):
    """
    Gets a patient's record data from the MIMIC-III Clinical database saves the data in a dictionary object
    :param patient_id: the id of the patient whose record data is to be retrieved
    :return: the dictionary object representing the patient's record data
    """
    # format patient data
    patient_data = PATIENTS_DB[PATIENTS_DB['SUBJECT_ID'] == patient_id].iloc[0].to_dict()
    patient_data.pop("SUBJECT_ID", None)
    patient_data.pop("ROW_ID", None)
    patient_data.pop("DOD", None)
    patient_data.pop("DOD_HOSP", None)
    patient_data.pop("DOD_SSN", None)

    # format admissions data
    patient_admissions_data = ADMISSIONS_DB[ADMISSIONS_DB['SUBJECT_ID'] == patient_id].iloc[0].to_dict()
    patient_data["ETHNICITY"] = patient_admissions_data["ETHNICITY"]
    patient_data["DIAGNOSIS"] = patient_admissions_data["DIAGNOSIS"].split('\\')

    # format diagnoses data
    patient_diagnoses_codes = list(DIAGNOSES_DB[DIAGNOSES_DB['SUBJECT_ID'] == patient_id].to_dict()['ICD9_CODE'].values())

    patient_diagnoses_values = []
    for code in patient_diagnoses_codes:
        match = ICDLOOKUP_DB[ICDLOOKUP_DB['ICD9_CODE'] == code]
        if not match.empty:
            short_title = match.iloc[0]['SHORT_TITLE']
            patient_diagnoses_values.append(short_title)
        else:
            print(f"Warning: ICD9 code {code} not found in ICDLOOKUP_DB. Appending code to list")
            patient_diagnoses_values.append(code)

    patient_data["DIAGNOSIS"] = patient_data["DIAGNOSIS"] + patient_diagnoses_values

    return patient_data



def process_data(record, patient_id):
    """
    Takes a patient record and down processes it to be fed into the model
    :param record: the record that we are down processing
    :return:
    """

    patient_id = patient_id.split('/')[-1]
    os.makedirs(f'PPG_model/processed_data/{patient_id}/episodes', exist_ok=True)
    os.makedirs(f'PPG_model/processed_data/{patient_id}/episodes/BP', exist_ok=True)
    os.makedirs(f'PPG_model/processed_data/{patient_id}/episodes/SS', exist_ok=True)

    # process waveform episodes for blood pressure model and save them
    print(f"Saving patient: {patient_id} blood pressure model episodes...")
    bp_episodes = extract_waveform_episodes(record)
    for i, ep in enumerate(bp_episodes):
        with open(f'PPG_model/processed_data/{patient_id}/episodes/BP/episode_{i}.p', 'wb') as f:
            pickle.dump(ep, f)

    # process waveform episodes for sleep stage model and save them
    print(f"Saving patient: {patient_id} sleep stage model episodes...")
    ss_episodes = extract_waveform_episodes(record, fs=100, episode_len=30, step_len=30)
    for i, ep in enumerate(ss_episodes):
        with open(f'PPG_model/processed_data/{patient_id}/episodes/SS/episode_{i}.p', 'wb') as f:
            pickle.dump(ep, f)

    # get patient record data and save it
    print(f"Saving patient: {patient_id} records...")
    patient_records = get_patient_record_data(int(patient_id[1:]))
    patient_records["SS_EP_COUNT"] = len(ss_episodes)
    patient_records["BP_EP_COUNT"] = len(bp_episodes)
    with open(f'processed_data/{patient_id}/records.p', 'wb') as f:
        pickle.dump(patient_records, f)


def query_patient(patient_id, record_name):
    """
    Queries the waveform data for a specific patient and the record data for a specific patient and saves this data if
    the data is available
    :param patient_id: the patient id for the patient we are interested in querying from the waveform database
    :param record_name: the name of the record we are interested in querying from the waveform database
    :return: a boolean indicating if the patient data is available or not
    """

    print(f"Querying available patient {patient_id} waveform data...")

    # query sample
    sample_record = wfdb.rdrecord(record_name,
                                  pn_dir=f"mimic3wdb-matched/1.0/{patient_id.strip('/')}",
                                  sampfrom=0,
                                  sampto=1
                                  )

    print("Checking to see if all needed data is available...")
    # check to see if all needed signals are available and handle
    if check_available_signals(sample_record.sig_name):
        print(f"Data available! Downloading patient: {patient_id} waveform data...")

        header = wfdb.rdheader(record_name, pn_dir=f"mimic3wdb-matched/1.0/{patient_id.strip('/')}")
        available_samples = header.sig_len
        target_samples = 125 * 60 * 60 * 24  # 24 hours at 125 Hz
        sampto = min(available_samples, target_samples)

        record = wfdb.rdrecord(record_name,
                               pn_dir=f"mimic3wdb-matched/1.0/{patient_id.strip('/')}",
                               channels=[sample_record.sig_name.index('II'),
                                         sample_record.sig_name.index('PLETH'),
                                         sample_record.sig_name.index('ABP')],
                               sampto=sampto
                               )

        print(f"Processing record: {record_name} for usable model input...")
        process_data(record=record, patient_id=patient_id)
        return True

    else:
        print(f"Data not available for patient: {patient_id} :-(. Aborting...")
        return False



def main():
    # load in the mimic clinical databases first so they are only opened once throughout the program
    global PATIENTS_DB, ADMISSIONS_DB, ICUSTAYS_DB, DIAGNOSES_DB, ICDLOOKUP_DB, CHARTEVENTS_DB, ITEMLOOKUP_DB
    print("Loading Clinical Database Files...")
    PATIENTS_DB = pd.read_csv('PPG_model/raw_data/mimic-clinical/PATIENTS.csv')
    ADMISSIONS_DB = pd.read_csv('PPG_model/raw_data/mimic-clinical/ADMISSIONS.csv')
    ICUSTAYS_DB = pd.read_csv('PPG_model/raw_data/mimic-clinical/ICUSTAYS.csv')
    DIAGNOSES_DB = pd.read_csv("PPG_model/raw_data/mimic-clinical/DIAGNOSES_ICD.csv")
    ICDLOOKUP_DB = pd.read_csv("PPG_model/raw_data/mimic-clinical/D_ICD_DIAGNOSES.csv")
    print("Done")

    # open the record file
    with open('records-waveforms.txt', 'r') as f:
        records = [line.strip() for line in f]

    found_records_count = 0
    start_index = 0
    visited_patients = []
    patient_record_file = open('PPG_model/processed_data/patients.txt', 'w')
    for x in range(start_index, len(records)):
        print(f"Found Records: {found_records_count} Processing record: {x} / {len(records)}...")
        split_record = records[x].split('/')
        patient_id = split_record[0] + '/' + split_record[1]
        if patient_id not in visited_patients:
            record_name = split_record[-1]
            if query_patient(patient_id=patient_id, record_name=record_name):
                found_records_count += 1
                time.sleep(0.05)
            visited_patients.append(patient_id)
            patient_record_file.write(f"{patient_id}\n")
        else:
            print(f"Patient {patient_id} already processed!")

        os.system('cls' if os.name == 'nt' else 'clear')


if __name__ == '__main__':
    main()