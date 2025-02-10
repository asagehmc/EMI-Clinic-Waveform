import json
from datetime import datetime
from traceback import print_exc

import numpy as np
import sleepecg
import wfdb
import requests
import glob
import os

PHYSIONET = "https://physionet.org/files/"
MIMIC = "mimic3wdb-matched/1.0/"
# Manual Download: https://physionet.org/files/mimic3wdb-matched/1.0/

# required signal length in seconds
required_segment_len = 4 * 60 * 60 # (4 hours)
# minimum frequency for a segment to be acceptable
min_frequency = 125
# the proportion of nans in ecg data below which we skip the record
NAN_THRESHOLD = 0.99

# time interval for output
time_step = 30
# which signals we filter for in records
ecg_channel_name = "II"
abp_channel_name = "ABP"
required_signals = ["ABP", "II"]


def lerp_ecg(ecg_data, ecg_nans):
    # process ecg data (lerp out nans)
    empty_indexes = np.array(np.where(ecg_nans))[0]
    i = 0
    while i < len(empty_indexes):
        current_index = empty_indexes[i]
        lerp_size = 0
        while i < len(empty_indexes) and empty_indexes[i] == current_index + lerp_size:
            i += 1
            lerp_size += 1
        start = ecg_data[current_index - 1] if current_index > 0 else 0
        end = ecg_data[current_index + lerp_size] if current_index + lerp_size < len(ecg_data) else 0
        for lerp_num in range(lerp_size):
            ecg_data[current_index + lerp_num] = start + (end - start) * (lerp_num + 1) / (lerp_size + 1)
    return ecg_data


def process_bp_data(bp, fs):
    window_size = fs * time_step
    i = 0
    out = np.array([])
    bp_len = len(bp)
    while i + window_size <= bp_len:
        mean = np.nanmean(bp[i: min(i + window_size, bp_len)])
        out = np.append(out, round(float(mean), 2))
        i += window_size
    return out


def calculate_sleep_stages(ecg, fs):
    # detect heartbeats
    beats = sleepecg.detect_heartbeats(ecg, fs)
    sleepecg.plot_ecg(ecg, fs, beats=beats)

    # load SleepECG classifier (requires tensorflow)
    clf = sleepecg.load_classifier("wrn-gru-mesa-weighted", "SleepECG")

    # predict sleep stages
    record = sleepecg.SleepRecord(
        sleep_stage_duration=30,
        heartbeat_times=beats / fs,
    )

    stages = sleepecg.stage(clf, record, return_mode="prob")
    return stages


# converts a record into sleep cycles / abp per time_step
def convert_record(patient, record, error_path):
    try:
        # read the record from physionet
        record = wfdb.rdrecord(record, pn_dir=f"{MIMIC}{patient}")

        ecg_index = record.sig_name.index(ecg_channel_name)
        ecg_data = record.p_signal[:, ecg_index].astype(np.float64)

        abp_index = record.sig_name.index(abp_channel_name)
        abp_data = record.p_signal[:, abp_index].astype(np.float64)

        # get an array of indices that are NaN
        ecg_nans = np.isnan(ecg_data)
        if len(ecg_data) - sum(ecg_nans) < NAN_THRESHOLD * len(ecg_data):
            print(f"skipped segment {patient} {record} due to high nan-rate")
            return
        ecg_data = lerp_ecg(ecg_data, ecg_nans)
        sleep_stages = calculate_sleep_stages(ecg_data, record.fs)
        bp_30s = process_bp_data(abp_data, record.fs)

        return {
            "sleep_stages": sleep_stages.tolist(),
            "blood_pressure": bp_30s.tolist()
                }

    except Exception:
        error_write(error_path, f"Problem converting record [Patient: {patient} Record: {record}]")
        raise


def write_to_file(patient, segment_name, converted, segment_datetime):
    os.makedirs(f"data/patients/{patient}", exist_ok=True)
    with open(f"data/patients/{patient}/{segment_name}.json", "w") as f:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        converted["last_update"] = now
        converted["date"] = segment_datetime
        json.dump(converted, f, indent=2)
        print(f"saved record {patient} {segment_name}")


def scan_mimic(error_path):
    # get the list of patients from the db
    record_list = wfdb.get_record_list(MIMIC)

    # for showing percentages while running
    for patient in record_list[6:]:
        try:
            # get the list of records for an individual patient
            patient_records = wfdb.get_record_list(f"{MIMIC}{patient}")
            # get a list of the master headers:
            # NOTE: This is suuper ugly but to this point I don't know a better way to sort TODO figure out a better way
            master_header_names = [f for f in patient_records if f.startswith('p') and not f.endswith('n')]
            for header_name in master_header_names:
                # convert the patient master header into a MultiRecord
                header = wfdb.rdheader(f"{header_name}", rd_segments=True, pn_dir=f"{MIMIC}{patient}")
                for segment in header.segments:
                    try:
                        if segment is not None and "layout" in segment.record_name:
                            # we can skip all segment queries if the layout doesn't even have the right signals
                            has_required_signals = not (False in [(x in header.sig_name) for x in required_signals])
                            if not has_required_signals:
                                print(f"skipped patient {patient} due to missing signals")
                                break
                        elif segment is not None \
                                and segment.sig_len > required_segment_len * segment.fs \
                                and segment.fs >= min_frequency:
                            has_required_signals = not (False in [(x in segment.sig_name) for x in required_signals])
                            if has_required_signals:
                                converted = convert_record(patient, segment.record_name, error_path)
                                write_to_file(patient, segment.record_name, converted, header.base_datetime)
                            else:
                                print(f"skipped segment {patient} {segment.record_name} due to missing signals")
                    except Exception as e:
                        error_write(error_path, f"Problem parsing patient record [patient: {patient}, record: {segment.record_name if segment else '--'}]")
                        raise

        except Exception as e:
            error_write(error_path, f"Problem with patient [Patient: {patient}]")
            raise


def error_write(filename, message):
    with open(f"data/{filename}", "a") as f:
        f.write(message + "\n")
        print(message)


if __name__ == "__main__":
    error_file = "errors.txt"
    error_write(error_file, f"---------- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    scan_mimic(error_file)

