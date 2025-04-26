import json
from datetime import datetime

import numpy as np
import sleepecg
import wfdb
import os
from scipy.signal import find_peaks

PHYSIONET = "https://physionet.org/files/"
MIMIC = "mimic3wdb-matched/1.0/"
# Manual Download: https://physionet.org/files/mimic3wdb-matched/1.0/

# required signal length in seconds
required_segment_len = 4 * 60 * 60  # (4 hours)
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

# load SleepECG classifier (requires tensorflow)
clf = sleepecg.load_classifier("wrn-gru-mesa-weighted", "SleepECG")


def lerp_ecg(ecg_data, ecg_nans):
    """Compresses continuous ABP data to larger intervals

    Args:
        ecg_data: bp data from the record
        ecg_nans: indexes in ecg_data which are nan

    Returns:
        A processed ecg record with nan values averaged out
    """
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
    """Compresses continuous ABP data to larger intervals

        Args:
            bp: bp data from the record
            fs: the frequency of the ecg data

        Returns:
            A numpy array of length n with the average bp across each time interval
        """
    window_size = fs * time_step
    i = 0
    out = np.array([])
    bp_len = len(bp)
    # allow one peak every half of a second (works well for most heart rates of sleeping ~1/s)
    distance_btwn_peaks = 2 / fs
    while i + window_size <= bp_len:
        window = bp[i: i + window_size]
        peaks, _ = find_peaks(window, distance=distance_btwn_peaks)
        troughs, _ = find_peaks(-window, distance=distance_btwn_peaks)

        systolic = np.median(window[peaks])
        diastolic = np.median(window[troughs])

        out = np.append(out, [systolic, diastolic])
        i += window_size
    return out


def calculate_sleep_stages(ecg, fs):
    """Runs sleep staging

    Args:
        ecg: ecg data from the record
        fs: the frequency of the ecg data

    Returns:
        A numpy array of shape nx4 with the probabilities of each sleep stage for each interval
    """

    # detect heartbeats
    beats = sleepecg.detect_heartbeats(ecg, fs)
    sleepecg.plot_ecg(ecg, fs, beats=beats)

    record = sleepecg.SleepRecord(
        sleep_stage_duration=30,
        heartbeat_times=beats / fs,
    )

    # predict sleep stages
    stages = sleepecg.stage(clf, record, return_mode="prob")
    return stages


#
def convert_record(patient, record, error_path):
    """Converts a record into sleep cycles / abp per time_step

    Args:
        patient: The ID of the patient as a path (e.g. "p00/p000079")
        record: the Record object to convert
        error_path: filepath to error file

    Returns:
        A dictionary with keys "sleep_stages" and "blood_pressure" and the corresponding data
    """
    try:
        # read the record from physionet
        record = wfdb.rdrecord(record, pn_dir=f"{MIMIC}{patient}")

        # read ecg data
        ecg_index = record.sig_name.index(ecg_channel_name)
        ecg_data = record.p_signal[:, ecg_index].astype(np.float64)

        # read abp data
        abp_index = record.sig_name.index(abp_channel_name)
        abp_data = record.p_signal[:, abp_index].astype(np.float64)

        # get an array of indices that are NaN in ECG data
        ecg_nans = np.isnan(ecg_data)
        # skip segment if there are too many nans
        if len(ecg_data) - sum(ecg_nans) < NAN_THRESHOLD * len(ecg_data):
            print(f"skipped segment {patient} {record} due to high nan-rate")
            return
        # preprocess ECG data
        ecg_data = lerp_ecg(ecg_data, ecg_nans)
        # run sleep staging
        sleep_stages = calculate_sleep_stages(ecg_data, record.fs)
        # calculate abp segments
        bp_intervals = process_bp_data(abp_data, record.fs)

        return {
            "sleep_stages": sleep_stages.tolist(),
            "blood_pressure": bp_intervals.tolist()
        }

    except Exception:
        error_write(error_path, f"Problem converting record [Patient: {patient} Record: {record}]")
        # raise


def write_to_file(patient, segment_name, converted, segment_datetime):
    """Write patient data to file.

    Args:
        patient: The ID of the patient as a path (e.g. "p00/p000079")
        segment_name: The name of the segment
        converted: The output ss/bp data
        segment_datetime: The datetime for the segment that was processed

    Returns:
        None
    """
    os.makedirs(f"data/patients_cvd/{patient}", exist_ok=True)
    with open(f"data/patients_cvd/{patient}/{segment_name}.json", "w") as f:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        converted["last_update"] = now
        converted["date"] = str(segment_datetime)
        json.dump(converted, f, indent=2)
        print(f"saved record {patient}{segment_name}")


#
#
def scan_mimic(error_path, override_list=None, starting_point=None):
    """
    Iterates through the patients in mimic, finds records which are valid for ss processing
    and converts them, writing the result to a directory in data/patients

    Args:
        error_path: filename to write error messages to

    Returns:
        None
    """

    # get the list of patients from the db, or use override list if provided
    record_list = override_list if override_list is not None else wfdb.get_record_list(MIMIC)

    # jump to a predetermined starting point
    shortened_record_list = record_list[
                            record_list.index(f"{starting_point[0:3]}/{starting_point}/") if starting_point is not None \
                                else record_list:]
    for patient in shortened_record_list:
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
                                print(f"skipped patient record {patient} {header.record_name} due to missing signals")
                                break
                        elif segment is not None \
                                and segment.sig_len > required_segment_len * segment.fs \
                                and segment.fs >= min_frequency:
                            has_required_signals = not (False in [(x in segment.sig_name) for x in required_signals])
                            if has_required_signals:
                                converted = convert_record(patient, segment.record_name, error_path)
                                if converted is not None:
                                    write_to_file(patient, segment.record_name, converted, header.base_datetime)
                            else:
                                print(f"skipped segment {patient} {segment.record_name} due to missing signals")
                    except Exception as e:
                        error_write(error_path,
                                    f"Problem parsing patient record [patient: {patient}, record: {segment.record_name if segment else '--'}]")
                        # raise

        except Exception as e:
            error_write(error_path, f"Problem with patient [Patient: {patient}]")
            # raise


#
def error_write(filename, message):
    """Writes a line to the error file

    Args:
        filename: filename to write error messages to
        message: error msg

    Returns:
        None
    """
    with open(f"data/{filename}", "a") as f:
        f.write(message + "\n")
        print(message)


if __name__ == "__main__":
    # write a timestamp of the run to the error file
    error_file = "errors.txt"
    error_write(error_file, f"---------- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # run process
    override_list = None
    starting_point = None
    scan_mimic(error_file, override_list, starting_point)
