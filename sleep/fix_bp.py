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
    out = []
    bp_len = len(bp)
    # allow one peak every half of a second (works well for most heart rates of sleeping ~1/s)
    distance_btwn_peaks = fs/2
    while i + window_size <= bp_len:
        window = bp[i: i + window_size]
        peaks, _ = find_peaks(window, distance=distance_btwn_peaks)
        troughs, _ = find_peaks(-window, distance=distance_btwn_peaks)

        systolic = np.median(window[peaks])
        diastolic = np.median(window[troughs])

        out += [systolic, diastolic]
        i += window_size
    return out


def get_sleep_stages(patient_path):
    """Reads sleep staging from existing files

    Args:
        record_name: name of the record to read

    Returns:
        A numpy array of shape nx4 with the probabilities of each sleep stage for each interval
    """
    path = f"data/patients/{patient_path}.json"
    sleep = []
    print(path)
    if os.path.isfile(path):
        with open(path, "r") as f:
            data = json.load(f)
            sleep = data["sleep_stages"]
    else:
        path = f"data/patients_cvd/{patient_path}.json"
        print(path)
        if os.path.isfile(path):
            with open(path, "r") as f:
                data = json.load(f)
                sleep = data["sleep_stages"]
        else:
            print(f"MISSING FILE {patient_path}")
    return sleep


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

        # read abp data
        abp_index = record.sig_name.index(abp_channel_name)
        abp_data = record.p_signal[:, abp_index].astype(np.float64)

        # steal sleep staging from existing set
        sleep_stages = get_sleep_stages(f"{patient}{record.record_name}")
        # calculate abp segments
        bp_intervals = process_bp_data(abp_data, record.fs)

        return {
            "sleep_stages": sleep_stages,
            "blood_pressure": bp_intervals
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
    os.makedirs(f"data/fixed_patients/{patient}", exist_ok=True)
    with open(f"data/fixed_patients/{patient}/{segment_name}.json", "w") as f:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        converted["last_update"] = now
        converted["date"] = str(segment_datetime)
        json.dump(converted, f, indent=2)
        print(f"saved record {patient}{segment_name}")


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
    if starting_point is not None:
        shortened_record_list = record_list[record_list.index(f"{starting_point[0:3]}/{starting_point}/") if starting_point is not None else record_list:]
    else:
        shortened_record_list = record_list

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
    override_list = ['p09/p095708/`', 'p09/p095235/', 'p09/p090560/', 'p07/p077280/', 'p07/p072377/', 'p07/p072273/', 'p07/p072197/', 'p06/p060180/', 'p05/p050804/', 'p05/p050767/', 'p05/p050735/', 'p05/p050721/', 'p05/p050710/', 'p05/p050664/', 'p04/p044715/', 'p04/p040624/', 'p02/p025225/', 'p02/p024942/', 'p02/p024924/', 'p02/p024923/', 'p02/p024922/', 'p02/p021507/', 'p00/p002639/', 'p00/p002636/', 'p00/p002577/', 'p00/p002561/', 'p00/p002549/', 'p00/p002514/', 'p00/p002513/', 'p00/p002492/', 'p00/p002488/', 'p00/p002467/', 'p00/p002442/', 'p00/p002397/', 'p00/p002395/', 'p00/p002369/', 'p00/p002361/', 'p00/p002340/', 'p00/p002317/', 'p00/p002280/', 'p00/p002264/', 'p00/p002251/', 'p00/p002240/', 'p00/p002229/', 'p00/p002224/', 'p00/p002213/', 'p00/p002185/', 'p00/p002090/', 'p00/p002049/', 'p00/p002045/', 'p00/p001978/', 'p00/p001973/', 'p00/p001950/', 'p00/p001949/', 'p00/p001944/', 'p00/p001941/', 'p00/p001932/', 'p00/p001931/', 'p00/p001900/', 'p00/p001892/', 'p00/p001855/', 'p00/p001854/', 'p00/p001840/', 'p00/p001824/', 'p00/p001791/', 'p00/p001693/', 'p00/p001650/', 'p00/p001606/', 'p00/p001604/', 'p00/p001586/', 'p00/p001524/', 'p00/p001502/', 'p00/p001501/', 'p00/p001485/', 'p00/p001449/', 'p00/p001418/', 'p00/p001414/', 'p00/p001408/', 'p00/p001396/', 'p00/p001357/', 'p00/p001331/', 'p00/p001313/', 'p00/p001279/', 'p00/p001244/', 'p00/p001222/', 'p00/p001217/', 'p00/p001190/', 'p00/p001158/', 'p00/p001144/', 'p00/p001121/', 'p00/p001072/', 'p00/p001046/', 'p00/p001038/', 'p00/p001028/', 'p00/p001012/', 'p00/p000963/', 'p00/p000894/', 'p00/p000865/', 'p00/p000843/', 'p00/p000801/', 'p00/p000772/', 'p00/p000770/', 'p00/p000735/', 'p00/p000708/', 'p00/p000703/', 'p00/p000695/', 'p00/p000682/', 'p00/p000652/', 'p00/p000638/', 'p00/p000625/', 'p00/p000618/', 'p00/p000608/', 'p00/p000543/', 'p00/p000515/', 'p00/p000470/', 'p00/p000439/', 'p00/p000333/', 'p00/p000328/', 'p00/p000318/', 'p00/p000308/', 'p00/p000283/', 'p00/p000263/', 'p00/p000214/', 'p00/p000208/', 'p00/p000177/', 'p00/p000125/', 'p00/p000124/', 'p00/p000123/', 'p00/p000107/', 'p00/p000079/', 'p00/p000020/']
    scan_mimic(error_file, override_list)
