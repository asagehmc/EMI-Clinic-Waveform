import csv
import os
import subprocess
from math import ceil, floor

import numpy as np
import pyedflib
import xmltodict


def get_data_for_patient(id, nsrr_path="nsrr"):
    """
    download patient edf, read file and get sleep annotations, ppg signal out
    :param id: the record id to read
    :param nsrr_path: the path to the patient data to download
    :return: object containing ppg, patient data, and sleep staging data
    """
    signals_path = f"mesa/polysomnography/edfs/mesa-sleep-{id:04}.edf"
    annotations_path = f"mesa/polysomnography/annotations-events-nsrr/mesa-sleep-{id:04}-nsrr.xml"

    download_data(signals_path, nsrr_path)
    download_data(annotations_path, nsrr_path)

    # read the pleth signal
    f = pyedflib.EdfReader(f"mesa/polysomnography/edfs/mesa-sleep-{id:04}.edf")
    signal_labels = f.getSignalLabels()
    ppg_signal = f.readSignal(signal_labels.index("Pleth"))
    ppg_freq = f.getSampleFrequency(signal_labels.index("Pleth"))

    # read the sleep annotations
    with open(annotations_path, 'r', encoding='utf-8') as f:
        xml_string = f.read()
    ss_data = xmltodict.parse(xml_string)

    ss_annot = [x for x in ss_data["PSGAnnotation"]["ScoredEvents"]["ScoredEvent"] if x["EventType"] == "Stages|Stages"]
    ss_epochs = annotation_to_ss_epochs(ss_annot)

    patient_data = get_ith_patient_data(patient_id=id)

    return {
        "ppg": ppg_signal,
        "ppg_freq": int(ppg_freq),
        "ss": ss_epochs.tolist(),
        "patient_data": patient_data
    }


def get_ith_patient_data(patient_id):
    """
    read the MESA sleep data sheet and pull out/parse the selected elements
    :param patient_id: the patient ID as a number
    :return: an object containing parsed elements from the patient data table
    """
    ints = ["race1c", "gender1", "sleepage5c"]
    floats = ["htcm5", "wtlb5", "bmi5c", "nsrr_ahi_hp3r_aasm15"]
    strings = ["nsrr_age_gt89"]
    keys_to_keep = ints + floats + strings

    with open("mesa-sleep-dataset-0.7.0.csv", 'r') as file:
        reader = csv.reader(file)
        keys = []
        for i, row in enumerate(reader):
            if i == 0:
                keys = row
            if i == patient_id:
                patient_data = row
                out = {}
                for key_to_keep in keys_to_keep:
                    value = patient_data[keys.index(key_to_keep)]

                    # Convert to number type (or keep as string)
                    if key_to_keep in ints and value.strip():
                        try:
                            out[key_to_keep] = int(value)
                        except ValueError:
                            out[key_to_keep] = value
                    elif key_to_keep in floats and value.strip():
                        try:
                            out[key_to_keep] = float(value)
                        except ValueError:
                            out[key_to_keep] = value
                    else:
                        out[key_to_keep] = value

                return out
    return None


def download_data(path, nsrr_exe):
    """
    Downloads the edf and sleep annotation files for the given patient path
    :param path: patient path to download
    :param nsrr_exe: the nsrr gem path
    """
    if not os.path.exists(path):
        with open("token.txt", "r") as f:
            nsrr_token = f.readline().strip()

        result = subprocess.run(
            [nsrr_exe, "download", path, f"--token={nsrr_token}"],
            capture_output=True,
            text=True
        )
        # super janky way to do this, but works for now.
        if "1 file downloaded" in result.stdout:
            print(f"Successfully downloaded file {path}!")
        if "1 file skipped" in result.stdout:
            print(f"Skipped downloading file {path}!")
    else:
        print(f"Already downloaded file {path}!")


def annotation_to_ss_epochs(annotations):
    """
    Converts sleep stage annotations to time steps of 30s intervals
    :param annotations: the annotation object to convert
    :return: an array of sleep stage values
    """
    # Preprocess annotations: convert to (start, end, stage)
    processed_annots = [
        (float(a['Start']), float(a['Start']) + float(a['Duration']), int(a['EventConcept'].split('|')[1]))
        for a in annotations
    ]

    # Determine how long the recording is
    end_time = max(a[1] for a in processed_annots)
    n_epochs = floor(end_time / 30)

    # Initialize all epochs to 0
    result = [0] * n_epochs

    # go through the list of epochs, assign sleep stages
    for i in range(n_epochs):
        epoch_start = i * 30
        for start, end, stage in processed_annots:
            if start <= epoch_start < end:
                result[i] = stage
                break  # Use the first matching annotation

    return np.array(result)


def get_record_ids():
    """
    :return: a list of records from the mesa sleep csv
    """
    ids = []
    with open("mesa-sleep-dataset-0.7.0.csv", 'r') as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            if i != 0:
                ids += [int(row[0])]
    return ids


if __name__ == "__main__":
    a = get_data_for_patient(2)


