from datetime import datetime
from edfio import read_edf
import sleepecg
import numpy as np
import os
import xml.etree.ElementTree as ET
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score


def ecg_staging(path, dataset, weighted=False):
    edf = read_edf(path)

    # crop dataset (we only want data for the sleep duration)
    rec_start = datetime.combine(edf.startdate, edf.starttime)

    # get ECG time series and sampling frequency
    if dataset == "mesa-sleep":
        signal_type = "EKG"
    else:
        signal_type = "ECG"
    ecg = edf.get_signal(signal_type).data
    fs = edf.get_signal(signal_type).sampling_frequency

    print(f"ECG duration available: {len(ecg) / fs} seconds")

    # detect heartbeats
    beats = sleepecg.detect_heartbeats(ecg, fs)

    # choose classifier based on whether it is weighted or not
    if weighted:
        clf = sleepecg.load_classifier("wrn-gru-mesa-weighted", "SleepECG")
    else:
        clf = sleepecg.load_classifier("wrn-gru-mesa", "SleepECG")

    # predict sleep stages
    record = sleepecg.SleepRecord(
        sleep_stage_duration=30,
        recording_start_time=rec_start,
        heartbeat_times=beats / fs,
    )

    stages = sleepecg.stage(clf, record, return_mode="prob")

    stage_labels = np.argmax(stages, axis=1)  # Get index of max probability per row

    stage_mapping = {0: "Undefined", 1: "Non-REM", 2: "REM sleep", 3: "Wake"}
    stage_labels_named = [stage_mapping[i] for i in stage_labels]

    return stage_labels_named


def parse_shhs_xml(xml_file, epoch_length=30):
    """
    Parses an SHHS XML file to extract sleep stage annotations.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    scored_events = root.findall("ScoredEvents/ScoredEvent")
    sleep_events = [scored_event for scored_event in scored_events if
                    scored_event.find("EventType").text == "Stages|Stages"]

    sleep_stages = []

    start = 0
    for event in sleep_events:
        duration = float(event.find("Duration").text)
        sleep_stage = event.find("EventConcept").text[:-2]
        if sleep_stage not in ["Wake", "REM sleep"]:
            sleep_stage = "Non-REM"
        num_sub_events = int(duration / epoch_length)
        for i in range(num_sub_events):
            end = start + epoch_length
            sleep_stages.append([start, epoch_length, end, sleep_stage])
            start = end

    sleep_stages = np.array(sleep_stages)

    print(f"total duration: {sleep_stages[-1][2]}")

    return sleep_stages


def compare(pred, real):
    real = real[:, -1]

    print(len(real))
    print(len(pred))
    min_len = min(len(real), len(pred))
    real = real[:min_len]
    pred = pred[:min_len]

    acc = accuracy_score(real, pred)
    kappa = cohen_kappa_score(real, pred)
    conf_matrix = confusion_matrix(real, pred)

    # print(f"Accuracy: {acc:.4f}")
    # print(f"Cohen’s Kappa: {kappa:.4f}")
    # print("Confusion Matrix:")
    # print(conf_matrix)

    return acc


def save_predictions(pred, file_path):
    # Function to save the weighted + nonweighted predictions to a CSV file
    df = pd.DataFrame(pred, columns=["Time", "Stage"])
    df.to_csv(file_path, index=False)


if __name__ == "__main__":

    # set this to the dataset we want to use
    dataset = "shhs1"  # OPTIONS: 'shhs1' or 'mesa-sleep'

    if dataset not in ['mesa-sleep', 'shhs1']:
        raise ValueError("dataset MUST be one of mesa-sleep or shhs1")
    if dataset == "mesa-sleep":
        edf_path = "mesa/polysomnography/edfs/"
        xml_path = "mesa/polysomnography/annotations-events-nsrr/"
    else:
        edf_path = "shhs/polysomnography/edfs/shhs1/"
        xml_path = "shhs/polysomnography/annotations-events-nsrr/shhs1/"

    edfs = os.listdir(edf_path)
    xmls = os.listdir(xml_path)
    num_files = min(len(edfs), len(xmls))
    acc_sum = 0

    num = 0
    non_weighted_preds = []
    weighted_preds = []

    for edf in edfs:
        if edf[-4:] == ".edf":
            edf_path_full = f"{edf_path}{edf}"
            id_num = edf_path_full[(-10 + (2 * int(dataset == "mesa-sleep"))):-4]
            xml = f"{xml_path}{dataset}-{id_num}-nsrr.xml"
            num += 1
            print(edf_path_full)
            print(xml)

            # Non-weighted classifier for wake detection
            pred_non_weighted = ecg_staging(edf_path_full, dataset, weighted=False)
            non_weighted_preds.append(pred_non_weighted)

            # Weighted classifier for sleep stage detection
            pred_weighted = ecg_staging(edf_path_full, dataset, weighted=True)
            weighted_preds.append(pred_weighted)

            real = parse_shhs_xml(xml)
            acc_non_weighted = compare(pred_non_weighted, real)
            acc_weighted = compare(pred_weighted, real)
            acc_sum += acc_non_weighted
            print(f"Pass: {num}, Non-weighted Accuracy: {acc_non_weighted}, Weighted Accuracy: {acc_weighted}")

    accuracy_non_weighted = acc_sum/num
    print(f"Non-weighted Accuracy: {accuracy_non_weighted:.4f}")

    # Saving results to two different folders
    output_non_weighted = "output/non_weighted_predictions/"
    output_weighted = "output/weighted_predictions/"

    os.makedirs(output_non_weighted, exist_ok=True)
    os.makedirs(output_weighted, exist_ok=True)

    # Save both of our predictions to two different folders
    for i, edf in enumerate(edfs):
        if edf[-4:] == ".edf":
            # Non-weighted predictions
            save_predictions(non_weighted_preds[i], f"{output_non_weighted}{edf[:-4]}_non_weighted_predictions.csv")
            # Weighted predictions
            save_predictions(weighted_preds[i], f"{output_weighted}{edf[:-4]}_weighted_predictions.csv")

    print(f"Predictions saved to {output_non_weighted} and {output_weighted}")