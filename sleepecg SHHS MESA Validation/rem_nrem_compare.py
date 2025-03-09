from datetime import datetime
from edfio import read_edf
import sleepecg
import numpy as np
import os
import xml.etree.ElementTree as ET
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score


def ecg_staging(path, dataset):
    edf = read_edf(path)

    # crop dataset (we only want data for the sleep duration)
    # start = datetime(2023, 3, 1, 23, 0, 0)
    # stop = datetime(2023, 3, 2, 6, 0, 0)
    rec_start = datetime.combine(edf.startdate, edf.starttime)
    # edf.slice_between_seconds((start - rec_start).seconds, (stop - rec_start).seconds)

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

    # load SleepECG nonweighted "wrn-gru-mesa" classifier (requires tensorflow)
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
    sleep_events = [scored_event for scored_event in scored_events if scored_event.find("EventType").text == "Stages|Stages"]

    sleep_stages = []
    
    start = 0
    for event in sleep_events:
        duration = float(event.find("Duration").text)
        sleep_stage = event.find("EventConcept").text[:-2]
        if sleep_stage not in ["Wake", "REM sleep"]:
            sleep_stage = "Non-REM"
        num_sub_events = int(duration/epoch_length)
        for i in range(num_sub_events):
            end = start + epoch_length
            sleep_stages.append([start,epoch_length, end,sleep_stage])
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

    #print(f"Accuracy: {acc:.4f}")
    #print(f"Cohen’s Kappa: {kappa:.4f}")
    #print("Confusion Matrix:")
    #print(conf_matrix)

    return acc

def save_predictions(pred, file_path):
    """
    Saves the nonweighted predicted sleep stage annotations to a CSV file
    """
    # Convert the nonweighted predictions to a DataFrame
    df = pd.DataFrame(pred, columns=["Time", "Stage"])
    df.to_csv(file_path, index=False)
    print(f"Predictions saved to {file_path}")

if __name__ == "__main__":

    # set this to the dataset we want to use
    dataset = "shhs1" #OPTIONS: 'shhs1' or 'mesa-sleep'


    if dataset not in ['mesa-sleep', 'shhs1']:
        raise ValueError("dataset MUST be on of mesa-sleep or shhs1")
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
    for edf in edfs:
        if edf[-4:] == ".edf":
            edf = f"{edf_path}{edf}"
            id_num = edf[(-10 + (2*int(dataset=="mesa-sleep"))):-4]
            xml = f"{xml_path}{dataset}-{id_num}-nsrr.xml"
            num +=1
            print(edf)
            print(xml)
            pred = ecg_staging(edf, dataset)
            real = parse_shhs_xml(xml)
            acc = compare(pred, real)
            acc_sum += acc
            print(f"Pass: {num}, Accuracy: {acc}, Current Accuracy: {acc_sum/(num)}")

            # Save the nonweighted predictions to a CSV file
            output_folder = "output_predictions/"
            os.makedirs(output_folder, exist_ok=True)
            save_predictions(pred, f"{output_folder}{edf[:-4]}_nonwgt_predictions.csv")

    accuracy = acc_sum/(num)
    print(f"Accuracy: {accuracy:.4f}")




