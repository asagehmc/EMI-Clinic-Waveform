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
    rec_start = datetime.combine(edf.startdate, edf.starttime)

    # Getting the ECG time series and sampling frequency
    signal_type = "EKG" if dataset == "mesa-sleep" else "ECG"
    ecg = edf.get_signal(signal_type).data
    fs = edf.get_signal(signal_type).sampling_frequency

    print(f"ECG duration available: {len(ecg) / fs:.2f} seconds")

    try:
        beats = sleepecg.detect_heartbeats(ecg, fs)
    except Exception as e:
        print(f"Error detecting heartbeats for file {path} with fs={fs}: {e}")
        return None

    # Loading nonweighted classifier
    clf = sleepecg.load_classifier("wrn-gru-mesa", "SleepECG")

    # Predicting sleep stages
    record = sleepecg.SleepRecord(
        sleep_stage_duration=30,
        recording_start_time=rec_start,
        heartbeat_times=beats / fs,
    )
    stages = sleepecg.stage(clf, record, return_mode="prob")
    stage_labels = np.argmax(stages, axis=1)
    mapping = {0: "Undefined", 1: "Non-REM", 2: "REM sleep", 3: "Wake"}
    stage_labels_named = [mapping[i] for i in stage_labels]

    # Coverting multi-stage to binary: awake if 'Wake', otherwise sleep
    binary_labels = []
    for label in stage_labels_named:
        if label == 'Wake':
            binary_labels.append('awake')
        else:
            binary_labels.append('sleep')
    return binary_labels

def parse_shhs_xml(xml_file, epoch_length=30):
    """
    Parsing an SHHS XML file to extract sleep stage annotations.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    scored_events = root.findall("ScoredEvents/ScoredEvent")
    sleep_events = [event for event in scored_events
                    if event.find("EventType").text == "Stages|Stages"]

    sleep_stages = []
    start = 0
    for event in sleep_events:
        duration = float(event.find("Duration").text)
        stage = event.find("EventConcept").text[:-2]
        if stage not in ["Wake", "REM sleep"]:
            stage = "Non-REM"
        num_epochs = int(duration / epoch_length)
        for _ in range(num_epochs):
            end = start + epoch_length
            sleep_stages.append([start, epoch_length, end, stage])
            start = end
    sleep_stages = np.array(sleep_stages)
    print(f"Total duration from XML: {sleep_stages[-1][2]} seconds")
    return sleep_stages

def compare(pred, real, output_conf_file=None):
    """
    Comparing nonweighted (binary) predictions to the ground truth:
        Wake' -> 'awake'; all others -> 'sleep'
    """
    # Extracting stage column from XML annotations
    real_stages = real[:, -1]
    real_binary = np.array(['awake' if stage == 'Wake' else 'sleep' for stage in real_stages])
    pred_mapped = ['awake' if p == 'awake' else 'sleep' for p in pred]

    min_len = min(len(real_binary), len(pred_mapped))
    real_binary = real_binary[:min_len]
    pred_mapped = pred_mapped[:min_len]

    acc = accuracy_score(real_binary, pred_mapped)
    kappa = cohen_kappa_score(real_binary, pred_mapped)
    conf_matrix = confusion_matrix(real_binary, pred_mapped)

    print(f"Accuracy: {acc:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)

    if output_conf_file is not None:
        labels = ['awake', 'sleep']
        conf_df = pd.DataFrame(conf_matrix,
                               index=[f"True_{l}" for l in labels],
                               columns=[f"Pred_{l}" for l in labels])
        conf_df.to_csv(output_conf_file)
        print(f"Nonweighted confusion matrix saved to {output_conf_file}")

    return acc

def save_predictions(pred, file_path):
    """
    Saving the nonweighted predicted sleep stage annotations to a CSV file
    """
    df = pd.DataFrame({"Stage": pred})
    df.to_csv(file_path, index=False)
    print(f"Nonweighted predictions saved to {file_path}")

if __name__ == "__main__":
    # Choosing the dataset: 'shhs1' or 'mesa-sleep'
    dataset = "shhs1"
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
    file_count = 0
    accuracy_records = []

    # Creating output folders for nonweighted predictions and confusion matrices
    output_folder = "output_predictions/nonweighted_predictions/"
    os.makedirs(output_folder, exist_ok=True)
    conf_output_folder = "output_confusion_matrices/nonweighted/"
    os.makedirs(conf_output_folder, exist_ok=True)

    for edf in edfs:
        if edf.endswith(".edf"):
            edf_full_path = os.path.join(edf_path, edf)
            id_num = edf[-10:-4]
            xml_file = os.path.join(xml_path, f"{dataset}-{id_num}-nsrr.xml")
            file_count += 1
            print(f"\nProcessing nonweighted file: {edf_full_path}")
            print(f"Using XML file: {xml_file}")

            pred = ecg_staging(edf_full_path, dataset)
            if pred is None:
                print(f"Skipping file {edf_full_path} due to heartbeat detection error.")
                continue

            real = parse_shhs_xml(xml_file)
            # Defining unique filenames for the confusion matrix
            conf_file = os.path.join(conf_output_folder, f"{os.path.splitext(edf)[0]}_nonwgt_confusion_matrix.csv")
            acc = compare(pred, real, output_conf_file=conf_file)
            acc_sum += acc
            print(f"File {file_count} Nonweighted Accuracy: {acc:.4f}, Running average: {acc_sum / file_count:.4f}")

            accuracy_records.append({
                "File": os.path.basename(edf_full_path),
                "Accuracy": acc
            })

            output_file = os.path.join(output_folder, f"{os.path.splitext(edf)[0]}_nonwgt_predictions.csv")
            save_predictions(pred, output_file)

    overall_accuracy = acc_sum / file_count if file_count > 0 else 0
    print(f"\nOverall Nonweighted Accuracy: {overall_accuracy:.4f}")
    acc_df = pd.DataFrame(accuracy_records)
    acc_output_file = os.path.join(output_folder, "nonwgt_accuracy_records.csv")
    acc_df.to_csv(acc_output_file, index=False)
    print(f"Nonweighted accuracy records saved to {acc_output_file}")