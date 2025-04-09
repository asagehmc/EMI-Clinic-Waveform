
from datetime import datetime
from edfio import read_edf
import sleepecg
import numpy as np
import os
import xml.etree.ElementTree as ET
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
import scipy.signal

# Monkey-patch scipy.signal.butter to adjust the cutoff frequency if needed.
def custom_butter(N, Wn, btype, fs, **kwargs):
    # If Wn is a float, ensure it's strictly less than fs/2.
    if isinstance(Wn, (int, float)):
        if Wn >= fs / 2:
            Wn = fs / 2 - 0.1  # reduce slightly below the Nyquist limit.
    else:
        # If Wn is a list or tuple, adjust each element.
        Wn = [w if w < fs / 2 else fs / 2 - 0.1 for w in Wn]
    return scipy.signal.iirfilter(N, Wn, btype=btype, analog=False, ftype='butter', fs=fs, **kwargs)

scipy.signal.butter = custom_butter

def combined_ecg_staging(path, dataset):
    """
    Runs both the nonweighted and weighted algorithms on a given EDF file and combines their outputs.

    Nonweighted is used for binary sleep/wake classification and weighted for finer sleep staging.
    For each epoch, if the nonweighted classifier says "awake", then the final label is "awake".
    Otherwise, the final label is the weighted prediction (with a safeguard: if it comes back as "wake", map it to "nrem").
    """
    edf = read_edf(path)
    rec_start = datetime.combine(edf.startdate, edf.starttime)

    # Get the ECG signal and sampling frequency.
    signal_type = "EKG" if dataset == "mesa-sleep" else "ECG"
    ecg = edf.get_signal(signal_type).data
    fs = edf.get_signal(signal_type).sampling_frequency
    print(f"ECG duration available: {len(ecg) / fs:.2f} seconds")

    # Detect heartbeats (run once for both algorithms)
    try:
        beats = sleepecg.detect_heartbeats(ecg, fs)
    except Exception as e:
        print(f"Error detecting heartbeats for file {path} with fs={fs}: {e}")
        return None  # Skip this file if error occurs

    # Build the SleepRecord (30-second epochs)
    record = sleepecg.SleepRecord(
        sleep_stage_duration=30,
        recording_start_time=rec_start,
        heartbeat_times=beats / fs,
    )

    # ---- NONWEIGHTED ALGORITHM (binary sleep/wake) ----
    clf_nonwgt = sleepecg.load_classifier("wrn-gru-mesa", "SleepECG")
    stages_nonwgt = sleepecg.stage(clf_nonwgt, record, return_mode="prob")
    stage_labels_nonwgt = np.argmax(stages_nonwgt, axis=1)

    # Mapping for nonweighted: index 3 corresponds to Wake.
    mapping_nonwgt = {0: "Undefined", 1: "Non-REM", 2: "REM sleep", 3: "Wake"}
    stage_labels_nonwgt_named = [mapping_nonwgt[i] for i in stage_labels_nonwgt]

    # Convert to binary: "awake" if label is "Wake", otherwise "sleep"
    binary_labels = ['awake' if label == "Wake" else 'sleep' for label in stage_labels_nonwgt_named]

    # ---- WEIGHTED ALGORITHM (finer staging) ----
    clf_wgt = sleepecg.load_classifier("wrn-gru-mesa-weighted", "SleepECG")
    stages_wgt = sleepecg.stage(clf_wgt, record, return_mode="prob")
    stage_labels_wgt = np.argmax(stages_wgt, axis=1)

    # Mapping for weighted: we expect indices to map to "nrem", "nrem", "rem", "wake"
    mapping_wgt = {0: "nrem", 1: "nrem", 2: "rem", 3: "wake"}
    stage_labels_wgt_named = [mapping_wgt[i] for i in stage_labels_wgt]

    # ---- COMBINE THE PREDICTIONS ----
    final_labels = []
    for bin_label, wgt_label in zip(binary_labels, stage_labels_wgt_named):
        if bin_label == "awake":
            final_labels.append("awake")
        else:
            # If weighted unexpectedly predicts "wake" in a sleep epoch, default to "nrem"
            final_labels.append(wgt_label if wgt_label != "wake" else "nrem")

    return final_labels


def parse_shhs_xml_combined(xml_file, epoch_length=30):
    """
    Parses an SHHS XML file to extract sleep stage annotations.
    Each event is broken into epochs of given duration.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    scored_events = root.findall("ScoredEvents/ScoredEvent")
    # Only select events for sleep staging.
    sleep_events = [event for event in scored_events if event.find("EventType").text == "Stages|Stages"]

    sleep_stages = []
    start = 0
    for event in sleep_events:
        duration = float(event.find("Duration").text)
        stage = event.find("EventConcept").text[:-2]  # remove last 2 characters (if any formatting)
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


def compare_combined(pred, real):
    """
    Compares combined predictions to the ground truth.
    Ground truth is mapped as:
        "Wake"       -> "awake"
        "REM sleep"  -> "rem"
        All others   -> "nrem"
    """
    # Extract the stage labels (fourth column)
    real_stages = real[:, -1]
    real_mapped = np.array(['awake' if stage == 'Wake'
                            else 'rem' if stage == 'REM sleep'
                            else 'nrem' for stage in real_stages])

    pred = np.array(pred)
    # Truncate to the shortest length if needed.
    min_len = min(len(real_mapped), len(pred))
    real_mapped = real_mapped[:min_len]
    pred = pred[:min_len]

    acc = accuracy_score(real_mapped, pred)
    kappa = cohen_kappa_score(real_mapped, pred)
    conf_matrix = confusion_matrix(real_mapped, pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)

    return acc


if __name__ == "__main__":
    # Set dataset (shhs1) and corresponding paths.
    dataset = "shhs1"  # OPTIONS: 'shhs1' or 'mesa-sleep'
    if dataset == "mesa-sleep":
        edf_path = "mesa/polysomnography/edfs/"
        xml_path = "mesa/polysomnography/annotations-events-nsrr/"
    else:
        edf_path = "shhs/polysomnography/edfs/shhs1/"
        xml_path = "shhs/polysomnography/annotations-events-nsrr/shhs1/"

    edf_files = os.listdir(edf_path)
    xml_files = os.listdir(xml_path)
    num_files = min(len(edf_files), len(xml_files))
    acc_sum = 0
    file_count = 0
    accuracy_records = []

    # Create an output folder for combined predictions.
    output_folder = "output_predictions_combined/"
    os.makedirs(output_folder, exist_ok=True)

    # Process each EDF file.
    for edf in edf_files:
        if edf.endswith(".edf"):
            edf_full_path = os.path.join(edf_path, edf)
            # For shhs1, extract the ID from the filename (using the last 10 to -4 characters).
            id_num = edf[-10:-4]
            xml_file = os.path.join(xml_path, f"{dataset}-{id_num}-nsrr.xml")
            file_count += 1
            print(f"\nProcessing file: {edf_full_path}")
            print(f"Using XML file: {xml_file}")

            # Get the combined predictions.
            combined_pred = combined_ecg_staging(edf_full_path, dataset)
            if combined_pred is None:
                print(f"Skipping file {edf_full_path} due to heartbeat detection error.")
                continue

            # Parse the XML annotations.
            real = parse_shhs_xml_combined(xml_file)
            # Evaluate combined predictions.
            acc = compare_combined(combined_pred, real)
            acc_sum += acc
            print(f"File {file_count} Accuracy: {acc:.4f}, Current average: {acc_sum / file_count:.4f}")

            accuracy_records.append({
                "File": os.path.basename(edf_full_path),
                "Accuracy": acc
            })

            # Save the combined predictions to a CSV file.
            output_file = os.path.join(output_folder, f"{os.path.splitext(edf)[0]}_combined_predictions.csv")
            df = pd.DataFrame({"Stage": combined_pred})
            df.to_csv(output_file, index=False)
            print(f"Combined predictions saved to {output_file}")

    overall_accuracy = acc_sum / file_count if file_count > 0 else 0
    print(f"\nOverall Accuracy: {overall_accuracy:.4f}")

    # Save the accuracy summary.
    acc_df = pd.DataFrame(accuracy_records)
    acc_output_file = os.path.join(output_folder, "combined_accuracy_records.csv")
    acc_df.to_csv(acc_output_file, index=False)
    print(f"Accuracy records saved to {acc_output_file}")

