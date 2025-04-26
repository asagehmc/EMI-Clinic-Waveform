import json
import sys

import numpy as np
import os

from scipy.signal import find_peaks, resample

from PPGtoBP.PPG_model.helper_functions_bp_model import predict_bp_from_ppg, normalize_min_max
from PPGtoBP.download_nsrr import get_record_ids
from download_nsrr import get_data_for_patient


def extract_ppg_episodes(ppg, fs=125, episode_len=10, step_len=5):
    """
    Converts ppg signal into blocks for processing (converted to 125hz)
    :param ppg: signal to block
    :param fs: frequency of incoming signal
    :param episode_len: the length of the block in seconds
    :param step_len: the interval of time between blocks
    :return:
    """
    FS = fs
    EPISODE_LEN = episode_len
    STEP_LEN = step_len
    MODEL_FREQ = 125

    samples_per_episode = FS * EPISODE_LEN
    step_size = FS * STEP_LEN

    total_samples = len(ppg)

    episodes = []

    for start in range(0, total_samples - samples_per_episode, step_size):
        ppg_seg = ppg[start:start + samples_per_episode]
        if np.any(np.isnan(ppg_seg)):
            ppg_seg[np.isnan(ppg_seg)] = 0

        if fs != MODEL_FREQ:
            resampled_len = MODEL_FREQ * episode_len
            resampled_start = start // fs * MODEL_FREQ
            resampled_ppg = resample(ppg_seg, resampled_len)
            episodes.append({
                'ppg': resampled_ppg,
                'start_idx': resampled_start
            })

        else:
            episodes.append({
                'ppg': ppg_seg,
                'start_idx': start
            })
    return episodes


def get_sys_dias_from_signal_seg(signal_seg, distance_btwn_peaks):
    """
    Calculate systolic and diastolic BP from peaks/troughs in the waveform
    :param signal_seg: segment of data to process
    :param distance_btwn_peaks: min distance between peaks (to avoid local maxes)
    :return: systolic & diastolic bp for the segment
    """
    peaks, _ = find_peaks(signal_seg, distance=distance_btwn_peaks)
    troughs, _ = find_peaks(-signal_seg, distance=distance_btwn_peaks)

    systolic = np.median(signal_seg[peaks])
    diastolic = np.median(signal_seg[troughs])

    return systolic, diastolic


def process_data(record_id, record_data):
    """
    Takes a patient record and down processes it to be fed into the model
    :param record_id: the ID that we are down processing
    :param record_data: the record data to be processed
    """
    # values in seconds
    episode_len = 10  # how long each episode is
    step_len = 5  # how large of a time step between episodes
    out_len = 30  # output length (determines how many episodes we include per block)

    assert (out_len % step_len == 0)
    bp_episodes = extract_ppg_episodes(record_data["ppg"], fs=record_data["ppg_freq"], episode_len=episode_len,
                                       step_len=step_len)

    systolic_preds = []
    diastolic_preds = []

    n_processed = 0
    # step through the blocks and predict bp, save to JSON
    for episode_obj in bp_episodes:
        ppg = episode_obj["ppg"]
        normalized_ppg = normalize_min_max(ppg)
        predicted_bp = predict_bp_from_ppg(normalized_ppg)
        systolic, diastolic = get_sys_dias_from_signal_seg(predicted_bp, 200)
        systolic_preds.append(systolic)
        diastolic_preds.append(diastolic)
        n_processed += 1
        if n_processed % 1000 == 0:
            print(n_processed)

    # take the average across some number of prediction blocks to get a 30s block
    sys_30s = []
    dias_30s = []
    num_episodes_per_block = out_len // step_len
    for start in range(0, len(systolic_preds), num_episodes_per_block):
        sys_30s.append(float(np.mean(systolic_preds[start:start + num_episodes_per_block])))
        dias_30s.append(float(np.mean(diastolic_preds[start:start + num_episodes_per_block])))

    # make sure there's no major alignment issues between the bp predictions and sleep stages.
    assert (abs(len(sys_30s) - len(record_data["ss"])) < 4)
    # trim both blocks to the same size
    shorter_block = min(len(sys_30s), len(record_data["ss"]))

    # write data out to file
    out_json = {
        "systolic": sys_30s[:shorter_block],
        "diastolic": dias_30s[:shorter_block],
        "sleep_stages": record_data["ss"][:shorter_block],
        "patient": record_data["patient_data"]
    }

    with open(f"mesa_processed/mesa-sleep-{record_id:04}.json", 'w') as json_file:
        json.dump(out_json, json_file, indent=4)
    print(f"finished processing patient {record_id:04}")


def main():
    """
        Runs through each patient record and process, output json for predicted BP
    """
    record_ids = get_record_ids()
    start = 1213
    start_idx = record_ids.index(start)
    for record_id in record_ids[start_idx:]:
        try:
            print(f"Processing record {record_id:04}")
            if len(sys.argv) > 1:
                process_data(record_id, get_data_for_patient(record_id, sys.argv[1]))
            else:
                process_data(record_id, get_data_for_patient(record_id))
        except:
            print(f"Problem handling patient ID {record_id}")

        os.system('cls' if os.name == 'nt' else 'clear')


if __name__ == '__main__':
    main()
