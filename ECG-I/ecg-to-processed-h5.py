import math

import numpy as np
import scipy
import wfdb
from matplotlib import pyplot as plt
from scipy import signal

MIMIC = "mimic3wdb-matched/1.0/"
ecg_channel_name = "I"

def high_pass(freq=0.5):
    pass



def find_and_remove_noise(data):

    return data

# take 10 samples of 20s across the length of the data, find the median amplitude of waves across that time span
def find_noise(data, fs, num_snapshots=10):

    # how far apart we take the window readings
    window_spacing = math.floor(len(data)/num_snapshots)
    median_amplitudes = []

    for i in range(num_snapshots):
        start = i*window_spacing
        window_len = fs * 20
        ecg_segment = data[start:start+window_len]
        peaks = signal.findpeaks(ecg_segment, distance=2 * fs)
        troughs = signal.findpeaks(-ecg_segment, distance = 2 * fs)
        median_amplitudes.append(np.median(peaks) + np.median(troughs))

    median_amplitude = np.median(median_amplitudes)
    if median_amplitude < 0.1:
        median_amplitude = 0.1

    fft_window_length = window_spacing
    for i in range(num_snapshots):
        start = i * window_spacing





def remove_noise(data):

    return data


def process_segment(patient, segment):
    # record = wfdb.rdrecord(segment, pn_dir=f"{MIMIC}{patient}")
    record = wfdb.rdrecord("local_data/3278512_0014")

    # read ecg data
    ecg_index = record.sig_name.index(ecg_channel_name)
    ecg_data = record.p_signal[:, ecg_index].astype(np.float64)

    non_nan_data = np.nan_to_num(ecg_data)

    # high pass filter at frequency 0.5
    frequency = record.fs
    order = 4
    filter_frequency = 0.5
    sos = signal.butter(order, filter_frequency, btype="highpass", output="sos", fs=frequency)

    filtered = signal.sosfilt(sos, non_nan_data)

    # TODO: filter for "line noise"

    # resample to 256hz
    new_sample_rate = 256
    original_len = len(filtered)
    new_len = int(original_len * (new_sample_rate / record.fs))
    resampled_signal = signal.resample(filtered, new_len)

    # scale data:
    #       shift to median 0
    median = np.median(resampled_signal)
    shifted_signal = resampled_signal - median
    #       rescale so 90% of peaks are within +-0.5 TODO: actually figure out how to do this
    ecg_peak_scalar = 0.05
    high_ecg_peak = np.percentile(shifted_signal, 100 - ecg_peak_scalar)
    low_ecg_peak = np.percentile(shifted_signal, ecg_peak_scalar)

    resize_amt = 1 / (high_ecg_peak - low_ecg_peak)
    resized_signal = shifted_signal * resize_amt

    clamped_signal = np.clip(resized_signal, -1, 1)

    # chunk data into 30s intervals
    chunk_size = 256 * 30  # 30 second windows
    chunked_signal = clamped_signal[:new_len - (new_len % chunk_size)]
    chunked_signal = chunked_signal.reshape(-1, chunk_size)

    plt.figure(figsize=(10, 6))
    plt.plot(chunked_signal[34][:1600])  # You can also use plt.hist(chunk) for a histogram plot
    plt.show()


if __name__ == "__main__":
    segment = "3278512_0014"
    patient = "p01/p018753/"
    process_segment(patient, segment)
