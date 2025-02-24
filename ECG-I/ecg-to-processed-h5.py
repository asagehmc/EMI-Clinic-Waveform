import math

import numpy as np
from scaled_fft import scaled_fft
import wfdb
from matplotlib import pyplot as plt
from scipy import signal

MIMIC = "mimic3wdb-matched/1.0/"
ecg_channel_name = "I"


def high_pass(freq=0.5):
    pass


def find_and_remove_noise(ecg_signal, fs, min_freq, min_noise_ratio, filter_bw, max_removed_freqs):
    n_removed_freqs = 0
    removed_freqs = []
    while n_removed_freqs < max_removed_freqs:
        # get a list of frequencies to filter out for noise
        to_remove_freqs = find_noise(ecg_signal, fs, min_freq, min_noise_ratio, num_snapshots=10)

        # take out all of the to-remove-frequencies that have already been filtered
        mask = ~np.isin(to_remove_freqs, removed_freqs)
        to_remove_freqs = to_remove_freqs[mask]

        # if no more frequencies to remove, break here!
        if len(to_remove_freqs) == 0:
            break

        # remove noise frequencies!
        ecg_signal = remove_noise(ecg_signal, to_remove_freqs, fs, filter_bw)
        removed_freqs += to_remove_freqs
        n_removed_freqs += 1
    return ecg_signal


# take 10 samples of 20s across the length of the data, find the median amplitude of waves across that time span
def find_noise(ecg_signal, fs, min_freq, min_noise_ratio, num_snapshots=10):
    freq_div = 0.5

    # how far apart we take the window readings
    window_spacing = math.floor(len(ecg_signal) / num_snapshots)
    median_amplitudes = []

    for i in range(num_snapshots):
        start = i * window_spacing
        window_len = fs * 20
        ecg_segment = ecg_signal[start:start + window_len]
        peaks = signal.find_peaks(ecg_segment, distance=2 * fs)[0]
        troughs = signal.find_peaks(-ecg_segment, distance=2 * fs)[0]
        median_amplitudes.append(np.median(peaks) + np.median(troughs))

    median_amplitude = np.median(median_amplitudes)
    if median_amplitude < 0.1:
        median_amplitude = 0.1

    waterfall = np.zeros((num_snapshots, fs+1))
    fft_window_length = window_spacing
    frequencies = None
    # capture windows for each frequency
    for i in range(num_snapshots):
        start = i * window_spacing
        # run scaled fft on the given window
        vals, fqs = scaled_fft(ecg_signal[start:start + fft_window_length], fs, freq_div, median_amplitude/2)
        waterfall[i] = vals
        frequencies = fqs

    # calculate the average of each frequency bin
    mean_fft = waterfall.mean(axis=0)
    assert (len(frequencies) == len(mean_fft))

    # line up the frequencies with their values to filter
    combined_fqs = np.column_stack((mean_fft, frequencies))

    combined_fqs = combined_fqs[combined_fqs[:, 0] > min_noise_ratio]
    combined_fqs = combined_fqs[combined_fqs[:, 1] > min_freq]

    just_frequencies = combined_fqs[:, 1]

    if just_frequencies.shape[0] > 0:
        # remove all frequencies equal to the final frequency
        just_frequencies = just_frequencies[just_frequencies != just_frequencies[-1]]

    # sort descending
    sorted_array = just_frequencies[just_frequencies.argsort()[::-1]]
    return sorted_array


def remove_noise(ecg_signal, freqs_to_filter, sample_frequency, filter_bandwidth):
    for filt_freq in freqs_to_filter:
        print(filt_freq)
        b, a = signal.iirnotch(filt_freq / (sample_frequency / 2),
                               filter_bandwidth / (sample_frequency / 2),
                               fs=sample_frequency)

        ecg_signal = signal.filtfilt(b, a, ecg_signal)
    return ecg_signal


def process_segment(patient, segment):

    # record = wfdb.rdrecord(segment, pn_dir=f"{MIMIC}{patient}")
    record = wfdb.rdrecord("local_data/3278512_0014")

    # read ecg data
    ecg_index = record.sig_name.index(ecg_channel_name)
    ecg_data = record.p_signal[:, ecg_index].astype(np.float64)

    ecg_data = np.nan_to_num(ecg_data)

    # high pass filter at frequency 0.5
    frequency = record.fs
    order = 4
    filter_frequency = 0.5
    sos = signal.butter(order, filter_frequency, btype="highpass", output="sos", fs=frequency)
    ecg_data = signal.sosfilt(sos, ecg_data)

    # remove 60hz noise
    filter_bandwidth = 1.5
    ecg_data = remove_noise(ecg_data, [60], frequency, filter_bandwidth)

    # remove other constant noise:
    min_freq = 10
    min_noise_ratio = 4
    max_removed_freqs = 10

    ecg_data = find_and_remove_noise(ecg_data, frequency, min_freq, min_noise_ratio, filter_bandwidth, max_removed_freqs)

    # resample to 256hz
    new_sample_rate = 256
    original_len = len(ecg_data)
    new_len = int(original_len * (new_sample_rate / record.fs))
    resampled_signal = signal.resample(ecg_data, new_len)

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
