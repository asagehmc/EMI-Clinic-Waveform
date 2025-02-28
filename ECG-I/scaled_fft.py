import math

import numpy as np
import scipy.fft as fft
from scipy import signal
import matplotlib.pyplot as plt

def next_odd(n):
    return 2 * math.floor(math.ceil(n) / 2) + 1


def scaled_fft(ecg_signal, fs, freq_div, min_value=0.01):
    # how many frequencies we "bin" together in the moving mean of frequency magnitudes
    frequency_bins = max(next_odd(6 / freq_div + 1), 5)
    ecg_signal = ecg_signal[:round(fs/freq_div)]
    fft_len = len(ecg_signal)
    # take the magnitudes of the fft output
    y = fft.fft(ecg_signal, fft_len)
    y = np.abs(y)
    # trim the symmetry out of the fft result
    y = y[:math.floor(len(y)/2 + 1)]
    # preserve the signal magnitude after clipping symmetry (fs=0, symmetrical frequency don't get doubled)
    y[1:len(y) - 1] *= 2

    # calculate the list of frequencies for the fft
    frequencies = fs * np.array(range(0, round(fft_len/2)+1)) / fft_len

    fft_medians = signal.medfilt(y, kernel_size=frequency_bins)
    fft_medians[fft_medians < min_value] = min_value

    y /= fft_medians
    y[np.isinf(y)] = 1

    return y, frequencies
