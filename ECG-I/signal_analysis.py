import numpy as np
from matplotlib import pyplot as plt
from scipy import signal


def segment_quality(signal, percentile=98):
    abs_signal = np.abs(signal)
    p_50 = np.mean(abs_signal)
    p_98 = np.percentile(abs_signal, percentile)
    quality = p_98 / p_50

    return quality


def get_scale_for_peaks(signal_seg, fs, segments=100, segment_length=30, pct=95):
    assert len(signal_seg) > segments * fs * segment_length

    spacing = len(signal_seg) // segments
    quality_threshold = 6.5  # heuristic threshold for usable signal
    all_peaks = np.array([])
    all_troughs = np.array([])
    # calculate a metric for "noise level"
    for i in range(segments - 1):
        window = signal_seg[i * spacing: i * spacing + segment_length * fs]
        quality = segment_quality(window)
        if quality > quality_threshold:
            peaks = window[signal.find_peaks(window, distance=fs/2)[0]]
            troughs = window[signal.find_peaks(-window, distance=fs/2)[0]]
            all_peaks = np.append(all_peaks, peaks)
            all_troughs = np.append(all_troughs, troughs)
    peaks_percentile_val = np.percentile(all_peaks, pct)
    troughs_percentile_val = abs(np.percentile(all_troughs, pct))
    print(peaks_percentile_val)
    print(troughs_percentile_val)
    return 0.5 / (peaks_percentile_val if peaks_percentile_val > troughs_percentile_val else troughs_percentile_val)
