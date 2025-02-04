import mne
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def plot_ecg_from_edf(edf_file_path, channel_name=None, start_time=0, duration=None):
    """
    Read and plot ECG signal from an EDF file with specified time window.

    Parameters:
    -----------
    edf_file_path : str
        Path to the EDF file
    channel_name : str, optional
        Name of the ECG channel to plot. If None, will try to find an ECG channel
        automatically.
    start_time : float
        Start time in seconds
    duration : float, optional
        Duration to plot in seconds. If None, plots until the end of the signal
    """
    # Read the EDF file
    raw = mne.io.read_raw_edf(edf_file_path, preload=True)

    # If channel name is not provided, try to find an ECG channel
    if channel_name is None:
        ecg_channels = [ch for ch in raw.ch_names if 'ecg' in ch.lower()]
        if ecg_channels:
            channel_name = ecg_channels[0]
        else:
            raise ValueError("No ECG channel found. Please specify channel_name.")

    # Extract the ECG data
    ecg_data = raw[raw.ch_names.index(channel_name)][0][0]
    sampling_rate = raw.info['sfreq']

    # Calculate time indices for the window
    start_idx = int(start_time * sampling_rate)
    if duration is None:
        end_idx = len(ecg_data)
    else:
        end_idx = min(int((start_time + duration) * sampling_rate), len(ecg_data))

    # Create time array in seconds for the selected window
    time = np.arange(start_idx, end_idx) / sampling_rate

    # Select data for the specified window
    windowed_data = ecg_data[start_idx:end_idx]

    # Create the plot
    plt.figure(figsize=(15, 5))
    plt.plot(time, windowed_data, 'b-', linewidth=0.5)
    plt.title(f'ECG Signal from {channel_name} (Time Window: {start_time}s to {time[-1]:.1f}s)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude (μV)')
    plt.grid(True)

    # Add some information about the recording
    start_time_abs = raw.info['meas_date']
    if start_time_abs:
        start_time_str = start_time_abs.strftime('%Y-%m-%d %H:%M:%S')
        plt.text(0.02, 0.98, f'Recording start: {start_time_str}',
                 transform=plt.gca().transAxes, fontsize=8)

    # Show some basic signal statistics for the window
    mean_val = np.mean(windowed_data)
    std_val = np.std(windowed_data)
    plt.text(0.02, 0.94,
             f'Window statistics:\n'
             f'Mean: {mean_val:.2f} μV\nStd: {std_val:.2f} μV\n'
             f'Sampling rate: {sampling_rate} Hz',
             transform=plt.gca().transAxes, fontsize=8)

    plt.tight_layout()
    plt.show()

    return raw, windowed_data, sampling_rate


# Example usage
if __name__ == "__main__":
    # Replace with your EDF file path
    edf_file = "data/p000020/out/0005.edf"
    # edf_file = "data/test/sleep.edf"

    raw, ecg_data, fs = plot_ecg_from_edf(edf_file, duration=20)