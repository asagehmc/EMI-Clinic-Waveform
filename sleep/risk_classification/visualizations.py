import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import models
import importlib
importlib.reload(models)


def visualize_with_without_smote_for_rf(X,y,X_smote,y_smote):
    acc_list_bsm, avg_acc_bsm = models.compare_averages_summary_models(X,y)
    print(np.std(acc_list_bsm,axis=0))
    acc_list, avg_acc = models.compare_averages_summary_models(X_smote,y_smote)

    values = np.concatenate((np.array(acc_list_bsm)[:,0],np.array(acc_list_bsm)[:,1],np.array(acc_list_bsm)[:,2],np.array(acc_list)[:,0],np.array(acc_list)[:,1],np.array(acc_list)[:,2]))
    groups = (['Direct Accuracy']*50 + ['Positive Accuracy']*50 + ['Negative Accuracy']*50) * 2
    subgroups = ['Trained on Original Data']*150 + ['Trained on Data Oversampled with SMOTE']*150

    data = {'Percent':values, 'Accuracy Metric':groups, 'Subgroup':subgroups}
    plot_df = pd.DataFrame(data)

    sns.boxplot(x='Accuracy Metric', y='Percent', data=plot_df, hue='Subgroup', palette='Set2',
                medianprops=dict(color="red", alpha=1))
    plt.title('RF Accuracies over 50 Runs')
    plt.ylim([0,100])
    plt.grid()
    plt.show()


def fft_magnitude_phase_plot(N, dt, magnitudes, phases):
    fs = 1 / dt

    freqs = np.fft.fftfreq(N, d=dt)[:100]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(freqs, magnitudes)
    plt.title("FFT Magnitude Spectrum (First 100 Coefficients)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")

    plt.subplot(1, 2, 2)
    plt.plot(freqs, phases)
    plt.title("FFT Phase Spectrum (First 100 Coefficients)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase (radians)")

    plt.tight_layout()
    plt.show()

def fft_reconstruction_plot(original_signal, dt, magnitudes, phases, symmetric = True):
    N = len(original_signal)

    fft_partial = magnitudes * np.exp(1j * phases)

    fft_full = np.zeros(N, dtype=complex)

    fft_full[:100] = fft_partial
    if symmetric:
        fft_full[-(100 - 1):] = np.conj(fft_partial[1:][::-1])

    reconstructed_signal = np.fft.ifft(fft_full).real

    time = np.arange(N) * dt

    plt.figure(figsize=(10, 4))
    plt.plot(time, reconstructed_signal, label='Reconstructed Signal')
    plt.plot(time, original_signal, label='Original Signal', color='black', alpha=.5)
    plt.title("Reconstructed Signal from 100 FFT Coefficients")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
