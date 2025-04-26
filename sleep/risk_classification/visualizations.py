import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

import models
import importlib
importlib.reload(models)


def visualize_with_without_smote_for_rf(X,y,X_smote,y_smote):
    acc_list_bsm, avg_acc_bsm = models.compare_averages_summary_models(X,y)
    print(np.std(acc_list_bsm,axis=0))
    acc_list, avg_acc = models.compare_averages_summary_models(X_smote,y_smote)
    print(stats.ttest_ind(acc_list_bsm,acc_list))

    values = np.concatenate((np.array(acc_list_bsm)[:,0],np.array(acc_list_bsm)[:,1],np.array(acc_list_bsm)[:,2],np.array(acc_list)[:,0],np.array(acc_list)[:,1],np.array(acc_list)[:,2]))
    groups = (['Direct Accuracy']*50 + ['Positive Accuracy']*50 + ['Negative Accuracy']*50) * 2
    subgroups = ['Features From Sup. 80%']*150 + ['Features From Sup. 50% and Sup. 80%']*150

    data = {'Percent':values, 'Accuracy Metric':groups, 'Subgroup':subgroups}
    plot_df = pd.DataFrame(data)

    sns.boxplot(x='Accuracy Metric', y='Percent', data=plot_df, hue='Subgroup', palette='Set2',
                medianprops=dict(color="red", alpha=1))
    plt.title('RF Accuracies over 50 Runs')
    plt.ylim([0,100])
    plt.grid()
    plt.show()

def visualize_rf_for_three(X1,y1,X2,y2, X3,y3):
    acc_list_1, avg_acc_1 = models.compare_averages_summary_models(X1,y1)
    print(np.std(acc_list_1,axis=0))
    acc_list_2, avg_acc_2 = models.compare_averages_summary_models(X2, y2)
    print(np.std(acc_list_2, axis=0))
    acc_list_3, avg_acc_3 = models.compare_averages_summary_models(X3, y3)
    print(np.std(acc_list_3, axis=0))
    print("one and two")
    print(stats.ttest_ind(acc_list_1,acc_list_2))
    print("one and three")
    print(stats.ttest_ind(acc_list_1, acc_list_3))
    print("two and three")
    print(stats.ttest_ind(acc_list_2, acc_list_3))

    values = np.concatenate((np.array(acc_list_1)[:,0],np.array(acc_list_1)[:,1],np.array(acc_list_1)[:,2],
                             np.array(acc_list_2)[:,0],np.array(acc_list_2)[:,1],np.array(acc_list_2)[:,2],
                             np.array(acc_list_3)[:,0],np.array(acc_list_3)[:,1],np.array(acc_list_3)[:,2]))
    groups = (['Direct Accuracy']*50 + ['Positive Accuracy']*50 + ['Negative Accuracy']*50) * 3
    subgroups = ['Features From Sup. 20%']*150 + ['Features From Sup. 50%']*150 + ['Features From Sup. 80%']*150

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
