import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
from scipy.cluster.hierarchy import linkage, dendrogram

import models
import importlib
importlib.reload(models)


def visualize_with_without_smote_for_rf(X1,y1,X2,y2,subgroup_titles):
    """
    boxplots of accuracies for two different subgroups
    :param X1: 2d np array, feature matrix
    :param y1: 1d np array, true labels
    :param X2: 2d np array, feature matrix
    :param y2: 1d np array, true labels
    :param subgroup_titles: list of str of length 2, subgroup titles
    return: None
    """
    acc_list_1 = models.accuracy_averages_for_rf(X1,y1)
    acc_list_2 = models.accuracy_averages_for_rf(X2,y2)

    # independent t test for the pair
    print(stats.ttest_ind(acc_list_1,acc_list_2))

    values = np.concatenate((np.array(acc_list_1)[:,0],np.array(acc_list_1)[:,1],np.array(acc_list_1)[:,2],
                             np.array(acc_list_2)[:,0],np.array(acc_list_2)[:,1],np.array(acc_list_2)[:,2]))
    groups = (['Direct Accuracy']*50 + ['Positive Accuracy']*50 + ['Negative Accuracy']*50) * 2
    subgroups = [subgroup_titles[0]]*150 + [subgroup_titles[1]]*150

    data = {'Percent':values, 'Accuracy Metric':groups, 'Subgroup':subgroups}
    plot_df = pd.DataFrame(data)

    sns.boxplot(x='Accuracy Metric', y='Percent', data=plot_df, hue='Subgroup', palette='Set2',
                medianprops=dict(color="red", alpha=1))
    plt.title('RF Accuracies over 50 Runs')
    plt.ylim([0,100])
    plt.grid()
    plt.show()

def visualize_rf_for_three(X1,y1,X2,y2, X3,y3, subgroup_titles):
    """
    boxplots of accuracies for three different subgroups
    :param X1: 2d np array, feature matrix
    :param y1: 1d np array, true labels
    :param X2: 2d np array, feature matrix
    :param y2: 1d np array, true labels
    :param X3: 2d np array, feature matrix
    :param y3: 1d np array, true labels
    :param subgroup_titles: list of str of length 3, subgroup titles
    return: None
    """
    acc_list_1 = models.accuracy_averages_for_rf(X1,y1)
    acc_list_2 = models.accuracy_averages_for_rf(X2, y2)
    acc_list_3 = models.accuracy_averages_for_rf(X3, y3)

    # independent t tests for all pairs
    print("one and two")
    print(stats.ttest_ind(acc_list_1,acc_list_2))
    print("one and three")
    print(stats.ttest_ind(acc_list_1, acc_list_3))
    print("two and three")
    print(stats.ttest_ind(acc_list_2, acc_list_3))

    # flatten accuracies into list
    values = np.concatenate((np.array(acc_list_1)[:,0],np.array(acc_list_1)[:,1],np.array(acc_list_1)[:,2],
                             np.array(acc_list_2)[:,0],np.array(acc_list_2)[:,1],np.array(acc_list_2)[:,2],
                             np.array(acc_list_3)[:,0],np.array(acc_list_3)[:,1],np.array(acc_list_3)[:,2]))
    # get accuracy lables
    groups = (['Direct Accuracy']*50 + ['Positive Accuracy']*50 + ['Negative Accuracy']*50) * 3
    # get subgroup labels according to inputs
    subgroups = [subgroup_titles[0]]*150 + [subgroup_titles[1]]*150 + [subgroup_titles[2]]*150

    # make pd dataframe to plot
    data = {'Percent':values, 'Accuracy Metric':groups, 'Subgroup':subgroups}
    plot_df = pd.DataFrame(data)

    # plot the boxplot
    sns.boxplot(x='Accuracy Metric', y='Percent', data=plot_df, hue='Subgroup', palette='Set2',
                medianprops=dict(color="red", alpha=1))
    plt.title('RF Accuracies over 50 Runs')
    plt.ylim([0,100])
    plt.grid()
    plt.show()


def fft_magnitude_phase_plot(N, dt, magnitudes, phases):
    """
    plots FFT magnitude phase plot
    :param N: int, number of time stamps
    :param dt: int, time in seconds between samples
    :param magnitudes: list, magnitude coefficients of FFT
    :param phases: list, phase coefficients of FFT
    :return: None
    """
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
    """
    plots FFT reconstruction plot
    :param original_signal: 1d np array, original time series on which coefficients were computed
    :param dt: int, time in seconds between samples
    :param magnitudes: list, magnitude coefficients of FFT
    :param phases: list, phase coefficients of FFT
    :param symmetric: bool, whether to use symmetric FFT
    :return: 
    """
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


def reduce_dimensions(X, n_components=2):
    """
    :param X: 2d np array, feature matrix
    :param n_components: int, number of dimensions to reduce to
    :return: np array, feature matrix with reduced dimensions
    """
    # PCA first for efficiency
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X)

    # Then apply non-linear reduction
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    X_reduced = reducer.fit_transform(X_pca)
    
    return X_reduced

def clustering_visualization(X_reduced, clustering_labels, n_clusters, model_type, transform_type):
    """
    plots clusters
    :param X_reduced: np array, feature matrix with reduced dimensions
    :param clustering_labels: 1d np array, predicted cluster labels
    :param n_clusters: int, number of clusters
    :param model_type: str, type of model to use, one of "Hierarchical" or "KMeans" or "Spectral"
    :return:
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clustering_labels, cmap='Spectral', s=n_clusters)
    plt.colorbar(boundaries=np.arange(n_clusters+1) - 0.5).set_ticks(np.arange(n_clusters))
    plt.title(model_type + " Clustering (UMAP projection) with " + transform_type + " Normalization")
    plt.show()
    
def dendrogram_visualization(X, truncate_p=None):
    """
    plots dendrogram for Hierarchical Clustering
    :param X: np array, feature matrix
    :param truncate_p: int (optional), number of samples to truncate to
    :return: 
    """
    if len(X) <= 1000:
        Z = linkage(X, 'ward')
        plt.figure(figsize=(12, 6))
        if truncate_p is not None:
            dendrogram(Z, truncate_mode='lastp', p=truncate_p)
        else:
            dendrogram(Z)
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.show()
    else:
        print("too many samples")