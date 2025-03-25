import numpy as np
import os
import random

# SKETCHY DIRECTORY SOLUTION
os.chdir("/Users/lydiastone/PycharmProjects/EIT-Clinic-Waveform/sleep/risk_classification/time2feat")
from t2f.extraction.extractor import feature_extraction
from t2f.utils.importance_old import feature_selection
from t2f.model.clustering import ClusterWrapper
os.chdir("/Users/lydiastone/PycharmProjects/EIT-Clinic-Waveform/sleep/risk_classification")

from preprocessing import X_sum, y_sum, X_sum_dem, y_sum_dem, X_ts, y_ts

def train_t2f_model(X, transform_type, model_type, y=None, training_sampling=0):
    """
    :param X: features,
    :param transform_type: str
    :param model_type:
    :return:
    """
    if y is None:
        labels = {} # unsupervised mode
    else:
        i_label_sample = random.sample(range(len(y)), int(training_sampling*len(y)))
        labels = {i:y[i] for i in i_label_sample} # semi-supervised mode

    # binary classification
    n_clusters = 2  # Number of clusters

    # transpose from (patients, variables, timestamps) to (patients, timestamps, variables)
    X = np.transpose(X, (0,2,1))

    # Feature extraction
    df_feats = feature_extraction(X, batch_size=100, p=1)

    # Feature selection
    context = {'model_type': model_type, 'transform_type': transform_type}
    top_feats = feature_selection(df_feats, labels=labels, context=context)
    df_feats = df_feats[top_feats]

    # Clustering
    model = ClusterWrapper(n_clusters=n_clusters, model_type=model_type, transform_type=transform_type)
    y_pred = model.fit_predict(df_feats)
    print(y_pred.shape)

    return y_pred, model

def accuracies(y_pred, y_true):
    """
    gets accuracy metrics
    :param y_pred:
    :param y_true:
    :return:
    """
    direct_accuracy = 100 * np.sum(y_pred == y_true) / len(y_true)
    risk_accuracy = 100 * np.sum([1 if pred==true else 0 for pred,true in zip(y_pred, y_true) if true==1]) / np.sum(y_true)
    no_risk_accuracy = 100 * np.sum([1 if pred==true else 0 for pred,true in zip(y_pred, y_true) if true==0]) / (len(y_true) - np.sum(y_true))

    return direct_accuracy, risk_accuracy, no_risk_accuracy