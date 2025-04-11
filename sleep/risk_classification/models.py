import numpy as np
import os
import random
import pandas as pd

from pandas.errors import InvalidIndexError
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate, train_test_split

# Import SMOTE and Pipeline from imblearn for oversampling based on class imbalance
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# SKETCHY DIRECTORY SOLUTION
os.chdir("/Users/shreyabalaji/PycharmProjects/EIT-Clinic-Waveform/sleep/risk_classification/time2feat")
from time2feat.t2f.extraction.extractor import feature_extraction
from time2feat.t2f.utils.importance_old import feature_selection
from time2feat.t2f.model.clustering import ClusterWrapper

# Change back to the risk classification directory
os.chdir("/Users/shreyabalaji/PycharmProjects/EIT-Clinic-Waveform/sleep/risk_classification")


# Standard Supervised Models: Cross-Validation & Test Set Scoring
def cross_validate_summary_model(X, y, model_type):
    """
    Performs 5-fold cross-validation with the specified summary classifier.

    :param X: Feature matrix.
    :param y: Labels.
    :param model_type: str, one of "svc", "rfc", or "knc".
    :return: Cross-validation scores.
    """
    if model_type == "svc":
        model = SVC()
    elif model_type == "rfc":
        model = RandomForestClassifier()
    elif model_type == "knc":
        model = KNeighborsClassifier()
    else:
        raise ValueError("Model type not supported")
    return cross_validate(model, X, y, cv=5, scoring="accuracy")


def cross_validate_summary_model_with_smote(X, y, model_type, smote_sampling=1.0):
    """
    Runs 5-fold cross-validation using a pipeline that applies SMOTE on each training fold.

    :param X: Feature matrix.
    :param y: Labels.
    :param model_type: str, one of "svc", "rfc", or "knc".
    :param smote_sampling: float, SMOTE sampling strategy.
    :return: Cross-validation scores.
    """
    if model_type == "svc":
        classifier = SVC()
    elif model_type == "rfc":
        classifier = RandomForestClassifier()
    elif model_type == "knc":
        classifier = KNeighborsClassifier()
    else:
        raise ValueError("Model type not supported")

    pipe = Pipeline([
        ("smote", SMOTE(sampling_strategy=smote_sampling, random_state=42)),
        ("clf", classifier)
    ])
    return cross_validate(pipe, X, y, cv=5, scoring="accuracy")


def score_summary_model(X, y, model_type, test_size=0.2):
    """
    Splits the data into training and test sets, trains the chosen model, and computes accuracy metrics.

    :param X: Feature matrix.
    :param y: Labels.
    :param model_type: str, one of "svc", "rfc", or "knc".
    :param test_size: Fraction for the test data split.
    :return: Tuple of accuracy metrics.
    """
    if model_type == "svc":
        model = SVC()
    elif model_type == "rfc":
        model = RandomForestClassifier()
    elif model_type == "knc":
        model = KNeighborsClassifier()
    else:
        raise ValueError("Model type not supported")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracies(y_pred, y_test)


def score_summary_model_with_smote(X, y, model_type, test_size=0.2, smote_sampling=1.0):
    """
    Splits the data, applies SMOTE on the training set, trains the classifier, and computes accuracy metrics.

    :param X: Feature matrix.
    :param y: Labels.
    :param model_type: str, one of "svc", "rfc", or "knc".
    :param test_size: Fraction for the test data split.
    :param smote_sampling: SMOTE sampling strategy.
    :return: Tuple of accuracy metrics.
    """
    if model_type == "svc":
        model = SVC()
    elif model_type == "rfc":
        model = RandomForestClassifier()
    elif model_type == "knc":
        model = KNeighborsClassifier()
    else:
        raise ValueError("Model type not supported")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    sm = SMOTE(sampling_strategy=smote_sampling, random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)
    return accuracies(y_pred, y_test)


# Clustering Model Training & Accuracy Computation
def train_t2f_model(X, transform_type, model_type, y=None, training_sampling=0, batch_size=100):
    """
    Applies time2feat feature extraction, feature selection, and then clustering.

    :param X: The input waveform data.
    :param transform_type: Transformation type (e.g. 'std', 'minmax', or 'robust').
    :param model_type: Clustering model type (e.g. 'Hierarchical', 'KMeans', 'Spectral').
    :param y: (Optional) Labels for supervised or semi-supervised mode.
    :param training_sampling: Fraction of training samples (for semi-supervised mode).
    :param batch_size: Batch size to be used in feature extraction.
    :return: Predicted cluster labels, trained clustering model, and selected features.
    """
    # Set labels for unsupervised (empty dict) or semi-supervised mode.
    if y is None:
        labels = {}
    else:
        i_label_sample = random.sample(range(len(y)), int(training_sampling * len(y)))
        labels = {i: y[i] for i in i_label_sample}
        print("Semi-supervised labels:", labels)

    n_clusters = 2  # binary risk classification
    # Transpose from (patients, variables, timestamps) to (patients, timestamps, variables)
    X = np.transpose(X, (0, 2, 1))

    # Feature extraction with adjustable batch size
    df_feats = feature_extraction(X, batch_size=batch_size, p=1)
    print("Extracted features")

    # Feature selection (context can be used to pass additional settings)
    context = {'model_type': model_type, 'transform_type': transform_type}
    top_feats = feature_selection(df_feats, labels=labels, context=context)
    print("Selected features")
    df_feats = df_feats[top_feats]

    # Clustering
    model = ClusterWrapper(n_clusters=n_clusters, model_type=model_type, transform_type=transform_type)
    y_pred = model.fit_predict(df_feats)
    print("Clustering output shape:", y_pred.shape)

    return y_pred, model, top_feats


def accuracies(y_pred, y_true):
    """
    Computes various accuracy metrics between predicted and true labels.

    :param y_pred: Predicted cluster/model outputs.
    :param y_true: True labels.
    :return: A tuple (direct_accuracy, risk_accuracy, no_risk_accuracy, FPR, FNR)
    """
    direct_accuracy = 100 * np.sum(y_pred == y_true) / len(y_true)
    risk_accuracy = 100 * np.sum(
        [1 if pred == true else 0 for pred, true in zip(y_pred, y_true) if true == 1]) / np.sum(y_true)
    no_risk_accuracy = 100 * np.sum([1 if pred == true else 0 for pred, true in zip(y_pred, y_true) if true == 0]) / (
                len(y_true) - np.sum(y_true))
    FPR = 100 * np.sum([1 if pred != true else 0 for pred, true in zip(y_pred, y_true) if pred == 1]) / (
                len(y_true) - np.sum(y_true))
    FNR = 100 * np.sum([1 if pred != true else 0 for pred, true in zip(y_pred, y_true) if pred == 0]) / np.sum(y_true)
    return direct_accuracy, risk_accuracy, no_risk_accuracy, FPR, FNR


# Comparison Functions for Clustering and Summary Models
def compare_unsupervised_clustering(X, y):
    """
    Compares different clustering configurations in an unsupervised setting.

    :param X: Input waveform data.
    :param y: True labels.
    :return: Dictionary mapping each (model, transform) configuration to its accuracy metrics.
    """
    unsupervised_accuracies = {}
    for model_type in ['Hierarchical', 'KMeans', 'Spectral']:
        for transform_type in ['std', 'minmax', 'robust']:
            i = 0
            while i < 10:
                try:
                    y_pred, model, _ = train_t2f_model(X, transform_type, model_type)
                    break
                except Exception as e:
                    i += 1
                    if i >= 10:
                        print(f"Configuration ({model_type}, {transform_type}) failed after 10 tries")
                        y_pred = None
            if y_pred is not None:
                unsupervised_accuracies[(model_type, transform_type)] = accuracies(y_pred, y)
                print(
                    f"{model_type}, {transform_type} accuracies: {unsupervised_accuracies[(model_type, transform_type)]}")
    return unsupervised_accuracies


def compare_averaged_unsupervised_clustering(X, y, num_runs):
    """
    Runs multiple clustering evaluations and averages the accuracy metrics.

    :param X: Input waveform data.
    :param y: True labels.
    :param num_runs: Number of evaluation runs.
    :return: Dictionary mapping each (model, transform) configuration to averaged accuracy metrics.
    """
    unsupervised_accuracies = {}
    for model_type in ['Hierarchical', 'KMeans', 'Spectral']:
        for transform_type in ['std', 'minmax', 'robust']:
            acc_list = []
            for _ in range(num_runs):
                i = 0
                while i < 10:
                    try:
                        y_pred, model, _ = train_t2f_model(X, transform_type, model_type)
                        break
                    except Exception as e:
                        i += 1
                        if i >= 10:
                            y_pred = None
                if y_pred is not None:
                    acc_list.append(accuracies(y_pred, y))
            if acc_list:
                avg_acc = np.mean(np.array(acc_list), axis=0)
                unsupervised_accuracies[(model_type, transform_type)] = avg_acc
                print(f"{model_type}, {transform_type} averaged accuracies: {avg_acc}")
    return unsupervised_accuracies


def compare_averaged_supervised_clustering(X, y, training_sampling, num_runs):
    """
    Evaluates clustering in a semi-supervised setting (where part of the labels are known)
    across multiple runs and averages the results.

    :param X: Input waveform data.
    :param y: True labels.
    :param training_sampling: Fraction of samples used as labeled.
    :param num_runs: Number of evaluation runs.
    :return: Dictionary mapping each (model, transform) configuration to averaged accuracy metrics.
    """
    supervised_accuracies = {}
    for model_type in ['Hierarchical', 'KMeans', 'Spectral']:
        for transform_type in ['std', 'minmax', 'robust']:
            acc_list = []
            for _ in range(num_runs):
                i = 0
                while i < 10:
                    try:
                        y_pred, model, _ = train_t2f_model(X, transform_type, model_type, y, training_sampling)
                        break
                    except Exception as e:
                        i += 1
                        if i >= 10:
                            y_pred = None
                if y_pred is not None:
                    acc_list.append(accuracies(y_pred, y))
            if acc_list:
                avg_acc = np.mean(np.array(acc_list), axis=0)
                supervised_accuracies[(model_type, transform_type)] = avg_acc
                print(f"{model_type}, {transform_type} averaged supervised accuracies: {avg_acc}")
    return supervised_accuracies


def compare_summary_models(X1, y1, X2, y2):
    """
    Compares summary models (SVC, RFC, KNC) on two different datasets.

    :param X1: Feature matrix of first dataset.
    :param y1: Labels for first dataset.
    :param X2: Feature matrix of second dataset.
    :param y2: Labels for second dataset.
    """
    for (X, y) in [(X1, y1), (X2, y2)]:
        for model_type in ["svc", "rfc", "knc"]:
            cv_results = cross_validate_summary_model(X, y, model_type)
            print(
                f"{model_type} CV accuracy range: {min(cv_results['test_score']):.2f} - {max(cv_results['test_score']):.2f}")


def compare_averages_summary_models(X, y):
    """
    Runs multiple evaluations of summary models and prints averaged accuracies.

    :param X: Feature matrix.
    :param y: Labels.
    """
    for model_type in ["svc", "rfc", "knc"]:
        acc_list = []
        for _ in range(50):
            scores = score_summary_model(X, y, model_type)
            print(scores)
            acc_list.append(scores)
        avg_acc = np.mean(np.array(acc_list), axis=0)
        print(f"{model_type} averaged summary accuracies: {avg_acc}")


def get_selected_features_and_scores_over_n_runs(n, X, y, training_sampling):
    """
    Runs the training function n times to record both accuracy metrics and the frequency
    with which each feature is selected.

    :param n: Number of runs.
    :param X: Input waveform data.
    :param y: True labels.
    :param training_sampling: Fraction of labeled samples for semi-supervised mode.
    :return: Dictionary mapping each (model, transform) configuration to its accuracy metrics.
    """
    supervised_accuracies = {}
    for model_type in ['Hierarchical', 'KMeans', 'Spectral']:
        for transform_type in ['std', 'minmax', 'robust']:
            selected_features_dict = {}
            accuracies_list = []
            for _ in range(n):
                i = 0
                while i < 10:
                    try:
                        y_pred, model, top_feats = train_t2f_model(X, transform_type, model_type, y, training_sampling)
                        break
                    except Exception as e:
                        i += 1
                        if i >= 10:
                            top_feats = []
                accuracies_list.append(accuracies(y_pred, y))
                for feat in top_feats:
                    selected_features_dict[feat] = selected_features_dict.get(feat, 0) + 1
            averaged_accuracies = np.mean(np.array(accuracies_list), axis=0)
            supervised_accuracies[(model_type, transform_type)] = averaged_accuracies
            print(f"{model_type}, {transform_type} averaged accuracies: {averaged_accuracies}")
            print("Feature selection counts:", selected_features_dict)
    return supervised_accuracies


# Loading Data & Comparing Models
if __name__ == "__main__":
    from preprocessing import load_preprocessing_data

    # Load summary and time-series data
    X_sum, y_sum, X_sum_dem, y_sum_dem, X_ts, y_ts = load_preprocessing_data()

    # Incorporate the imbalance check for y_sum
    unique, counts = np.unique(y_sum, return_counts=True)
    total_samples = len(y_sum)
    print("Class distribution in y_sum:")
    for label, count in zip(unique, counts):
        percentage = count / total_samples * 100
        print(f"Class {label}: {count} samples ({percentage:.2f}%)")
    if len(counts) == 2:
        imbalance_ratio = max(counts) / min(counts)
        print(f"Imbalance ratio (majority / minority): {imbalance_ratio:.2f}")

    print(">>> Summary Models Comparison (Non-Demographic vs. Demographic) <<<")
    compare_summary_models(X_sum, y_sum, X_sum_dem, y_sum_dem)

    print("\n>>> Averaged Summary Models Accuracies <<<")
    compare_averages_summary_models(X_sum, y_sum)

    print("\n>>> SMOTE-Based Supervised Model Evaluation <<<")
    for model in ["svc", "rfc", "knc"]:
        smote_acc = score_summary_model_with_smote(X_sum, y_sum, model, test_size=0.2, smote_sampling=1.0)
        print(f"SMOTE {model} accuracies: {smote_acc}")

    print("\n>>> Averaged Unsupervised Clustering Accuracies <<<")
    unsup_accuracies = compare_averaged_unsupervised_clustering(X_ts, y_ts, num_runs=10)
    print("Final Averaged Unsupervised Clustering Accuracies:")
    print(unsup_accuracies)

    print("\n>>> Averaged Supervised (Semi-supervised) Clustering Accuracies <<<")
    sup_accuracies = compare_averaged_supervised_clustering(X_ts, y_ts, training_sampling=0.2, num_runs=10)
    print("Final Averaged Supervised Clustering Accuracies:")
    print(sup_accuracies)