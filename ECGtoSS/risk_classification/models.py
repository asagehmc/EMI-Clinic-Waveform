"""
Filename: models.py

Description: This file contains functions for training and testing risk classification models
"""

import numpy as np
import os
import random
import pandas as pd

from pandas.errors import InvalidIndexError
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV, RepeatedStratifiedKFold

# Import SMOTE, BorderlineSMOTE, and Pipeline from imblearn for oversampling based on class imbalance
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.pipeline import Pipeline

from t2f.extraction.extractor import feature_extraction
from t2f.utils.importance_old import feature_selection
from t2f.model.clustering import ClusterWrapper


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

def get_oversampler(method='smote', **kwargs):
    """
    Returns an oversampling object based on the provided method.

    :param method: str, name of oversampling technique ("smote" or "borderline_smote").
    :param kwargs: Additional keyword arguments forwarded to the oversampler.
    :return: An imblearn oversampler instance.
    :raises ValueError: If the specified method is unsupported.
    """
    method = method.lower()
    if method == 'smote':
        return SMOTE(random_state=42, **kwargs)
    elif method in ['borderline_smote', 'borderline-smote']:
        return BorderlineSMOTE(random_state=42, **kwargs)
    else:
        raise ValueError(f"Unsupported oversampling method: {method}")


def cross_validate_summary_model_with_smote(X, y, model_type, oversampling_method="smote", **smote_kwargs):
    """
    Runs 5-fold cross-validation using a pipeline that applies SMOTE on each training fold.

    :param X: Feature matrix.
    :param y: Labels.
    :param model_type: str, one of "svc", "rfc", or "knc".
    :param smote_sampling: float, SMOTE sampling strategy.
    :param **smote_kwargs: Additional keyword arguments that can be passed to SMOTE.
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

    oversampler = get_oversampler(oversampling_method, **smote_kwargs)
    pipe = Pipeline([
        ("oversampler", oversampler),
        ("clf", classifier)
    ])
    return cross_validate(pipe, X, y, cv=5, scoring="accuracy")


def score_rf_and_probability(X, y, test_size=0.2):
    """
    :param X: 2d np array, feature matrix
    :param y: 1d np array, labels
    :param test_size: float, percentage of data to test on
    :return: model: sklearn random forest classifier
             y_pred: predicted labels on the testing data
             y_prob: predicted probabilities on the testing data
             y_test: true labels on the testing data
             X_test: feature matrix of the testing data
    """
    model = RandomForestClassifier()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    return model, y_pred, y_prob, y_test, X_test

def optimize_rfc_with_smote(X, y, cv=5, scoring="accuracy", oversampling_method="smote"):
    """
    Performs hyperparameter tuning for the RandomForestClassifier using a pipeline
    that applies an oversampling method (e.g., SMOTE or Borderline-SMOTE) on each training fold.

    :param X: Feature matrix.
    :param y: Labels.
    :param cv: int, number of cross-validation folds (default is 5).
    :param scoring: str, metric used for evaluating hyperparameter combinations (default is "accuracy").
    :param oversampling_method: str, oversampling technique ("smote" or "borderline_smote").
    :return: Tuple containing the fitted GridSearchCV object and detailed cross-validation results.
    """
    param_grid = {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 3, 5],
    }
    oversampler = get_oversampler(oversampling_method)
    pipeline = Pipeline([
        ("oversampler", oversampler),
        ("clf", RandomForestClassifier(random_state=42))
    ])
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=2)
    grid_search.fit(X, y)
    print("Best hyperparameters found:", grid_search.best_params_)
    print("Best CV score:", grid_search.best_score_)
    results = grid_search.cv_results_
    return grid_search, results


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


def score_summary_model_with_smote(X, y, model_type, test_size=0.2, oversampling_method="smote"):
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
    oversampler = get_oversampler(oversampling_method)
    X_train_res, y_train_res = oversampler.fit_resample(X_train, y_train)
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)
    return accuracies(y_pred, y_test)


def score_tuned_summary_model(X, y, tuned_model, test_size=0.2):
    """
    Splits the data into training and test sets, fits the provided tuned model, and computes accuracy metrics.

    :param X: Feature matrix.
    :param y: Labels.
    :param tuned_model: Pre-tuned classifier (e.g., RandomForestClassifier with optimized parameters).
    :param test_size: float, fraction of data to be used as the test set (default is 0.2).
    :return: Tuple of accuracy metrics (direct accuracy, risk accuracy, no-risk accuracy, FPR, FNR).
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    tuned_model.fit(X_train, y_train)
    y_pred = tuned_model.predict(X_test)
    return accuracies(y_pred, y_test)

def score_summary_model_with_smote_extended(X, y, model_type, test_size=0.2, oversampling_method="smote"):
    """
    Splits the data into training and test sets, applies the specified oversampling method on the training set,
    trains the model, and computes accuracy metrics.

    :param X: Feature matrix.
    :param y: Labels.
    :param model_type: str, currently supports only "rfc".
    :param test_size: float, fraction of data to be used as the test set (default is 0.2).
    :param oversampling_method: str, oversampling technique ("smote" or "borderline_smote").
    :return: Tuple of accuracy metrics (direct accuracy, risk accuracy, no-risk accuracy, FPR, FNR).
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    oversampler = get_oversampler(oversampling_method)
    X_train_res, y_train_res = oversampler.fit_resample(X_train, y_train)

    if model_type == "rfc":
        model = RandomForestClassifier(random_state=42)
    else:
        raise ValueError("This evaluation supports only the RFC model type.")

    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)

    return accuracies(y_pred, y_test)

# Running the above extended evaluation several times and average results.
def compare_averaged_extended_evaluations(X, y, model_type="rfc", test_size=0.2, oversampling_method="smote",
                                          num_runs=10):
    """
    Runs multiple extended evaluations and averages the results over several runs.

    :param X: Feature matrix.
    :param y: Labels.
    :param model_type: str, currently supports only "rfc".
    :param test_size: float, fraction of data reserved for the test split (default is 0.2).
    :param oversampling_method: str, oversampling technique ("smote" or "borderline_smote").
    :param num_runs: int, number of evaluation runs to average.
    :return: Array of averaged accuracy metrics.
    """
    results = []
    for _ in range(num_runs):
        res = score_summary_model_with_smote_extended(X, y, model_type, test_size=test_size,
                                                      oversampling_method=oversampling_method)
        results.append(res)
    avg_results = np.mean(np.array(results), axis=0)
    print(f"\nAveraged extended evaluation for RFC with {oversampling_method} over {num_runs} runs: {avg_results}")
    return avg_results

# Clustering Model Training & Accuracy Computation
def calculate_features(X):
    """
    Transposes input data and performs time2feat feature extraction.

    :param X: 3D array shaped (patients, variables, timestamps).
    :return: Pandas DataFrame of extracted features.
    """
    # transpose from (patients, variables, timestamps) to (patients, timestamps, variables)
    X = np.transpose(X, (0, 2, 1))

    # Feature extraction
    df_feats = feature_extraction(X, batch_size=100, p=1)

    return df_feats

def train_t2f_model_from_calculated_features(X_feats, transform_type, model_type, y=None, training_sampling=0):
    """
    Applies feature selection and clustering on precomputed features in unsupervised or semi-supervised mode.

    :param X_feats: DataFrame of extracted features.
    :param transform_type: str, transformation type for clustering.
    :param model_type: str, clustering algorithm type.
    :param y: Optional array of true labels for semi-supervised mode.
    :param training_sampling: float, fraction of labeled samples to use (default is 0).
    :return: Tuple (predicted labels, clustering model, selected feature list).
    """
    if y is None:
        labels = {} # unsupervised mode
    else:
        i_label_sample = random.sample(range(len(y)), int(training_sampling*len(y)))
        labels = {i:y[i] for i in i_label_sample} # semi-supervised mode
        print(labels)

    # binary classification
    n_clusters = 2  # Number of clusters

    # Feature selection
    context = {'model_type': model_type, 'transform_type': transform_type}
    top_feats = feature_selection(X_feats, labels=labels, context=context)
    print("selected features")
    df_feats = X_feats[top_feats]

    # Clustering
    model = ClusterWrapper(n_clusters=n_clusters, model_type=model_type, transform_type=transform_type)
    y_pred = model.fit_predict(df_feats)
    print(y_pred.shape)

    return y_pred, model, top_feats

def train_t2f_model(X, transform_type, model_type, y=None, training_sampling=0):
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
        labels = {} # unsupervised mode
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
    # overall accuracy
    direct_accuracy = 100 * np.sum(y_pred == y_true) / len(y_true)
    # positive accuracy
    risk_accuracy = 100 * np.sum(
        [1 if pred == true else 0 for pred, true in zip(y_pred, y_true) if true == 1]) / np.sum(y_true)
    # negative accuracy
    no_risk_accuracy = 100 * np.sum([1 if pred == true else 0 for pred, true in zip(y_pred, y_true) if true == 0]) / (
                len(y_true) - np.sum(y_true))
    # false positive rate
    FPR = 100 * np.sum([1 if pred != true else 0 for pred, true in zip(y_pred, y_true) if pred == 1]) / (
                len(y_true) - np.sum(y_true))
    # false negative rate
    FNR = 100 * np.sum([1 if pred != true else 0 for pred, true in zip(y_pred, y_true) if pred == 0]) / np.sum(y_true)
    return direct_accuracy, risk_accuracy, no_risk_accuracy, FPR, FNR


# Comparison Functions for Clustering and Summary Models
def compare_unsupervised_clustering(X_feats, y):
    """
    Compares different clustering configurations in an unsupervised setting.

    :param X_feats: Calculated features from input waveform data.
    :param y: True labels.
    :return: Dictionary mapping each (model, transform) configuration to its accuracy metrics.
    """
    unsupervised_accuracies = {}
    for model_type in ['Hierarchical', 'KMeans', 'Spectral']:
        for transform_type in ['std', 'minmax', 'robust']:
            i = 0
            while i < 10:
                try:
                    y_pred, model = train_t2f_model_from_calculated_features(X_feats, transform_type, model_type)
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


def compare_averaged_unsupervised_clustering(X_feats, y, num_runs):
    """
    Runs multiple clustering evaluations and averages the accuracy metrics.

    :param X_feats: Calculated features from input waveform data.
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
                        y_pred, model, _ = train_t2f_model_from_calculated_features(X_feats, transform_type, model_type)
                        break
                    except Exception as e:
                        i += 1
                        if i >= 10:
                            y_pred = None
                if y_pred is not None:
                    acc_list.append(accuracies(y_pred, y))
            if acc_list:
                avg_acc = np.mean(np.array(acc_list), axis=0)
                avg_std = np.std(np.array(acc_list), axis=0)
                unsupervised_accuracies[(model_type, transform_type)] = np.array([avg_acc[:3], avg_std[:3]])
                print(f"{model_type}, {transform_type} averaged accuracies: {avg_acc}")
                print(f"{model_type}, {transform_type} std: {avg_std}")
    return unsupervised_accuracies

def compare_averaged_supervised_clustering(X_feats, y, training_sampling, num_runs):
    """
    Evaluates clustering in a semi-supervised setting (where part of the labels are known)
    across multiple runs and averages the results.

    :param X_feats: Calculated features from input waveform data.
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
                        y_pred, model, _ = train_t2f_model_from_calculated_features(X_feats, transform_type, model_type, y, training_sampling)
                        break
                    except Exception as e:
                        i += 1
                        if i >= 10:
                            y_pred = None
                if y_pred is not None:
                    acc_list.append(accuracies(y_pred, y))
            if acc_list:
                avg_acc = np.mean(np.array(acc_list), axis=0)
                avg_std = np.std(np.array(acc_list), axis=0)
                supervised_accuracies[(model_type, transform_type)] = np.array([avg_acc[:3], avg_std[:3]])
                print(f"{model_type}, {transform_type} averaged accuracies: {avg_acc}")
                print(f"{model_type}, {transform_type} std: {avg_std}")
    return supervised_accuracies

def compare_unsupervised_clustering_MESA(X_feats, n_clusters=2):
    """
    :param X_feats: pd df, calculated features from input waveform data using tsfresh
    :param n_clusters: int, number of clusters
    :return: np array, array of predicted labels
    """
    y_preds = []
    for model_type in ['Hierarchical', 'KMeans', 'Spectral']:
        for transform_type in ['std', 'minmax', 'robust']:
            print(model_type, " ", transform_type)
            # Feature selection
            context = {'model_type': model_type, 'transform_type': transform_type}
            top_feats = feature_selection(X_feats, labels={}, context=context)
            df_feats = X_feats[top_feats]

            # Clustering
            model = ClusterWrapper(n_clusters=n_clusters, model_type=model_type, transform_type=transform_type)
            y_pred = model.fit_predict(df_feats)
            y_preds.append(y_pred)

    y_preds = np.array(y_preds)

    return y_preds


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

    return

def accuracy_averages_for_rf(X, y):
    """
    Runs multiple evaluations of summary models and prints averaged accuracies.

    :param X: Feature matrix.
    :param y: Labels.
    """
    model_type = "rfc"
    acc_list = []
    for _ in range(50):
        scores = score_summary_model(X, y, model_type)
        print(scores)
        acc_list.append(scores)
    avg_acc = np.mean(np.array(acc_list), axis=0)
    avg_std = np.std(np.array(acc_list), axis=0)
    print(f"{model_type} averaged summary accuracies: {avg_acc}")
    print(f"{model_type} has standard devation of accuracies: {avg_std}")
    return acc_list

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
        avg_std = np.std(np.array(acc_list), axis=0)
        print(f"{model_type} averaged summary accuracies: {avg_acc}")
        print(f"{model_type} has standard devation of accuracies: {avg_std}")


def get_selected_features_and_scores_over_n_runs(n, X_feats, y, training_sampling):
    """
    Runs the training function n times to record both accuracy metrics and the frequency
    with which each feature is selected.

    :param n: Number of runs.
    :param X_feats: Calculated features from input waveform data.
    :param y: True labels.
    :param training_sampling: Fraction of labeled samples for semi-supervised mode.
    :return: Dictionary mapping each (model, transform) configuration to its accuracy metrics.
    """
    supervised_accuracies = {}
    all_models_dict = {}
    for model_type in ['Hierarchical','KMeans','Spectral']:
        for transform_type in ['std','minmax', 'robust']:
            selected_features_dict = {}
            accuracies_list = []
            for _ in range(n):
                i = 0
                while i < 10:
                    try:
                        y_pred, model, top_feats = train_t2f_model_from_calculated_features(X_feats, transform_type, model_type, y, training_sampling)
                        break
                    except Exception as e:
                        i += 1
                        if i >= 10:
                            top_feats = []
                accs = accuracies(y_pred, y)
                accuracies_list.append(accs)
                for feat in top_feats:
                    selected_features_dict[feat] = selected_features_dict.get(feat, 0) + 1
                    if accs[0] > 50 and accs[1] < 80 and accs[2] < 80:
                        all_models_dict[feat] = all_models_dict.get(feat, 0) + 1
            avg_acc = np.mean(np.array(accuracies_list), axis=0)
            avg_std = np.std(np.array(accuracies_list), axis=0)
            supervised_accuracies[(model_type, transform_type)] = np.array([avg_acc[:3], avg_std[:3]])
            print(f"{model_type}, {transform_type} averaged accuracies: {avg_acc}")
            print(f"{model_type}, {transform_type} std: {avg_std}")
    return supervised_accuracies, all_models_dict

# # Loading Data & Comparing Models
# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     from preprocessing import load_preprocessing_data, load_full_data, load_bp_only_data
#     from models import cross_validate_summary_model, cross_validate_summary_model_with_smote
#
#     # Load summary and time-series data
#     X_sum, y_sum, X_sum_dem, y_sum_dem, X_ts, y_ts = load_preprocessing_data()
#
#     # Incorporate the imbalance check for y_sum
#     unique, counts = np.unique(y_sum, return_counts=True)
#     total_samples = len(y_sum)
#     print("Class distribution in y_sum:")
#     for label, count in zip(unique, counts):
#         percentage = count / total_samples * 100
#         print(f"Class {label}: {count} samples ({percentage:.2f}%)")
#     if len(counts) == 2:
#         imbalance_ratio = max(counts) / min(counts)
#         print(f"Imbalance ratio (majority / minority): {imbalance_ratio:.2f}")
#
#     # Applying SMOTE to the summary data (X_sum, y_sum)
#     oversampler = get_oversampler('smote')
#     X_sum_res, y_sum_res = oversampler.fit_resample(X_sum, y_sum)
#
#     # Printing the class distribution and imbalance ratio after SMOTE is applied
#     unique_res, counts_res = np.unique(y_sum_res, return_counts=True)
#     total_samples_res = len(y_sum_res)
#     print("\nClass distribution in y_sum after SMOTE:")
#     for label, count in zip(unique_res, counts_res):
#         percentage = count / total_samples_res * 100
#         print(f"Class {label}: {count} samples ({percentage:.2f}%)")
#     if len(counts_res) == 2:
#         imbalance_ratio_res = max(counts_res) / min(counts_res)
#         print(f"Imbalance ratio (majority / minority) after SMOTE: {imbalance_ratio_res:.2f}")
#
#
#     # Varied additional oversampling to see impacts on RFC + SMOTE accuracies
#     extras = [15, 20, 25, 30]  # extra oversampling numbers
#
#     X_full, y_full, *rest = load_full_data()  # returns X, y, X_dem, y_dem, X_ts, y_ts
#
#     print("\n>>> RFC Baselines <<<")
#     # Just RFC, no oversampling
#     cv_plain = cross_validate_summary_model(X_full, y_full, model_type="rfc")
#     mean_plain = np.mean(cv_plain["test_score"]) * 100
#     std_plain = np.std(cv_plain["test_score"]) * 100
#     print(f"RFC (no oversampling)      → Mean CV acc: {mean_plain:5.2f}%  ± {std_plain:4.2f}%")
#
#     # RFC + SMOTE without additional oversampling
#     cv_sm = cross_validate_summary_model_with_smote(
#         X_full, y_full,
#         model_type="rfc",
#         oversampling_method="smote"
#     )
#     mean_sm = np.mean(cv_sm["test_score"]) * 100
#     std_sm = np.std(cv_sm["test_score"]) * 100
#     print(f"RFC + SMOTE (default)      → Mean CV acc: {mean_sm:5.2f}%  ± {std_sm:4.2f}%")
#
#     print("\n>>> RFC + SMOTE: extra synthetic samples per class <<<")
#     # get the original counts per class once
#     orig_counts = np.bincount(y_full.astype(int))
#     majority = orig_counts.max()
#
#     for e in extras:
#         # balance both classes to (majority + e)
#         target_n = majority + e
#         sampling_strategy = {0: target_n, 1: target_n}
#
#         cv = cross_validate_summary_model_with_smote(
#             X_full, y_full,
#             model_type="rfc",
#             oversampling_method="smote",
#             sampling_strategy=sampling_strategy
#         )
#
#         mean_acc = np.mean(cv["test_score"]) * 100
#         std_acc = np.std(cv["test_score"]) * 100
#         print(f"+{e:2d} samples →  Mean CV acc: {mean_acc:5.2f}%  ± {std_acc:4.2f}%")
#
#
#     # Comparing SMOTE vs. Borderline-SMOTE for RFC
#     oversampling_methods = ['smote', 'borderline_smote']
#     for method in oversampling_methods:
#         print("\n>>> Extended RFC Evaluation using {} over multiple runs <<<".format(method))
#         compare_averaged_extended_evaluations(X_sum, y_sum, model_type="rfc", test_size=0.2,
#                                               oversampling_method=method, num_runs=10)
#
#     print(">>> Summary Models Comparison (Non-Demographic vs. Demographic) <<<")
#     compare_summary_models(X_sum, y_sum, X_sum_dem, y_sum_dem)
#
#     print("\n>>> Averaged Summary Models Accuracies <<<")
#     compare_averages_summary_models(X_sum, y_sum)
#
#     print("\n>>> SMOTE-Based Supervised Model Evaluation <<<")
#     for model in ["svc", "rfc", "knc"]:
#         smote_acc = score_summary_model_with_smote(X_sum, y_sum, model, test_size=0.2, smote_sampling=1.0)
#         print(f"SMOTE {model} accuracies: {smote_acc}")
#
#     print("\n>>> Averaged Unsupervised Clustering Accuracies <<<")
#     unsup_accuracies = compare_averaged_unsupervised_clustering(X_ts, y_ts, num_runs=10)
#     print("Final Averaged Unsupervised Clustering Accuracies:")
#     print(unsup_accuracies)
#
#     print("\n>>> Averaged Supervised (Semi-supervised) Clustering Accuracies <<<")
#     sup_accuracies = compare_averaged_supervised_clustering(X_ts, y_ts, training_sampling=0.2, num_runs=10)
#     print("Final Averaged Supervised Clustering Accuracies:")
#     print(sup_accuracies)
#
#     # Comparing Accuracies for just using BP versus using BP + Sleep Stage Inputs
#
#     rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
#
#
#     def get_X_y(loader):
#         vals = loader()
#         return vals[0], vals[1]
#
#
#     models = {
#         "SVC": SVC(),
#         "RFC": RandomForestClassifier(),
#         "KNC": KNeighborsClassifier()
#     }
#
#     scenarios = [
#         ("BP + Sleep Stages", load_full_data),
#         ("BP Only", load_bp_only_data)
#     ]
#
#     # Boxplot comparing all three classifiers (SVC, RFC, KNC)
#     fig1, ax1 = plt.subplots(figsize=(8, 4))
#     box_data, labels1 = [], []
#
#     for scen_name, loader in scenarios:
#         X, y = get_X_y(loader)
#         for name, clf in models.items():
#             scores = cross_validate(clf, X, y, cv=rkf, scoring="accuracy", n_jobs=-1)["test_score"] * 100
#             mean, std = scores.mean(), scores.std()
#             print(f"{name} ({scen_name}):  Mean = {mean:.2f}%  ± {std:.2f}%")
#             box_data.append(scores)
#             labels1.append(f"{name}\n({scen_name})")
#
#     ax1.boxplot(box_data, labels=labels1, patch_artist=True, showfliers=False)
#     ax1.set_title("50×5‑Fold CV Accuracies: SVC / RFC / KNC")
#     ax1.set_ylabel("Accuracy (%)")
#     ax1.grid(axis="y", linestyle="--", alpha=0.7)
#     plt.xticks(rotation=30, ha="right")
#     plt.tight_layout()
#
#     # Boxplot comparing RFC + SMOTE
#     fig2, ax2 = plt.subplots(figsize=(6, 4))
#     box_data_sm, labels2 = [], []
#
#     for scen_name, loader in scenarios:
#         X, y = get_X_y(loader)
#         pipe = Pipeline([("smote", SMOTE(random_state=42)), ("rfc", RandomForestClassifier())])
#         scores = cross_validate(pipe, X, y, cv=rkf, scoring="accuracy", n_jobs=-1)["test_score"] * 100
#         mean, std = scores.mean(), scores.std()
#         print(f"RFC+SMOTE ({scen_name}):  Mean = {mean:.2f}%  ± {std:.2f}%")
#         box_data_sm.append(scores)
#         labels2.append(scen_name)
#
#     ax2.boxplot(box_data_sm, labels=labels2, patch_artist=True, showfliers=False)
#     ax2.set_title("50×5‑Fold CV: RFC + SMOTE")
#     ax2.set_ylabel("Accuracy (%)")
#     ax2.grid(axis="y", linestyle="--", alpha=0.7)
#     plt.tight_layout()
#
#     # Boxplot comparing RFC + Borderline‑SMOTE
#     fig3, ax3 = plt.subplots(figsize=(6, 4))
#     box_data_bl, labels3 = [], []
#
#     for scen_name, loader in scenarios:
#         X, y = get_X_y(loader)
#         pipe = Pipeline([("smote", BorderlineSMOTE(random_state=42)), ("rfc", RandomForestClassifier())])
#         scores = cross_validate(pipe, X, y, cv=rkf, scoring="accuracy", n_jobs=-1)["test_score"] * 100
#         mean, std = scores.mean(), scores.std()
#         print(f"RFC+Borderline‑SMOTE ({scen_name}):  Mean = {mean:.2f}%  ± {std:.2f}%")
#         box_data_bl.append(scores)
#         labels3.append(scen_name)
#
#     ax3.boxplot(box_data_bl, labels=labels3, patch_artist=True, showfliers=False)
#     ax3.set_title("50×5‑Fold CV: RFC + Borderline‑SMOTE")
#     ax3.set_ylabel("Accuracy (%)")
#     ax3.grid(axis="y", linestyle="--", alpha=0.7)
#     plt.tight_layout()
#
#     plt.show()