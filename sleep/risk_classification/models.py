import numpy as np
import os
import random

from pandas.errors import InvalidIndexError
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split


# SKETCHY DIRECTORY SOLUTION
os.chdir("/Users/lydiastone/PycharmProjects/EIT-Clinic-Waveform/sleep/risk_classification/time2feat")
from time2feat.t2f.extraction.extractor import feature_extraction
from time2feat.t2f.utils.importance_old import feature_selection
from time2feat.t2f.model.clustering import ClusterWrapper
os.chdir("/Users/lydiastone/PycharmProjects/EIT-Clinic-Waveform/sleep/risk_classification")

# from preprocessing import X_sum, y_sum, X_sum_dem, y_sum_dem, X_ts, y_ts

def cross_validate_summary_model(X, y, model_type):
    """
    :param X: features
    :param y: labels
    :param model_type: str: "svc", "rfc", or "knc"
    :return: trained model and cross validation scores
    """
    if model_type == "svc":
        model = SVC()
    elif model_type == "rfc":
        model = RandomForestClassifier()
    elif model_type == "knc":
        model = KNeighborsClassifier()
    else:
        return "model type not supported"

    return cross_validate(model, X, y, cv=5, scoring="accuracy")

def score_summary_model(X, y, model_type, test_size=0.2):
    if model_type == "svc":
        model = SVC()
    elif model_type == "rfc":
        model = RandomForestClassifier()
    elif model_type == "knc":
        model = KNeighborsClassifier()
    else:
        return "model type not supported"

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    model.fit(X_train, y_train)

    return model.score(X_test, y_test)

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
        print(labels)

    # binary classification
    n_clusters = 2  # Number of clusters

    # transpose from (patients, variables, timestamps) to (patients, timestamps, variables)
    X = np.transpose(X, (0,2,1))

    # Feature extraction
    df_feats = feature_extraction(X, batch_size=100, p=1)
    print("extracted features")

    # Feature selection
    context = {'model_type': model_type, 'transform_type': transform_type}
    top_feats = feature_selection(df_feats, labels=labels, context=context)
    print("selected features")
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

def compare_unsupervised_clustering(X,y):
    unsupervised_accuracies = {}
    for model_type in ['Hierarchical','KMeans','Spectral']:
        for transform_type in ['std','minmax','robust']:
            failed = True
            i = 0
            while failed == True:
                try:
                    y_pred, model = train_t2f_model(X, transform_type, model_type)
                    failed = False
                except Exception as e:
                    print(e)
                    if i > 10:
                        print('failed 10 times')
                        return
                    i += 1

            unsupervised_accuracies[(model_type,transform_type)] = accuracies(y_pred, y)

    return unsupervised_accuracies

def compare_averaged_unsupervised_clustering(X,y):
    unsupervised_accuracies = {}
    for model_type in ['Hierarchical','KMeans','Spectral']:
        for transform_type in ['std','minmax','robust']:
            accuracies_list = []
            for i in range(10):
                failed = True
                i = 0
                while failed == True:
                    if i >10:
                        print('failed 10 times')
                        return
                    try:
                        y_pred, model = train_t2f_model(X, transform_type, model_type)
                        failed = False
                    except Exception as e:
                        print(e)
                        i += 1
                        continue

                accuracies_list += [accuracies(y_pred, y)]


            averaged_accuracies = np.apply_along_axis(np.mean,0,np.array(accuracies_list))
            unsupervised_accuracies[(model_type,transform_type)] = averaged_accuracies

            print((model_type, transform_type))
            print(averaged_accuracies)
            print()

    return unsupervised_accuracies

def compare_averaged_supervised_clustering(X,y,training_sampling):
    supervised_accuracies = {}
    for model_type in ['Hierarchical','KMeans','Spectral']:
        for transform_type in ['std','minmax','robust']:
            accuracies_list = []
            for i in range(10):
                failed = True
                while failed == True:
                    try:
                        y_pred, model = train_t2f_model(X, transform_type, model_type, y, training_sampling)
                        failed = False
                    except:
                        continue

                accuracies_list += [accuracies(y_pred, y)]

            averaged_accuracies = np.apply_along_axis(np.mean,0,np.array(accuracies_list))
            supervised_accuracies[(model_type,transform_type)] = averaged_accuracies

            print((model_type,transform_type))
            print(averaged_accuracies)
            print()

    return supervised_accuracies

def compare_summary_models(X1,y1,X2,y2):
    for pair in [(X1,y1),(X2,y2)]:
        print("pair")
        X = pair[0]
        y = pair[1]

        for model_type in ["svc","rfc","knc"]:
            print(model_type)
            cross_val = cross_validate_summary_model(X, y, model_type)
            print(str(min(cross_val['test_score'])) + ' - ' + str(max(cross_val['test_score'])))

    return

def compare_averages_summary_models(X1,y1,X2,y2):
    for pair in [(X1,y1),(X2,y2)]:
        print("pair")
        X = pair[0]
        y = pair[1]

        for model_type in ["svc","rfc","knc"]:
            accuracies_list = []
            for i in range(10):
                accuracies_list += [score_summary_model(X,y,model_type)]
            print(model_type)
            print(np.mean(accuracies_list))
            print()

    return