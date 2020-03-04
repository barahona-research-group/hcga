"""function for analysis of graph features"""

import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler


def analysis(feature_matrix, features_info, labels):
    """main function to compute analysis"""

    feature_matrix_norm = normalise_feature_data(feature_matrix.T)
    fit_model(feature_matrix_norm, labels)
    shap_values = shapley_analysis(feature_matrix_norm, labels)

    return shap_values


def fit_model(
    X, y, model=RandomForestClassifier(n_estimators=100, max_depth=30), verbose=True
):
    """
    Perform classification of a normalized feature data
    """

    n_splits = number_folds(y)
    skf = StratifiedKFold(n_splits=n_splits, random_state=10, shuffle=True)
    testing_accuracy, top_feats = classify_folds(X, y, model, skf, verbose=False)
    X = reduce_feature_set(X, top_feats)
    testing_accuracy, top_feats = classify_folds(X, y, model, skf)

    return testing_accuracy, top_feats


def classify_folds(X, y, model, skf, verbose=True):
    testing_accuracy = []
    top_feats = []
    for train_index, test_index in skf.split(X, y):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        y_pred = model.fit(X_train, y_train).predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)

        if verbose:
            print(
                "Fold accuracy: --- {0:.3f} --- , Fold balanced accuracy: --- {1:.3f} --- )".format(
                    acc, balanced_acc
                )
            )

        testing_accuracy.append(acc)

        top_feats.append(model.feature_importances_)

    if verbose:
        print("Final accuracy: --- {0:.3f} --- ".format(np.mean(testing_accuracy)))

    return testing_accuracy, top_feats


def reduce_feature_set(X, top_feats, threshold=0.9):
    """    Reduce the feature set   """
    mean_importance = np.mean(np.asarray(top_feats), 0)
    sorted_mean_importance = np.sort(mean_importance)[::-1]

    # Taking only features till we have reached 90% importance
    sum_importance = 0
    final_index = 0
    for i in range(len(sorted_mean_importance)):
        sum_importance = sum_importance + sorted_mean_importance[i]
        if sum_importance > threshold:
            final_index = i
            break
    if final_index < 3:  # take top 2 if no features are selected
        final_index = 3

    top_feat_indices = np.argsort(mean_importance)[::-1][:final_index]

    X_reduced = X[:, top_feat_indices]

    return X_reduced  # , top_feat_indices


def shapley_analysis(X, y, clf=lgb.LGBMClassifier()):

    n_splits = number_folds(y)

    shap_values = None
    (X_train, X_test, y_train, y_test) = train_test_split(
        X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42
    )

    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds_idx = [
        (train_idx, val_idx) for train_idx, val_idx in folds.split(X_train, y=y_train)
    ]
    acc_scores = []
    oof_preds = np.zeros(y.shape[0])

    for n_fold, (train_idx, valid_idx) in enumerate(folds_idx):

        train_x, train_y = X[train_idx, :], y[train_idx]
        valid_x, valid_y = X[valid_idx, :], y[valid_idx]

        clf.fit(
            train_x,
            train_y,
            eval_set=[(train_x, train_y), (valid_x, valid_y)],
            verbose=False,
        )

        explainer = shap.TreeExplainer(clf)

        if shap_values is None:
            shap_values = explainer.shap_values(X_test)
        else:
            shap_values = [
                x + y for x, y in zip(shap_values, explainer.shap_values(X_test))
            ]

        oof_preds[valid_idx] = clf.predict(valid_x)  # [:, 1]
        acc_scores.append(balanced_accuracy_score(valid_y, oof_preds[valid_idx]))

    # print( 'Balanced accuracy: ', np.mean(acc_scores))

    shap_values = [x / n_splits for x in shap_values]
    shap.summary_plot(shap_values, X_test)

    return shap_values


def normalise_feature_data(feature_matrix):
    """
    Normalise the feature matrix using sklearn scaler to remove the mean and scale to unit variance

    """
    return StandardScaler().fit_transform(feature_matrix)


def number_folds(y):
    counts = np.bincount(y)
    n_splits = int(np.min(counts[counts > 0]) / 2)
    if n_splits < 2:
        n_splits = 2
    elif n_splits > 10:
        n_splits = 10
    return n_splits


def define_feature_set(g, data="all"):
    """
    Select subset of features from graphs depending on options
    Parameters
    ----------

    g: Graphs instance from hcga.graphs to be modified

    data: string
        the type of features to classify
            * 'all' : all the features calculated
            * 'feats' : features based on node features and node labels only
            * 'topology' : features based only on the graph topology features

    Returns
    -------
    feature_names: list
        List of feature names to use
    """

    feature_names = [col for col in g.graph_feature_matrix.columns]

    if data != "all":
        matching_NL = [i for i, j in enumerate(feature_names) if "NL" in j]
        matching_NF = [i for i, j in enumerate(feature_names) if "NF" in j]
        matching = matching_NL + matching_NF

    if data == "topology":
        X = np.delete(X, matching, axis=1)
        feature_names = [i for j, i in enumerate(feature_names) if j not in matching]

    elif data == "feats":
        X = X[:, matching]
        feature_names = [i for j, i in enumerate(feature_names) if j in matching]
    return feature_names
