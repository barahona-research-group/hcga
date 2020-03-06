"""function for analysis of graph features"""

import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
import xgboost
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from .utils import filter_features


def analysis(features, features_info, mode="sklearn_RF", verbose=True, folder="."):
    """main function to classify graphs"""

    good_features = filter_features(features)
    normed_features = normalise_feature_data(good_features)

    if mode == "sklearn_RF":

        from sklearn.ensemble import RandomForestClassifier

        learning_model = RandomForestClassifier(n_estimators=100, max_depth=30)
        X, testing_accuracy, top_features = fit_model(
            normed_features, model=learning_model, verbose=verbose
        )

        return X, testing_accuracy, top_features

    if mode == "shap":
        shap_values = shapley_analysis(normed_features, labels)
        return shap_values


def fit_model(
    features, model=None, verbose=True, reduce_set=True,
):
    """Perform classification of a normalized feature data"""
    if model is None:
        raise Exception("Please provide a model for classification")

    X = features.drop(columns=["labels"])
    y = features["labels"]

    n_splits = number_folds(y)
    print("Using", n_splits, "splits")

    stratified_kfold = StratifiedKFold(n_splits=n_splits, random_state=10, shuffle=True)
    testing_accuracy, top_features = classify_folds(
        X, y, model, stratified_kfold, verbose=verbose
    )

    if reduce_set:
        X = reduce_feature_set(X, top_features, importance_threshold=0.2)
        testing_accuracy, top_features = classify_folds(X, y, model, stratified_kfold)

    return X, testing_accuracy, top_features


def classify_folds(X, y, model, stratified_kfold, verbose=True):
    testing_accuracy = []
    top_features = []
    for train_index, test_index in stratified_kfold.split(X, y):
        X_train, X_test = X.loc[train_index], X.loc[test_index]
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

        top_features.append(model.feature_importances_)

    if verbose:
        print("Final accuracy: --- {0:.3f} --- ".format(np.mean(testing_accuracy)))

    return testing_accuracy, top_features


def reduce_feature_set(X, top_features, importance_threshold=0.9):
    """    Reduce the feature set   """

    mean_importance = np.mean(np.array(top_features), axis=0)
    rank_feat = np.argsort(mean_importance)[::-1]
    n_feat = len(
        np.where(
            np.cumsum(mean_importance[rank_feat])
            < importance_threshold * mean_importance.sum()
        )[0]
    )
    n_feat = max(3, n_feat)
    rank_feat = rank_feat[:n_feat]

    print(n_feat, "to get .9 of total importance")

    X_reduced = X.iloc[:, rank_feat]

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


def normalise_feature_data(features):
    """Normalise the feature matrix using sklearn scaler to remove the mean and scale to unit variance"""
    labels = features["labels"]
    normed_features = pd.DataFrame(
        StandardScaler().fit_transform(features), columns=features.columns
    )
    normed_features["labels"] = labels
    return normed_features


def number_folds(y):
    counts = np.bincount(y)
    n_splits = int(np.min(counts[counts > 0]) / 2)
    return np.clip(n_splits, 2, 10)


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
