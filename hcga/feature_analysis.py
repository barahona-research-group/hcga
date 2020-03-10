"""function for analysis of graph features"""

import time

import numpy as np
import pandas as pd
import shap
import xgboost
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from .utils import filter_features


def analysis(
    features,
    features_info,
    mode="sklearn",
    classifier_type="RF",
    verbose=True,
    folder=".",
    kfold=False,
):
    """main function to classify graphs"""

    good_features = filter_features(features)
    normed_features = normalise_feature_data(good_features)

    if classifier_type == "RF":
        from sklearn.ensemble import RandomForestClassifier

        classifier = RandomForestClassifier(n_estimators=100, max_depth=30)
    elif classifier_type == "LGBM":
        from lightgbm import LGBMClassifier

        classifier = LGBMClassifier()
    else:
        raise Exception("Unknown classifier type: {}".format(classifier_type))
    if mode == "sklearn":

        if kfold:
            X, testing_accuracy, top_features = fit_model(
                normed_features, classifier=classifier, verbose=verbose
            )
        else:
            print("To implement!")
            return None, None, None

        return X, testing_accuracy, top_features

    if mode == "shap":
        if kfold:
            shap_values = shapley_analysis_kfold(
                normed_features, classifier=classifier, verbose=verbose
            )
        else:
            shap_values = shapley_analysis(
                normed_features, classifier=classifier, verbose=verbose
            )

        # TODO: have a consistent ouptut of feature classification to save, and plot later
        return shap_values, shap_values, shap_values


def _features_to_Xy(features):
    """decompose features dataframe to X and y"""
    X = features.drop(columns=["labels"])
    y = features["labels"]
    return X, y


def fit_model(
    features, classifier=None, verbose=True, reduce_set=True,
):
    """Perform classification of a normalized feature data"""
    if classifier is None:
        raise Exception("Please provide a model for classification")

    X, y = _features_to_Xy(features)

    n_splits = _number_folds(y)
    print("Using", n_splits, "splits")

    stratified_kfold = StratifiedKFold(n_splits=n_splits, random_state=10, shuffle=True)
    testing_accuracy, top_features = classify_folds(
        X, y, classifier, stratified_kfold, verbose=verbose
    )

    if reduce_set:
        X = reduce_feature_set(X, top_features, importance_threshold=0.2)
        testing_accuracy, top_features = classify_folds(
            X, y, classifier, stratified_kfold
        )

    return X, testing_accuracy, top_features


def classify_folds(X, y, classifier, stratified_kfold, verbose=True):
    testing_accuracy = []
    top_features = []
    for train_index, test_index in stratified_kfold.split(X, y):
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)

        if verbose:
            print(
                "Fold accuracy: --- {0:.3f} --- , Fold balanced accuracy: --- {1:.3f} --- )".format(
                    acc, balanced_acc
                )
            )

        testing_accuracy.append(acc)

        top_features.append(classifier.feature_importances_)

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

    return X_reduced


def shapley_analysis(features, classifier=None, verbose=False):
    """shapeley analysis"""

    if classifier is None:
        raise Exception("Please provide a model for classification")

    X, y = _features_to_Xy(features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42
    )

    classifier.fit(X_train, y_train)
    explainer = shap.TreeExplainer(classifier, feature_perturbation="interventional",)

    shap_values = explainer.shap_values(X_test, check_additivity=False)
    acc_scores = balanced_accuracy_score(y_test, classifier.predict(X_test))

    if verbose:
        print("Balanced accuracy: ", acc_scores)

    shap.summary_plot(shap_values[0], X_test)  # , plot_type='dot')
    force = shap.force_plot(
        explainer.expected_value[0], shap_values[0][0, :], X_test.iloc[0, :]
    )
    shap.save_html("test.html", force)
    # shap.dependence_plot("harmonic centrality_max_E", shap_values[0], X_test)

    return shap_values


def shapley_analysis_kfold(features, classifier=None, verbose=False):
    """shapeley analysis"""

    if classifier is None:
        raise Exception("Please provide a model for classification")

    X, y = _features_to_Xy(features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42
    )

    n_splits = _number_folds(y)
    print("Using", n_splits, "splits")

    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    shap_values = None
    acc_scores = []
    oof_preds = np.zeros(y.shape[0])
    for train_index, test_index in folds.split(X_train, y=y_train):

        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        classifier.fit(
            X_train, y_train,
        )

        explainer = shap.TreeExplainer(
            classifier, feature_perturbation="interventional",
        )

        if shap_values is None:
            shap_values = explainer.shap_values(X_test, check_additivity=False)
        else:
            shap_values = [
                x + y
                for x, y in zip(
                    shap_values, explainer.shap_values(X_test, check_additivity=False)
                )
            ]

        oof_preds[test_index] = classifier.predict(X_test)
        acc_scores.append(balanced_accuracy_score(y_test, oof_preds[test_index]))

        if verbose:
            print("Fold accuracy: --- {0:.3f} ---)".format(acc_scores[-1]))

    if verbose:
        print("Balanced accuracy: ", np.mean(acc_scores))

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


def _number_folds(y):
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
