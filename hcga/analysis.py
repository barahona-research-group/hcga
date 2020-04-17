"""function for analysis of graph features"""

import os
import time
from pathlib import Path
import logging
from functools import partial
import multiprocessing

import numpy as np
import pandas as pd
import shap
import xgboost
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from . import plotting, utils

L = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def _get_classifier(classifier):
    """Get a classifier."""
    if isinstance(classifier, str):
        if classifier == "RF":
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(n_estimators=100, max_depth=30)
        if classifier == "LGBM":
            from lightgbm import LGBMClassifier

            return LGBMClassifier()
        raise Exception("Unknown classifier type: {}".format(classifier))
    return classifier


def analysis(
    features,
    features_info,
    graphs,
    interpretability=1,
    shap=True,
    classifier="RF",
    folder=".",
    kfold=True,
    plot=True,
    max_feats=20,
):
    """Main function to classify graphs and plot results.
    
    Parameters 
    -----
    interpretability: an integer in range 1-5
        1 is all features, 5 is features with interpretability=5
    
    """
    features, features_info = filter_interpretable(
        features, features_info, interpretability
    )

    good_features = utils.filter_features(features)
    normed_features = normalise_feature_data(good_features)
    classifier = _get_classifier(classifier)

    if kfold:
        X, y, shap_values, top_features = fit_model_kfold(
            normed_features, classifier=classifier,
        )
    else:
        X, y, shap_values, top_features = fit_model(
            normed_features, classifier=classifier,
        )

    results_folder = Path(folder) / (
        "results_interpretability_" + str(interpretability)
    )
    if not Path(results_folder).exists():
        os.mkdir(results_folder)

    if plot:
        if shap:
            plotting.shap_plots(X, y, shap_values, results_folder, graphs, max_feats=max_feats)
        else:
            plotting.basic_plots(X, top_features, results_folder)

    output_csv(
        normed_features, features_info, top_features, shap_values, results_folder
    )

    return X, shap_values


def output_csv(features, features_info, feature_importance, shap_values, folder):
    """save csv file with analysis data"""
    X, y = _features_to_Xy(features)

    index_rows = [
        "feature_info",
        "feature_interpretability",
        "feature_importance",
        "shap_average",
    ]
    output_df = pd.DataFrame(columns=X.columns, index=index_rows)

    output_df.loc["feature_importance"] = np.vstack(feature_importance).mean(axis=0)

    output_df.loc["shap_average"] = np.sum(np.mean(np.abs(shap_values), axis=1), axis=0)

    # looping over shap values for each class
    for i, shap_class in enumerate(shap_values):
        output_df.loc["shap_importance: class {}".format(i)] = np.vstack(
            shap_class
        ).mean(axis=0)

    for feat in output_df.columns:
        output_df[feat]["feature_info"] = features_info[feat]["feature_description"]
        output_df[feat]["feature_interpretability"] = features_info[feat][
            "feature_interpretability"
        ].score

    # sort by shap average
    output_df = output_df.T.sort_values("shap_average", ascending=False).T

    output_df.to_csv(os.path.join(folder, "importance_results.csv"))


def fit_model_kfold(features, compute_shap=True, classifier=None):
    """shapeley analysis"""
    if classifier is None:
        raise Exception("Please provide a model for classification")

    X, y = _features_to_Xy(features)

    n_splits = _number_folds(y)
    L.info("Using " + str(n_splits) + " splits")

    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    indices_all = (indices for indices in folds.split(X, y=y))

    top_features = []
    shap_values = []
    acc_scores = []
    with multiprocessing.Pool(n_splits) as pool:
        for top_feature, acc_score, shap_value in pool.imap(
            partial(compute_fold, X, y, classifier, compute_shap), indices_all
        ):
            top_features.append(top_feature)
            acc_scores.append(acc_score)
            shap_values.append(shap_value)

    L.info("Balanced accuracy: " + str(np.round(np.mean(acc_scores), 3)))

    mean_shap_values = list(np.mean(shap_values, axis=0))

    return X, y, mean_shap_values, top_features


def compute_fold(X, y, classifier, compute_shap, indices):
    """Compute a single fold for parallel computation."""
    train_index, val_index = indices
    X_train, X_val = X.loc[train_index], X.loc[val_index]
    y_train, y_val = y[train_index], y[val_index]

    classifier.fit(
        X_train, y_train,
    )
    top_features = classifier.feature_importances_

    explainer = shap.TreeExplainer(classifier, feature_perturbation="interventional",)

    shap_values = np.array(explainer.shap_values(X, check_additivity=False))

    acc_score = balanced_accuracy_score(y_val, classifier.predict(X_val))

    L.info("Fold accuracy: --- {0:.3f} ---".format(acc_score))
    return top_features, acc_score, shap_values


def fit_model(features, compute_shap=True, classifier=None):
    """shapeley analysis"""

    explainer = None
    shap_values = None

    if classifier is None:
        raise Exception("Please provide a model for classification")

    X, y = _features_to_Xy(features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42
    )

    classifier.fit(X_train, y_train)
    top_features = classifier.feature_importances_

    if compute_shap:
        explainer = shap.TreeExplainer(
            classifier, feature_perturbation="interventional"
        )
        shap_values = explainer.shap_values(X_test, check_additivity=False)

    acc_scores = balanced_accuracy_score(y_test, classifier.predict(X_test))

    L.info("Balanced accuracy: " + str(acc_scores))

    return X, y, explainer, shap_values, top_features


def _features_to_Xy(features):
    """decompose features dataframe to X and y"""
    X = features.drop(columns=["labels"])
    y = features["labels"]
    return X, y


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


def reduce_feature_set(X, y, top_features, classifier, importance_threshold=0.9):
    """Reduce the feature set."""

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


def filter_interpretable(features, features_info, interpretability):
    """Get only features with certain interpretability."""
    interpretability = min(5, interpretability)
    for feat in list(features_info.keys()):
        score = features_info[feat]["feature_interpretability"].score
        if score < interpretability:
            features = features.drop(columns=[feat])
            del features_info[feat]
    return features, features_info
