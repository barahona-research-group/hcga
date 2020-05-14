"""function for analysis of graph features."""

import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler

from . import plotting, utils

L = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

# pylint: disable-all


def _features_to_Xy(features):
    """decompose features dataframe to X and y."""
    X = features.drop(columns=["labels"])
    y = features["labels"]
    return X, y


def _normalise_feature_data(features):
    """Normalise the feature matrix using sklearn scaler to remove the mean and scale to unit variance."""
    labels = features["labels"]
    normed_features = pd.DataFrame(
        StandardScaler().fit_transform(features), columns=features.columns
    )
    normed_features.index = features.index
    normed_features["labels"] = labels
    return normed_features


def _number_folds(y):
    """Get number of folds."""
    counts = y.value_counts()
    n_splits = int(np.min(counts[counts > 0]) / 2)
    return np.clip(n_splits, 2, 10)


def _reduce_correlation_feature_set(
    X, shap_top_features, n_top_features=100, alpha=0.99
):
    """Reduce the feature set by taking uncorrelated features."""
    rank_feat_ids = np.argsort(shap_top_features)[::-1]
    # loop over and add those with minimal correlation
    selected_features = [rank_feat_ids[0]]
    corr_matrix = np.corrcoef(X.T)
    for rank_feat_id in rank_feat_ids:
        # find if there is a correlation larger than alpha
        if not (corr_matrix[selected_features, rank_feat_id] > alpha).any():
            selected_features.append(rank_feat_id)
        if len(selected_features) > n_top_features:
            break

    L.info(
        "Now using a reduced set of {} features with < {} correlation.".format(
            len(selected_features), alpha
        )
    )

    return X[X.columns[selected_features]]


def _filter_interpretable(features, features_info, interpretability):
    """Get only features with certain interpretability."""
    interpretability = min(5, interpretability)
    for feat in list(features_info.keys()):
        score = features_info[feat]["interpretability"].score
        if score < interpretability:
            if feat in features.columns:
                features = features.drop(columns=[feat])
                del features_info[feat]
    return features, features_info


def _get_classifier(classifier):
    """Get a classifier."""
    if isinstance(classifier, str):
        if classifier == "RF":
            from sklearn.ensemble import RandomForestClassifier

            L.info("... Using RandomForest classifier ...")
            return RandomForestClassifier(n_estimators=1000, max_depth=30)
        if classifier == "XG":
            from xgboost import XGBClassifier

            L.info("... Using Xgboost classifier ...")

            return XGBClassifier()
        if classifier == "LGBM":
            L.info("... Using LGBM classifier ...")
            from lightgbm import LGBMClassifier

            return LGBMClassifier()
        raise Exception("Unknown classifier type: {}".format(classifier))
    return classifier


def analysis(
    features,
    features_info,
    graphs,
    interpretability=1,
    grid_search=False,
    shap=True,
    classifier="XG",
    folder=".",
    kfold=True,
    plot=True,
    max_feats=20,
    graph_removal=0.3,
    reduced_set_size=100,
    reduced_set_max_correlation=0.9,
):
    """Main function to classify graphs and plot results."""
    L.info("{} total features".format(len(features.columns)))

    features = utils.filter_graphs(features, graph_removal=graph_removal)
    features = utils.filter_features(features)
    L.info("{} valid features".format(len(features.columns)))

    features, features_info = _filter_interpretable(
        features, features_info, interpretability
    )
    L.info(
        "{} with interpretability {}".format(len(features.columns), interpretability)
    )

    features = _normalise_feature_data(features)
    classifier = _get_classifier(classifier)

    if grid_search and kfold:
        L.info("Using grid_search  and kfold")
        X, y, shap_values, top_features = fit_grid_search(
            features, classifier=classifier,
        )
    elif kfold:
        L.info("Using kfold")
        X, y, shap_values, top_features = fit_model_kfold(
            features,
            compute_shap=shap,
            classifier=classifier,
            reduced_set_size=reduced_set_size,
            reduced_set_max_correlation=reduced_set_max_correlation,
        )
    else:
        X, y, shap_values, top_features = fit_model(
            features, classifier=classifier, compute_shap=shap
        )

    results_folder = Path(folder) / (
        "results_interpretability_" + str(interpretability)
    )

    if not Path(results_folder).exists():
        os.mkdir(results_folder)

    if plot:
        if shap:
            plotting.shap_plots(
                X, y, shap_values, results_folder, graphs, max_feats=max_feats
            )
        else:
            plotting.basic_plots(X, top_features, results_folder)

    output_csv(features, features_info, top_features, shap_values, results_folder)

    return X, shap_values


def output_csv(features, features_info, feature_importance, shap_values, folder):
    """save csv file with analysis data."""
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
    if len(shap_values) > 1:
        for i, shap_class in enumerate(shap_values):
            output_df.loc["shap_importance: class {}".format(i)] = np.vstack(
                shap_class
            ).mean(axis=0)

    for feat in output_df.columns:
        # feat_fullname = feat[0] + "_" + feat[1]
        output_df[feat]["feature_info"] = features_info[feat]["description"]
        output_df[feat]["feature_interpretability"] = features_info[feat][
            "interpretability"
        ].score

    # sort by shap average
    output_df = output_df.T.sort_values("shap_average", ascending=False).T

    output_df.to_csv(os.path.join(folder, "importance_results.csv"))


def fit_model_kfold(
    features,
    compute_shap=True,
    classifier=None,
    reduced_set_size=100,
    reduced_set_max_correlation=0.9,
):
    """Classify graphs with kfold."""
    if classifier is None:
        raise Exception("Please provide a model for classification")

    X, y = _features_to_Xy(features)

    n_splits = _number_folds(y)
    L.info("Using " + str(n_splits) + " splits")

    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)

    top_features = []
    shap_values = []
    acc_scores = []
    for indices in folds.split(X, y=y):
        top_feature, acc_score, shap_value = compute_fold(
            X, y, classifier, indices, compute_shap
        )
        top_features.append(top_feature)
        acc_scores.append(acc_score)
        shap_values.append(shap_value)

    L.info(
        "Accuracy: {} +/- {}".format(
            str(np.round(np.mean(acc_scores), 3)), str(np.round(np.std(acc_scores), 3))
        )
    )

    # taking average of folds
    if any(isinstance(shap_value, list) for shap_value in shap_values):
        shap_fold_average = np.mean(shap_values, axis=0)
        shap_fold_average = [
            shap_fold_average[label, :, :]
            for label in range(shap_fold_average.shape[0])
        ]
    else:
        shap_fold_average = [np.mean(shap_values, axis=0)]

    shap_top_features = np.sum(np.mean(np.abs(shap_fold_average), axis=1), axis=0)

    X_reduced_corr = _reduce_correlation_feature_set(
        X,
        shap_top_features,
        n_top_features=reduced_set_size,
        alpha=reduced_set_max_correlation,
    )

    shap_values = []
    acc_scores = []
    indices_all = (indices for indices in folds.split(X, y=y))
    for indices in indices_all:
        top_feature, acc_score, _ = compute_fold(
            X_reduced_corr, y, classifier, indices, compute_shap=False
        )
        acc_scores.append(acc_score)

    L.info(
        "Accuracy with reduced set: {} +/- {}".format(
            str(np.round(np.mean(acc_scores), 3)), str(np.round(np.std(acc_scores), 3))
        )
    )
    return X, y, shap_fold_average, top_features


def compute_fold(X, y, classifier, indices, compute_shap=True):
    """Compute a single fold for parallel computation."""
    train_index, val_index = indices

    # this is so that we take the graph index rather than the pandas index
    x_train_idx = X.index[train_index]
    y_train_idx = y.index[train_index]
    x_val_idx = X.index[val_index]
    y_val_idx = y.index[val_index]

    X_train, X_val = X.loc[x_train_idx], X.loc[x_val_idx]
    y_train, y_val = y.loc[y_train_idx], y.loc[y_val_idx]

    classifier.fit(X_train, y_train)
    top_features = classifier.feature_importances_

    acc_score = accuracy_score(y_val, classifier.predict(X_val))
    L.info("Fold accuracy: --- {0:.3f} ---".format(acc_score))

    if compute_shap:
        explainer = shap.TreeExplainer(
            classifier, feature_perturbation="interventional",
        )
        shap_values = explainer.shap_values(X, check_additivity=False)
        return top_features, acc_score, shap_values
    else:
        return top_features, acc_score, None


def fit_model(features, classifier, compute_shap=True):
    """shapeley analysis."""

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

    acc_scores = accuracy_score(y_test, classifier.predict(X_test))

    L.info("Accuracy: " + str(acc_scores))

    return X, y, explainer, shap_values, top_features


def fit_grid_search(features, compute_shap=True, classifier=None):

    if classifier is None:
        raise Exception("Please provide a model for classification")

    X, y = _features_to_Xy(features)

    n_splits = _number_folds(y)
    L.info("Using " + str(n_splits) + " splits")

    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    max_depth = [5, 10, 15, 20, 25, 30, 35, 50, 70, 100]
    max_depth.append(None)
    min_samples_split = [2, 5, 10, 20]
    min_samples_leaf = [1, 2, 4, 10]
    bootstrap = [True, False]
    max_features = ["auto", "sqrt"]

    random_grid = {
        "n_estimators": n_estimators,
        "max_features": max_features,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "bootstrap": bootstrap,
    }

    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=10)

    model = RandomizedSearchCV(
        estimator=classifier,
        param_distributions=random_grid,
        n_iter=100,
        cv=folds,
        verbose=2,
        random_state=42,
        n_jobs=-1,
    )

    start_time = time.time()
    model.fit(X, y)
    L.info(time.time() - start_time, "seconds")

    optimal_model = model.best_estimator_

    X, y, mean_shap_values, top_features = fit_model_kfold(
        features, compute_shap=True, classifier=optimal_model
    )

    return X, y, mean_shap_values, top_features
