"""function for analysis of graph features."""
import logging
import os
from datetime import datetime
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


def _get_classifier(classifier):
    """Get a classifier."""
    if isinstance(classifier, str):
        if classifier == "RF":
            from sklearn.ensemble import RandomForestClassifier

            print("... Using RandomForest classifier ...")
            return RandomForestClassifier(n_estimators=1000, max_depth=30)
        if classifier == "XG":
            from xgboost import XGBClassifier

            print("... Using Xgboost classifier ...")

            return XGBClassifier()
        if classifier == "LGBM":
            print("... Using LGBM classifier ...")
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
):
    """Main function to classify graphs and plot results.

    Args:
        interpretability: an integer in range 1-5
            1 is all features, 5 is features with interpretability=5
    """
    features, features_info = filter_interpretable(
        features, features_info, interpretability
    )

    filtered_features = utils.filter_samples(features, sample_removal=0.3)
    good_features = utils.filter_features(filtered_features)
    normed_features = _normalise_feature_data(good_features)
    classifier = _get_classifier(classifier)

    if grid_search and kfold:
        X, y, shap_values, top_features = fit_grid_search(
            normed_features, classifier=classifier,
        )
    elif kfold:
        X, y, shap_values, top_features = fit_model_kfold(
            normed_features, classifier=classifier, reduced_set_size=100,
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
            plotting.shap_plots(
                X, y, shap_values, results_folder, graphs, max_feats=max_feats
            )
        else:
            plotting.basic_plots(X, top_features, results_folder)

    output_csv(
        normed_features, features_info, top_features, shap_values, results_folder
    )

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


def fit_model_kfold(features, compute_shap=True, classifier=None, reduced_set_size=100):
    """shapeley analysis."""
    if classifier is None:
        raise Exception("Please provide a model for classification")

    X, y = _features_to_Xy(features)

    n_splits = _number_folds(y)
    L.info("Using " + str(n_splits) + " splits")

    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
    indices_all = (indices for indices in folds.split(X, y=y))

    top_features = []
    shap_values = []
    acc_scores = []

    for indices in indices_all:
        top_feature, acc_score, shap_value = compute_fold(
            X, y, classifier, compute_shap, indices
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
    if any(isinstance(el, list) for el in shap_values):
        shap_fold_average = np.mean(shap_values, axis=0)
        shap_fold_average = [
            shap_fold_average[label, :, :]
            for label in range(shap_fold_average.shape[0])
        ]
    else:
        shap_fold_average = [np.mean(shap_values, axis=0)]

    shap_top_features = np.sum(np.mean(np.abs(shap_fold_average), axis=1), axis=0)

    X_reduced_corr = reduce_correlation_feature_set(
        X, shap_top_features, n_feats=reduced_set_size
    )

    shap_values = []
    acc_scores = []
    compute_shap = False
    indices_all = (indices for indices in folds.split(X, y=y))
    for indices in indices_all:
        top_feature, acc_score = compute_fold(
            X_reduced_corr, y, classifier, compute_shap, indices
        )
        acc_scores.append(acc_score)

    L.info(
        "Reduced set via correlation: accuracy: {} +/- {}".format(
            str(np.round(np.mean(acc_scores), 3)), str(np.round(np.std(acc_scores), 3))
        )
    )

    # mean_shap_values = list(np.mean(shap_values, axis=0))

    return X, y, shap_fold_average, top_features


def compute_fold(X, y, classifier, compute_shap, indices):
    """Compute a single fold for parallel computation."""
    train_index, val_index = indices

    # this is so that we take the graph index rather than the pandas index
    x_train_idx = X.index[train_index]
    y_train_idx = y.index[train_index]
    x_val_idx = X.index[val_index]
    y_val_idx = y.index[val_index]

    X_train, X_val = X.loc[x_train_idx], X.loc[x_val_idx]
    y_train, y_val = y.loc[y_train_idx], y.loc[y_val_idx]

    classifier.fit(
        X_train, y_train,
    )
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
        return top_features, acc_score


def fit_model(features, compute_shap=True, classifier=None):
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

    L.info("accuracy: " + str(acc_scores))

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

    start_time = timer(None)
    model.fit(X, y)
    timer(start_time)  # timing ends here for "start_time" variable
    optimal_model = model.best_estimator_

    X, y, mean_shap_values, top_features = fit_model_kfold(
        features, compute_shap=True, classifier=optimal_model
    )

    return X, y, mean_shap_values, top_features


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
    counts = y.value_counts()
    n_splits = int(np.min(counts[counts > 0]) / 2)
    return np.clip(n_splits, 2, 10)


def reduce_correlation_feature_set(X, shap_top_features, n_feats=100, alpha=0.99):
    """Reduce the feature set by taking uncorrelated features."""

    rank_feat = np.argsort(shap_top_features)[::-1]
    # loop over and add those with minimal correlation
    _feats = [rank_feat[0]]
    corr_matrix = np.corrcoef(X.T)
    for i in rank_feat:
        # feat_idx = rank_feat[i]
        # find if there is a correlation larger than alpha
        if not (corr_matrix[_feats, i] > alpha).any():
            _feats.append(i)
        if len(_feats) > n_feats:
            break

    print(n_feats, " of features with <0.99 correlation.")

    X_reduced = X[X.columns[_feats]]
    return X_reduced


def reduce_feature_set(X, top_features, importance_threshold=0.5):
    """Reduce the feature set."""

    # rank_feat = np.argsort(shap_top_features)[::-1]
    mean_importance = np.mean(np.array(top_features), axis=0)
    rank_feat = np.argsort(mean_importance)[::-1]
    n_feat = len(
        np.where(
            np.cumsum(mean_importance[rank_feat])
            < importance_threshold * mean_importance.sum()
        )[0]
    )
    n_feat = max(3, n_feat)

    # n_feat = 2*np.int(np.sqrt(X.shape[1]))

    rank_feat = rank_feat[:n_feat]

    # print(n_feat, " features used in reduced feature set")

    print(n_feat, "to get {} of total importance".format(importance_threshold))

    X_reduced = X[X.columns[rank_feat]]

    return X_reduced


def filter_interpretable(features, features_info, interpretability):
    """Get only features with certain interpretability."""
    interpretability = min(5, interpretability)
    for feat in list(features_info.keys()):
        score = features_info[feat]["interpretability"].score
        if score < interpretability:
            if feat in features.columns:
                features = features.drop(columns=[feat])
                del features_info[feat]
    return features, features_info


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print(
            "\n Time taken: %i hours %i minutes and %s seconds."
            % (thour, tmin, round(tsec, 2))
        )
