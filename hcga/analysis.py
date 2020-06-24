"""function for analysis of graph features."""
import logging
import os
import time
import itertools
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    KFold,
    train_test_split,
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from . import plotting, utils

L = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

# pylint: disable=too-many-locals,too-many-arguments


def _features_to_Xy(features):
    """decompose features dataframe to X and y."""
    X = features.drop(columns=["labels"])
    y = features["labels"]
    return X, y


def _normalise_feature_data(features,):
    """Normalise the feature matrix using sklearn scaler to remove the mean and scale to unit variance."""
    if "labels" in features:
        labels = features["labels"]

    normed_features = pd.DataFrame(
        StandardScaler().fit_transform(features), columns=features.columns
    )
    normed_features.index = features.index

    if "labels" in features:
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
        "Now using a reduced set of %s features with < %s correlation.",
        str(len(selected_features)),
        str(alpha),
    )

    return X[X.columns[selected_features]]


def print_accuracy(acc_scores, analysis_type):
    """Print the classification or regression accuracies."""
    if analysis_type == "classification":
        L.info(
            "Accuracy: %s +/- %s",
            str(np.round(np.mean(acc_scores), 3)),
            str(np.round(np.std(acc_scores), 3)),
        )
    elif analysis_type == "regression":
        L.info(
            "Mean absolute error: %s +/- %s",
            str(np.round(np.mean(acc_scores), 3)),
            str(np.round(np.std(acc_scores), 3)),
        )


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


def _get_model(model, analysis_type):
    """Get a model."""
    if isinstance(model, str):
        if model == "RF":
            if analysis_type == "classification":
                from sklearn.ensemble import RandomForestClassifier

                L.info("... Using RandomForest classifier ...")
                model = RandomForestClassifier(n_estimators=1000, max_depth=30)
            if analysis_type == "regression":
                from sklearn.ensemble import RandomForestRegressor

                L.info("... Using RandomForest regressor ...")
                model = RandomForestRegressor(n_estimators=1000, max_depth=30)

        elif model == "XG":
            if analysis_type == "classification":
                from xgboost import XGBClassifier

                L.info("... Using Xgboost classifier ...")
                model = XGBClassifier()
            if analysis_type == "regression":
                from xgboost import XGBRegressor

                L.info("... Using Xgboost regressor ...")
                model = XGBRegressor()

        elif model == "LGBM":
            if analysis_type == "classification":
                L.info("... Using LGBM classifier ...")
                from lightgbm import LGBMClassifier

                model = LGBMClassifier()
            if analysis_type == "regression":
                L.info("... Using LGBM regressor ...")
                from lightgbm import LGBMRegressor

                model = LGBMRegressor()
        else:
            raise Exception("Unknown model type: {}".format(model))
    return model


def analysis(  # pylint: disable=inconsistent-return-statements
    features,
    features_info,
    graphs,
    analysis_type="classification",
    folder=".",
    graph_removal=0.3,
    interpretability=1,
    model="XG",
    kfold=True,
    reduce_set=True,
    reduced_set_size=100,
    reduced_set_max_correlation=0.9,
    grid_search=False,
    plot=True,
    max_feats_plot=20,
):
    """Main function to classify graphs and plot results."""
    L.info("%s total features", str(len(features.columns)))

    features = utils.filter_graphs(features, graph_removal=graph_removal)
    features = utils.filter_features(features)
    L.info("%s valid features", str(len(features.columns)))

    features, features_info = _filter_interpretable(
        features, features_info, interpretability
    )
    L.info(
        "%s with interpretability %s", str(len(features.columns)), str(interpretability)
    )

    features = _normalise_feature_data(features)

    if analysis_type == "unsupervised":
        unsupervised_learning(features)
        return

    model = _get_model(model, analysis_type)

    if grid_search and kfold:
        L.info("Using grid_search  and kfold")
        X, y, shap_values, top_features = fit_grid_search(features, model,)
    elif kfold:
        L.info("Using kfold")
        X, y, top_features, shap_values, _ = fit_model_kfold(
            features,
            model,
            analysis_type,
            reduce_set=reduce_set,
            reduced_set_size=reduced_set_size,
            reduced_set_max_correlation=reduced_set_max_correlation,
        )
    else:
        X, y, top_features, shap_values = fit_model(features, model,)

    results_folder = Path(folder) / (
        "results_interpretability_" + str(interpretability)
    )

    if not Path(results_folder).exists():
        os.mkdir(results_folder)

    if plot:
        plotting.shap_plots(
            X,
            y,
            shap_values,
            results_folder,
            graphs,
            analysis_type,
            max_feats=max_feats_plot,
        )

    output_csv(features, features_info, top_features, shap_values, results_folder)

    return X, shap_values


def unsupervised_learning(features):
    """ implementing some basic unsupervised approaches to the data """

    pca = PCA(n_components=2)
    pca.fit(features)
    print("Variance of PC component 1: {}".format(pca.explained_variance_ratio_[0]))
    print("Variance of PC component 2: {}".format(pca.explained_variance_ratio_[1]))
    plotting.pca_plot(features, pca)


def output_csv(features_df, features_info_df, feature_importance, shap_values, folder):
    """Save csv file with analysis data."""
    result_df = features_info_df.copy()
    result_df.loc["feature_importance", features_df.columns[:-1]] = np.vstack(
        feature_importance
    ).mean(axis=0)
    result_df.loc["shap_average", features_df.columns[:-1]] = np.sum(
        np.mean(np.abs(shap_values), axis=1), axis=0
    )

    if len(shap_values) > 1:
        for i, shap_class in enumerate(shap_values):
            result_df.loc[
                "shap_importance: class {}".format(i), features_df.columns[:-1]
            ] = np.vstack(shap_class).mean(axis=0)

    result_df = result_df.sort_values("shap_average", axis=1, ascending=False)
    result_df.to_csv(os.path.join(folder, "importance_results.csv"))


def classify_pairwise(
    features,
    features_info,
    model,
    graph_removal=0.3,
    interpretability=1,
    n_top_features=5,
    min_acc=0.8,
    reduce_set=False,
    reduced_set_size=100,
    reduced_set_max_correlation=0.5,
):
    """Classify all possible pairs of clases with kfold and returns top features.

    The top features for each pair with high enough accuracies are collected in a list,
    for later analysis.
    Args:
        features (dataframe): extracted features
        features_info (): WIP
        classifier (str): name of the classifier to use
        n_top_features (int): number of top features to save
        min_acc (float): minimum accuracy to save top features
        reduce_set (bool): is True, the classification will be rerun
                           on a reduced set of top features (from shapely analysis)
        reduce_set_size (int): number of features to keep for reduces set
        reduced_set_max_correlation (float): to discared highly correlated top features
                                             in reduced set of features
    Returns:
        (dataframe, list, int): accuracies dataframe, list of top features, number of top pairs
    """
    L.info("%s total features", str(len(features.columns)))

    features = utils.filter_graphs(features, graph_removal=graph_removal)
    features = utils.filter_features(features)
    L.info("%s valid features", str(len(features.columns)))

    features, features_info = _filter_interpretable(
        features, features_info, interpretability
    )
    L.info(
        "%s with interpretability %s", str(len(features.columns)), str(interpretability)
    )

    features = _normalise_feature_data(features)

    analysis_type = "classification"
    classifier = _get_model(model, analysis_type=analysis_type)
    classes = features.labels.unique()
    class_pairs = list(itertools.combinations(classes, 2))
    accuracy_matrix = pd.DataFrame(columns=classes, index=classes)

    top_features = []
    n_pairs = 0
    for pair in tqdm(class_pairs):
        features_pair = features.loc[
            (features.labels == pair[0]) | (features.labels == pair[1])
        ]
        X, _, shap_top_features, _, acc_scores = fit_model_kfold(
            features_pair,
            classifier,
            analysis_type,
            reduce_set=reduce_set,
            reduced_set_size=reduced_set_size,
            reduced_set_max_correlation=reduced_set_max_correlation,
        )

        accuracy_matrix.loc[pair[0], pair[1]] = np.round(np.mean(acc_scores), 3)
        accuracy_matrix.loc[pair[1], pair[0]] = np.round(np.mean(acc_scores), 3)

        if np.mean(acc_scores) > min_acc:
            top_features_raw = X.columns[
                np.argsort(shap_top_features)[-n_top_features:]
            ]
            top_features += [f_class + "_" + f for f_class, f in top_features_raw]
            n_pairs += 1

    return accuracy_matrix, top_features, n_pairs


def fit_model_kfold(
    features,
    model,
    analysis_type,
    reduce_set=True,
    reduced_set_size=100,
    reduced_set_max_correlation=0.9,
):
    """Classify graphs from extracted features with kfold.

    Args:
        features (dataframe): extracted features
        classifier (str): name of the classifier to use
        reduce_set (bool): is True, the classification will be rerun
                           on a reduced set of top features (from shapely analysis)
        reduce_set_size (int): number of features to keep for reduces set
        reduced_set_max_correlation (float): to discared highly correlated top features
                                             in reduced set of features
    Returns:
        WIP
    """
    if model is None:
        raise Exception("Please provide a model for classification")

    X, y = _features_to_Xy(features)

    if analysis_type == "classification":
        n_splits = _number_folds(y)
        L.info("Using %s splits", str(n_splits))
        folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
    elif analysis_type == "regression":
        n_splits = 10
        folds = KFold(n_splits=n_splits, shuffle=True, random_state=1)

    shap_values = []
    acc_scores = []
    for indices in folds.split(X, y=y):
        _, acc_score, shap_value = compute_fold(X, y, model, indices, analysis_type,)
        acc_scores.append(acc_score)
        shap_values.append(shap_value)

    print_accuracy(acc_scores, analysis_type)

    if any(isinstance(shap_value, list) for shap_value in shap_values):
        shap_fold_average = list(np.mean(shap_values, axis=0))
    else:
        shap_fold_average = [np.mean(shap_values, axis=0)]

    shap_top_features = np.sum(np.mean(np.abs(shap_fold_average), axis=1), axis=0)

    if not reduce_set:
        return X, y, shap_top_features, shap_fold_average, acc_scores

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
        _, acc_score, shap_value = compute_fold(
            X_reduced_corr, y, model, indices, analysis_type
        )
        shap_values.append(shap_value)
        acc_scores.append(acc_score)

    if any(isinstance(shap_value, list) for shap_value in shap_values):
        shap_fold_average = list(np.mean(shap_values, axis=0))
    else:
        shap_fold_average = [np.mean(shap_values, axis=0)]

    shap_top_features = np.sum(np.mean(np.abs(shap_fold_average), axis=1), axis=0)

    if analysis_type == "classification":
        L.info(
            "Accuracy with reduced set: %s +/- %s",
            str(np.round(np.mean(acc_scores), 3)),
            str(np.round(np.std(acc_scores), 3)),
        )
    elif analysis_type == "regression":
        L.info(
            "Mean Absolute Error with reduced set: %s +/- %s",
            str(np.round(np.mean(acc_scores), 3)),
            str(np.round(np.std(acc_scores), 3)),
        )

    return X_reduced_corr, y, shap_top_features, shap_fold_average, acc_scores


def compute_fold(X, y, model, indices, analysis_type):
    """Compute a single fold for parallel computation."""
    train_index, val_index = indices

    # this is so that we take the graph index rather than the pandas index
    x_train_idx = X.index[train_index]
    y_train_idx = y.index[train_index]
    x_val_idx = X.index[val_index]
    y_val_idx = y.index[val_index]

    X_train, X_val = X.loc[x_train_idx], X.loc[x_val_idx]
    y_train, y_val = y.loc[y_train_idx], y.loc[y_val_idx]

    model.fit(X_train, y_train)
    top_features = model.feature_importances_

    if analysis_type == "classification":
        acc_score = accuracy_score(y_val, model.predict(X_val))
        L.info("Fold accuracy: --- %s ---", str(np.round(acc_score, 3)))
    elif analysis_type == "regression":
        acc_score = mean_absolute_error(y_val, model.predict(X_val))
        L.info("Mean Absolute Error: --- %s ---", str(np.round(acc_score, 3)))

    explainer = shap.TreeExplainer(model, feature_perturbation="interventional",)
    shap_values = explainer.shap_values(X, check_additivity=False)

    return top_features, acc_score, shap_values


def fit_model(features, model):
    """shapeley analysis."""

    explainer = None
    shap_values = None

    if model is None:
        raise Exception("Please provide a model for classification")

    X, y = _features_to_Xy(features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42
    )

    model.fit(X_train, y_train)
    top_features = model.feature_importances_

    explainer = shap.TreeExplainer(model, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_test, check_additivity=False)

    acc_scores = accuracy_score(y_test, model.predict(X_test))

    L.info("Accuracy: %s", str(acc_scores))

    return X, y, top_features, shap_values


def fit_grid_search(features, model):
    """Classify using a grid search, slow and WIP."""
    if model is None:
        raise Exception("Please provide a model for classification")

    X, y = _features_to_Xy(features)

    n_splits = _number_folds(y)
    L.info("Using %s splits", str(n_splits))

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
        estimator=model,
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

    X, y, top_features, mean_shap_values, _ = fit_model_kfold(
        features, optimal_model, "classification",
    )

    return X, y, mean_shap_values, top_features
