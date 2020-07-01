"""function for analysis of graph features."""
import logging
import os
import itertools
from pathlib import Path
import warnings
from tqdm import tqdm

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    RepeatedKFold,
    train_test_split,
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from . import plotting, utils

L = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
warnings.simplefilter("ignore")

# pylint: disable=too-many-locals


def _features_to_Xy(features):
    """decompose features dataframe to numpy arrayes X and y."""
    X = features.drop(columns=["label"])
    y = features["label"]
    return X, y


def _normalise_feature_data(features,):
    """Normalise the feature matrix to remove the mean and scale to unit variance."""
    if "label" in features:
        label = features["label"]

    normed_features = pd.DataFrame(
        StandardScaler().fit_transform(features), columns=features.columns
    )
    normed_features.index = features.index

    if "label" in features:
        normed_features["label"] = label

    return normed_features


def _number_folds(y):
    """Get number of folds."""
    counts = y.value_counts()
    n_splits = int(np.min(counts[counts > 0]) / 2)
    return np.clip(n_splits, 2, 10)


def _get_reduced_feature_set(X, shap_top_features, n_top_features=100, alpha=0.99):
    """Reduce the feature set by taking uncorrelated features."""
    rank_feat_ids = np.argsort(shap_top_features)[::-1]

    # loop over and add those with minimal correlation
    selected_features = [rank_feat_ids[0]]
    corr_matrix = np.corrcoef(X.T)
    for rank_feat_id in rank_feat_ids:
        # find if there is a correlation larger than alpha
        if not (corr_matrix[selected_features, rank_feat_id] > alpha).any():
            selected_features.append(rank_feat_id)
        if len(selected_features) >= n_top_features:
            break

    L.info(
        "Now using a reduced set of %s features with < %s correlation.",
        str(len(selected_features)),
        str(alpha),
    )

    return X.columns[selected_features]


def _print_accuracy(acc_scores, analysis_type, reduced=False):
    """Print the classification or regression accuracies."""
    if reduced:
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

    else:
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
                model = XGBClassifier(objective="reg:squarederror")
            if analysis_type == "regression":
                from xgboost import XGBRegressor

                L.info("... Using Xgboost regressor ...")
                model = XGBRegressor(objective="reg:squarederror")

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


def analysis(  # pylint: disable=too-many-arguments,too-many-locals
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
    plot=True,
    max_feats_plot=20,
    n_repeats=1,
    random_state=1,
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
    model = _get_model(model, analysis_type)

    if kfold:
        L.info("Using kfold")

        (
            X,
            y,
            acc_scores,
            shap_values,
            shap_feat_importance,
            reduced_features,
        ) = fit_model_kfold(
            features,
            model,
            analysis_type,
            reduce_set=reduce_set,
            reduced_set_size=reduced_set_size,
            reduced_set_max_correlation=reduced_set_max_correlation,
            n_repeats=n_repeats,
            random_state=random_state,
        )
    else:
        X, y, acc_scores, shap_values, shap_feat_importance = fit_model(features, model)

    if analysis_type == "regression":
        shap_values = [shap_values]

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

    _save_to_csv(
        features_info,
        shap_values,
        shap_feat_importance,
        reduced_features,
        results_folder,
    )

    return X, acc_scores, shap_values, shap_feat_importance


def unsupervised_learning(features):
    """ implementing some basic unsupervised approaches to the data """

    pca = PCA(n_components=2)
    pca.fit(features)
    L.info("Variance of PC component 1: %s", pca.explained_variance_ratio_[0])
    L.info("Variance of PC component 2: %s", pca.explained_variance_ratio_[1])
    plotting.pca_plot(features, pca)


def _save_to_csv(
    features_info_df,
    shap_values,
    shap_feat_importance,
    reduced_features,
    folder="results",
):
    """Save csv file with analysis data."""
    result_df = features_info_df.copy()
    result_df.loc["shap_feature_importance", reduced_features] = shap_feat_importance

    if len(shap_values) > 1:
        for i, shap_class in enumerate(shap_values):
            result_df.loc[
                "shap_importance: class {}".format(i), reduced_features
            ] = np.vstack(shap_class).mean(axis=0)

    result_df = result_df.sort_values(
        "shap_feature_importance", axis=1, ascending=False
    )
    result_df.to_csv(os.path.join(folder, "importance_results.csv"))


def classify_pairwise(
    features,
    features_info,
    model,
    graph_removal=0.3,
    interpretability=1,
    n_top_features=5,
    reduce_set=False,
    reduced_set_size=100,
    reduced_set_max_correlation=0.5,
    n_repeats=1,
):
    """Classify all possible pairs of clases with kfold and returns top features.

    The top features for each pair with high enough accuracies are collected in a list,
    for later analysis.
    Args:
        features (dataframe): extracted features
        features_info (): WIP
        classifier (str): name of the classifier to use
        n_top_features (int): number of top features to save
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
    classes = features.label.unique()
    class_pairs = list(itertools.combinations(classes, 2))
    accuracy_matrix = pd.DataFrame(columns=classes, index=classes)

    top_features = {}
    for pair in tqdm(class_pairs):
        features_pair = features.loc[
            (features.label == pair[0]) | (features.label == pair[1])
        ]
        X, _, acc_scores, _, shap_feat_importance, _ = fit_model_kfold(
            features_pair,
            classifier,
            analysis_type,
            reduce_set=reduce_set,
            reduced_set_size=reduced_set_size,
            reduced_set_max_correlation=reduced_set_max_correlation,
            n_repeats=n_repeats,
        )

        accuracy_matrix.loc[pair[0], pair[1]] = np.round(np.mean(acc_scores), 3)
        accuracy_matrix.loc[pair[1], pair[0]] = np.round(np.mean(acc_scores), 3)

        top_features_raw = X.columns[np.argsort(shap_feat_importance)[-n_top_features:]]
        top_features[pair] = [f_class + "_" + f for f_class, f in top_features_raw]

    return accuracy_matrix, top_features


def _get_shap_feature_importance(shap_values):
    """From a list of shap values per folds, compute the global shap feature importance."""
    mean_shap_values = np.mean(shap_values, axis=0)
    if isinstance(mean_shap_values, list):
        global_mean_shap_values = np.sum(mean_shap_values, axis=0)
    else:
        global_mean_shap_values = mean_shap_values

    return mean_shap_values, np.mean(abs(global_mean_shap_values), axis=0)


def _evaluate_kfold(X, y, model, folds, analysis_type):
    """Evaluate the kfolds."""
    shap_values = []
    acc_scores = []
    for indices in folds.split(X, y=y):
        acc_score, shap_value = compute_fold(X, y, model, indices, analysis_type)
        shap_values.append(shap_value)
        acc_scores.append(acc_score)
    return acc_scores, shap_values


def fit_model_kfold(
    features,
    model,
    analysis_type,
    reduce_set=True,
    reduced_set_size=100,
    reduced_set_max_correlation=0.9,
    n_repeats=1,
    random_state=1,
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
        folds = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
        )
    elif analysis_type == "regression":
        n_splits = _number_folds(y)
        folds = RepeatedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
        )

    acc_scores, shap_values = _evaluate_kfold(X, y, model, folds, analysis_type)
    _print_accuracy(acc_scores, analysis_type)
    mean_shap_values, shap_feat_importance = _get_shap_feature_importance(shap_values)

    if not reduce_set:
        return X, y, acc_scores, mean_shap_values, shap_feat_importance, X.columns

    reduced_features = _get_reduced_feature_set(
        X,
        shap_feat_importance,
        n_top_features=reduced_set_size,
        alpha=reduced_set_max_correlation,
    )
    X_reduced = X[reduced_features]

    acc_scores, shap_values = _evaluate_kfold(X_reduced, y, model, folds, analysis_type)
    _print_accuracy(acc_scores, analysis_type, reduced=True)
    mean_shap_values, shap_feat_importance = _get_shap_feature_importance(shap_values)

    return (
        X_reduced,
        y,
        acc_scores,
        mean_shap_values,
        shap_feat_importance,
        reduced_features,
    )


def compute_fold(X, y, model, indices, analysis_type):
    """Compute a single fold for parallel computation."""
    train_index, val_index = indices
    model.fit(X.iloc[train_index], y.iloc[train_index])

    if analysis_type == "classification":
        acc_score = accuracy_score(y.iloc[val_index], model.predict(X.iloc[val_index]))
        L.info("Fold accuracy: --- %s ---", str(np.round(acc_score, 3)))
    elif analysis_type == "regression":
        acc_score = mean_absolute_error(
            y.iloc[val_index], model.predict(X.iloc[val_index])
        )

        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(y, model.predict(X), "+")
        plt.savefig("test.png")
        plt.close()

        L.info("Mean Absolute Error: --- %s ---", str(np.round(acc_score, 3)))

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    return acc_score, shap_values


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

    acc_scores = accuracy_score(y_test, model.predict(X_test))

    explainer = shap.TreeExplainer(model, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_test, check_additivity=False)

    L.info("Accuracy: %s", str(acc_scores))

    mean_shap_values, shap_feat_importance = _get_shap_feature_importance([shap_values])
    return X, y, acc_scores, mean_shap_values, shap_feat_importance
