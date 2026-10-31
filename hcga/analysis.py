"""
Functions for the analysis of extracted feature matrix.
"""

import itertools
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import (
    RepeatedKFold,
    RepeatedStratifiedKFold,
    ShuffleSplit,
    StratifiedShuffleSplit,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm

from hcga.io import load_fitted_model, save_fitted_model
from hcga.plotting import plot_analysis, plot_prediction, shap_class_names

L = logging.getLogger(__name__)


def features_to_Xy(features):
    """Decompose features dataframe to numpy arrays X and y."""
    if "label" in features:
        return features.drop(columns=["label"]), features["label"]

    return features, None


def _add_label(X, y):
    """Append the label column to a feature matrix (as ("label", "") for MultiIndex columns)."""
    X = X.copy()
    label_col = ("label", "") if isinstance(X.columns, pd.MultiIndex) else "label"
    X[label_col] = y.loc[X.index].values
    return X


def _encode_labels(y):
    """Encode class labels to 0..n_classes-1, as required by xgboost."""
    label_encoder = LabelEncoder()
    y_encoded = pd.Series(label_encoder.fit_transform(y), index=y.index, name=y.name)
    return y_encoded, label_encoder


def _normalise_feature_data(features, scaler=None, fit_scaler=True):
    """Normalise the feature matrix (without label) to zero mean and unit variance."""
    if scaler is None:
        scaler = StandardScaler()
        fit_scaler = True

    if fit_scaler:
        normed_features = scaler.fit_transform(features)
    else:
        normed_features = scaler.transform(features)

    normed_features = pd.DataFrame(normed_features, columns=features.columns, index=features.index)
    return normed_features, scaler


def _number_folds(y):
    """Get number of folds."""
    counts = y.value_counts()
    L.info("Counts of graphs/label: \n%s", counts)
    n_splits = int(np.min(counts[counts > 0]) / 2)
    return np.clip(n_splits, 2, 10)


def _get_reduced_feature_set(X, shap_top_features, n_top_features=100, alpha=0.9):
    """Reduce the feature set by taking uncorrelated features."""
    rank_feat_ids = np.argsort(shap_top_features)[::-1]

    selected_features = [rank_feat_ids[0]]
    with np.errstate(invalid="ignore", divide="ignore"):
        corr_matrix = np.abs(np.corrcoef(X.T))
    for rank_feat_id in rank_feat_ids[1:]:
        if not (corr_matrix[selected_features, rank_feat_id] > alpha).any():
            selected_features.append(rank_feat_id)
        if len(selected_features) >= n_top_features:
            break

    L.info(
        "Now using a reduced set of %s features with < %s correlation.",
        str(len(selected_features)),
        str(alpha),
    )
    if len(selected_features) == 1:
        L.warning("Only one features selected, things may break!")

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
    scores = features_info.loc["interpretability"].map(lambda score: score.score)
    to_drop = scores.index[scores < interpretability].intersection(features.columns)
    return features.drop(columns=to_drop), features_info.drop(columns=to_drop)


def _filter_graphs(features, graph_removal=0.05):
    """Remove samples with more than X% bad values."""
    n_graphs = len(features.index)
    features = features[features.isnull().sum(axis=1) / len(features.columns) < graph_removal]
    L.info(
        "%s graphs were removed for more than %s fraction of bad features",
        str(n_graphs - len(features.index)),
        str(graph_removal),
    )
    return features


def _filter_features(features):
    """Filter features and create feature matrix."""
    nan_features = features.replace([np.inf, -np.inf], np.nan)
    valid_features = nan_features.dropna(axis=1).astype("float64")
    return valid_features.drop(
        valid_features.std()[(valid_features.std() == 0)].index, axis=1
    ).columns


def _get_model(model, analysis_type, random_state=42):
    """Get a model."""
    if isinstance(model, str):
        if model == "RF":
            if analysis_type == "classification":
                from sklearn.ensemble import RandomForestClassifier

                L.info("... Using RandomForest classifier ...")
                model = RandomForestClassifier(
                    n_estimators=1000, max_depth=30, random_state=random_state, n_jobs=-1
                )
            if analysis_type == "regression":
                from sklearn.ensemble import RandomForestRegressor

                L.info("... Using RandomForest regressor ...")
                model = RandomForestRegressor(
                    n_estimators=1000, max_depth=30, random_state=random_state, n_jobs=-1
                )

        elif model == "XG":
            if analysis_type == "classification":
                from xgboost import XGBClassifier

                L.info("... Using Xgboost classifier ...")
                model = XGBClassifier(eval_metric="mlogloss")
            if analysis_type == "regression":
                from xgboost import XGBRegressor

                L.info("... Using Xgboost regressor ...")
                model = XGBRegressor(objective="reg:squarederror", eval_metric="mlogloss")

        else:
            raise Exception(f"Unknown model type: {model}")
    return model


def _to_class_list(shap_values):
    """Normalise a SHAP output to a list of per-class (n_samples, n_features) arrays.

    Depending on the model and the shap version, shap returns a list of per-class arrays,
    a (n_samples, n_features) array (regression, binary xgboost) or a
    (n_samples, n_features, n_classes) array (multiclass, binary random forest).
    """
    if isinstance(shap_values, list):
        return [np.asarray(shap_class) for shap_class in shap_values]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 2:
        return [shap_values]
    return [shap_values[:, :, i] for i in range(shap_values.shape[2])]


def _get_shap_feature_importance(shap_values):
    """From a list of shap values per folds, compute the global shap feature importance.

    Returns:
        mean_shap_values (list): per-class (n_samples, n_features) arrays averaged across folds
        shap_feature_importance (array): (n_features,) mean |SHAP| over samples and classes
    """
    # average across folds, per class
    per_fold = [np.stack(_to_class_list(shap_value)) for shap_value in shap_values]
    mean_shap_values = list(np.mean(per_fold, axis=0))

    # mean |SHAP| across graphs, then across classes
    shap_feature_importance = np.mean(
        [np.mean(np.abs(shap_class), axis=0) for shap_class in mean_shap_values], axis=0
    )
    return mean_shap_values, shap_feature_importance


def _evaluate_kfold(X, y, model, folds, analysis_type, compute_shap):
    """Evaluate the kfolds."""
    shap_values = []
    acc_scores = []
    for indices in folds.split(X, y=y):
        acc_score, shap_value = _compute_fold(X, y, model, indices, analysis_type, compute_shap)
        shap_values.append(shap_value)
        acc_scores.append(acc_score)
    return acc_scores, shap_values


def _evaluate_kfold_nested(
    X, y, model, folds, analysis_type, reduced_set_size, reduced_set_max_correlation
):
    """Evaluate the reduced feature set with the selection nested inside each training fold.

    In each fold the model is fitted on the training rows, the reduced set is selected from SHAP
    values on the training rows only, and the refitted reduced model is scored on the held-out
    rows, so the score is not biased by selecting features on the test labels.
    """
    acc_scores = []
    for train_index, val_index in folds.split(X, y=y):
        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        model.fit(X_train, y_train)
        _, shap_feature_importance = _get_shap_feature_importance(
            [_compute_shap(model, X_train, X_train)]
        )
        reduced_features = _get_reduced_feature_set(
            X_train,
            shap_feature_importance,
            n_top_features=reduced_set_size,
            alpha=reduced_set_max_correlation,
        )
        acc_score, _ = _compute_fold(
            X[reduced_features], y, model, (train_index, val_index), analysis_type, False
        )
        acc_scores.append(acc_score)
    return acc_scores


def fit_model_kfold(
    features,
    model,
    analysis_type="classification",
    reduce_set=True,
    reduced_set_size=100,
    reduced_set_max_correlation=0.9,
    n_repeats=1,
    random_state=42,
    n_splits=None,
    compute_shap=True,
    nested_selection=True,
):
    """Classify graphs from extracted features with kfold.

    Args:
        features (dataframe): extracted features
        model (str): model to preform analysis
        analysis_type (str): 'classification' or 'regression'
        reduce_set (bool): is True, the classification will be rerun
                           on a reduced set of top features (from shapely analysis)
        reduce_set_size (int): number of features to keep for reduces set
        reduced_set_max_correlation (float): to discared highly correlated top features
                                             in reduced set of features
        n_repeats (int): number of k-fold repeats
        random_state (int): rng seed
        n_splits (int): numbere of split for k-fold, None=automatic estimation
            (from the class counts for classification, 5 for regression)
        compute_shap (bool): compute SHAP values or not
        nested_selection (bool): if True, the reduced-set scores come from selecting the reduced
            set inside each training fold (unbiased). The reduced set selected on all data is only
            reported as the descriptive list of top features. If False, the reduced set selected
            on all data is re-scored on the same folds (optimistically biased).

    Returns:
        (dict): dictionary with results
    """
    if model is None:
        raise Exception("Please provide a model for classification")

    X, y = features_to_Xy(features)

    label_encoder = None
    if analysis_type == "classification":
        y, label_encoder = _encode_labels(y)
        if n_splits is None:
            n_splits = _number_folds(y)
        L.info("Using %s splits", str(n_splits))
        folds = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
        )
    elif analysis_type == "regression":
        if n_splits is None:
            n_splits = 5
        L.info("Using %s splits", str(n_splits))
        folds = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    acc_scores, shap_values = _evaluate_kfold(X, y, model, folds, analysis_type, compute_shap)
    _print_accuracy(acc_scores, analysis_type)

    if compute_shap:
        mean_shap_values, shap_feature_importance = _get_shap_feature_importance(shap_values)
    else:
        mean_shap_values = None
        shap_feature_importance = None

    analysis_results = {
        "X": X,
        "y": y,
        "acc_scores": acc_scores,
        "mean_shap_values": mean_shap_values,
        "shap_values": shap_values,
        "shap_feature_importance": shap_feature_importance,
        "reduced_features": None,
        "label_encoder": label_encoder,
    }
    if not reduce_set:
        return analysis_results

    if not compute_shap:
        return analysis_results

    # descriptive reduced set of top features, selected on all data
    reduced_features = _get_reduced_feature_set(
        X,
        shap_feature_importance,
        n_top_features=reduced_set_size,
        alpha=reduced_set_max_correlation,
    )
    reduced_acc_scores, reduced_shap_values = _evaluate_kfold(
        X[reduced_features], y, model, folds, analysis_type, compute_shap
    )
    if nested_selection:
        L.info("Scoring the reduced set with the selection nested in each training fold")
        reduced_acc_scores = _evaluate_kfold_nested(
            X, y, model, folds, analysis_type, reduced_set_size, reduced_set_max_correlation
        )
    _print_accuracy(reduced_acc_scores, analysis_type, reduced=True)
    (
        reduced_mean_shap_values,
        reduced_shap_feature_importance,
    ) = _get_shap_feature_importance(reduced_shap_values)

    analysis_results.update(
        {
            "reduced_features": reduced_features,
            "reduced_shap_values": reduced_shap_values,
            "shap_values": shap_values,
            "reduced_acc_scores": reduced_acc_scores,
            "reduced_mean_shap_values": reduced_mean_shap_values,
            "reduced_shap_feature_importance": reduced_shap_feature_importance,
        }
    )
    return analysis_results


def fit_model(
    features,
    model,
    analysis_type="classification",
    reduce_set=True,
    reduced_set_size=100,
    reduced_set_max_correlation=0.9,
    test_size=0.2,
    random_state=42,
    compute_shap=True,
):
    """Train a single model.

    Args:
        features (dataframe): extracted features
        model (str): model to preform analysis
        analysis_type (str): 'classification' or 'regression'
        reduce_set (bool): is True, the classification will be rerun
                           on a reduced set of top features (from shapely analysis)
        reduce_set_size (int): number of features to keep for reduces set
        reduced_set_max_correlation (float): to discared highly correlated top features
                                             in reduced set of features
        random_state (int): rng seed
        test_size (float): size of test dataset (see sklearn.model_selection.ShuffleSplit)
        compute_shap (bool): compute SHAP values or not

    Returns:
        (dict): dictionary with results
    """
    if model is None:
        raise Exception("Please provide a model for classification")

    X, y = features_to_Xy(features)

    label_encoder = None
    if analysis_type == "classification":
        y, label_encoder = _encode_labels(y)
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
    else:
        splitter = ShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    indices = next(splitter.split(X, y))
    acc_score, shap_values = _compute_fold(X, y, model, indices, analysis_type, compute_shap)

    if compute_shap:
        mean_shap_values, shap_feature_importance = _get_shap_feature_importance([shap_values])
    else:
        mean_shap_values = None
        shap_feature_importance = None

    analysis_results = {
        "X": X,
        "y": y,
        "acc_score": acc_score,
        "mean_shap_values": mean_shap_values,
        "shap_feature_importance": shap_feature_importance,
        "model": model,
        "indices": indices,
        "reduced_features": None,
        "label_encoder": label_encoder,
    }
    if not reduce_set or not compute_shap:
        return analysis_results

    reduced_features = _get_reduced_feature_set(
        X,
        shap_feature_importance,
        n_top_features=reduced_set_size,
        alpha=reduced_set_max_correlation,
    )
    reduced_model = clone(model)
    reduced_acc_score, reduced_shap_values = _compute_fold(
        X[reduced_features], y, reduced_model, indices, analysis_type, compute_shap
    )
    (
        reduced_mean_shap_values,
        reduced_shap_feature_importance,
    ) = _get_shap_feature_importance([reduced_shap_values])

    analysis_results.update(
        {
            "reduced_features": reduced_features,
            "reduced_acc_score": reduced_acc_score,
            "reduced_mean_shap_values": reduced_mean_shap_values,
            "reduced_shap_feature_importance": reduced_shap_feature_importance,
            "reduced_model": reduced_model,
        }
    )
    return analysis_results


def _compute_fold(X, y, model, indices, analysis_type, compute_shap):
    """Compute a single fold for parallel computation."""
    train_index, val_index = indices
    model.fit(X.iloc[train_index], y.iloc[train_index])

    if analysis_type == "classification":
        acc_score = accuracy_score(y.iloc[val_index], model.predict(X.iloc[val_index]))
        L.info("Fold accuracy: --- %s ---", str(np.round(acc_score, 3)))

    elif analysis_type == "regression":
        acc_score = mean_absolute_error(y.iloc[val_index], model.predict(X.iloc[val_index]))
        L.info("Mean Absolute Error: --- %s ---", str(np.round(acc_score, 3)))

    if compute_shap:
        shap_values = _compute_shap(model, X, X.iloc[train_index])
    else:
        shap_values = None

    return acc_score, shap_values


def _compute_shap(model, X, X_background):
    """Compute the SHAP values of a fitted model on X."""
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:  # pylint: disable=broad-except
        explainer = shap.KernelExplainer(getattr(model, "predict_proba", model.predict), X_background)
    return explainer.shap_values(X)


def _preprocess_features(
    features, features_info, graph_removal, interpretability, trained_model=None
):
    """Collect all feature filters.

    The label column is split off first, so that it is not filtered, cast to float or scaled.
    With a trained model, the features are aligned to the model's features and scaled with the
    stored scaler (not refitted on the new data).
    """
    L.info("%s total features", str(len(features.columns)))

    X, y = features_to_Xy(features)
    X = _filter_graphs(X, graph_removal=graph_removal)

    if trained_model is not None:
        _, scaler, model_features = trained_model[:3]
        X = X.replace([np.inf, -np.inf], np.nan).reindex(columns=model_features.columns)
        features_info = model_features
        X, scaler = _normalise_feature_data(X, scaler=scaler, fit_scaler=False)
    else:
        good_features = _filter_features(X)
        X = X[good_features]
        features_info = features_info[good_features]
        L.info("%s valid features", str(len(X.columns)))

        X, features_info = _filter_interpretable(X, features_info, interpretability)
        L.info("%s with interpretability %s", str(len(X.columns)), str(interpretability))

        X, scaler = _normalise_feature_data(X)

    if y is not None:
        X = _add_label(X, y)

    return X, features_info, scaler


def train_all(features, model, label_encoder=None):
    """Train on all available data."""
    X, y = features_to_Xy(features)
    if label_encoder is not None:
        y = label_encoder.transform(y)
    model.fit(X, y)
    L.info("Fitting model to all data")
    return model


def predict_unlabelled(model, features, label_encoder=None):
    """Predict unlabelled data, with predictions in the original label values."""
    X, _ = features_to_Xy(features)
    predictions = model.predict(X)
    if label_encoder is None:
        label_encoder = getattr(model, "label_encoder_", None)
    if label_encoder is not None:
        predictions = label_encoder.inverse_transform(np.asarray(predictions).astype(int))
    return predictions


def analysis(  # pylint: disable=too-many-arguments,too-many-locals,too-many-branches
    features,
    features_info,
    graphs=None,
    analysis_type="classification",
    folder=".",
    graph_removal=0.3,
    interpretability=1,
    model="XG",
    compute_shap=True,
    kfold=True,
    reduce_set=True,
    reduced_set_size=100,
    reduced_set_max_correlation=0.9,
    plot=True,
    max_feats_plot=20,
    max_feats_plot_dendrogram=100,
    n_repeats=1,
    n_splits=None,
    random_state=42,
    test_size=0.2,
    trained_model=None,
    save_model=False,
    nested_selection=True,
):
    """Main function to classify graphs and plot results.

    Args:
        features (dataframe): extracted features
        features_info (dataframe): features information
        graphs (GraphCollection): input graphs
        analysis_type (str): 'classification' or 'regression'
        folder (str): folder to save analysis
        graph_removal (float): remove samples with more than graph_removal % bad values
        interpretabiliy (int): filter out features below this interpretability
        model (str): model to preform analysis
        compute_shap (bool): compute SHAP values or not
        kfold (bool): run with kfold
        reduce_set (bool): is True, the classification will be rerun
                           on a reduced set of top features (from shapely analysis)
        reduce_set_size (int): number of features to keep for reduces set
        reduced_set_max_correlation (float): to discared highly correlated top features
                                             in reduced set of features
        plot (bool): save plots
        max_feats_plot (int): max number of feature analysis to plot
        n_repeats (int): number of k-fold repeats
        n_splits (int): numbere of split for k-fold, None=automatic estimation
        random_state (int): rng seed
        test_size (float): size of test dataset (see sklearn.model_selection.ShuffleSplit)
        trained_model (str): provide path to pretrained model to apply to new data
        save_modeel (bool): save the obtained model to reuse later
        nested_selection (bool): with kfold, score the reduced set by selecting it inside each
            training fold (see fit_model_kfold)

    Returns:
        (dict): dictionary with results
    """

    if trained_model is None:
        model = _get_model(model, analysis_type, random_state=random_state)
    else:
        if isinstance(trained_model, (tuple, list)):
            model = trained_model[0]
        elif isinstance(trained_model, str):
            trained_model = load_fitted_model(trained_model)
            model = trained_model[0]

    features, features_info, scaler = _preprocess_features(
        features, features_info, graph_removal, interpretability, trained_model
    )

    if not Path(folder).exists():
        os.mkdir(folder)

    if trained_model is not None:
        y_predictions = predict_unlabelled(model, features)
        _save_predictions_to_csv(features, y_predictions, folder=folder)
        return y_predictions

    if kfold:
        analysis_results = fit_model_kfold(
            features,
            model,
            analysis_type=analysis_type,
            reduce_set=reduce_set,
            reduced_set_size=reduced_set_size,
            reduced_set_max_correlation=reduced_set_max_correlation,
            n_repeats=n_repeats,
            random_state=random_state,
            n_splits=n_splits,
            compute_shap=compute_shap,
            nested_selection=nested_selection,
        )
    else:
        analysis_results = fit_model(
            features,
            model,
            analysis_type=analysis_type,
            reduce_set=reduce_set,
            reduced_set_size=reduced_set_size,
            reduced_set_max_correlation=reduced_set_max_correlation,
            random_state=random_state,
            test_size=test_size,
            compute_shap=compute_shap,
        )

    label_encoder = analysis_results["label_encoder"]
    if save_model:
        fitted_model = train_all(features, clone(model), label_encoder=label_encoder)
        # stored with the model, to return predictions in the original labels
        fitted_model.label_encoder_ = label_encoder

    results_folder = Path(folder) / ("results_interpretability_" + str(interpretability))
    if not Path(results_folder).exists():
        os.mkdir(results_folder)

    if compute_shap:
        if plot and kfold:
            plot_analysis(
                analysis_results,
                results_folder,
                graphs,
                analysis_type,
                max_feats_plot,
                max_feats_plot_dendrogram,
            )
        if plot and not kfold:
            plot_prediction(analysis_results, results_folder)

        if kfold:
            _save_to_csv(features_info, analysis_results, results_folder)

    model_folder = Path(folder) / ("fitted_model")
    if not Path(model_folder).exists():
        os.mkdir(model_folder)

    if save_model:
        save_fitted_model(fitted_model, scaler, features_info, model_folder)

    return analysis_results


def _save_predictions_to_csv(features, predictions, folder="results"):
    """Save the prediction results for unlabelled data."""
    results_df = pd.DataFrame(data=predictions, index=features.index, columns=["y_prediction"])
    results_df.to_csv(os.path.join(folder, "prediction_results.csv"))


def _save_to_csv(features_info_df, analysis_results, folder="results"):
    """Save csv file with analysis data."""

    result_df = features_info_df.copy()
    columns = analysis_results["X"].columns
    result_df.loc["shap_feature_importance"] = pd.Series(
        analysis_results["shap_feature_importance"], index=columns
    )
    label_encoder = analysis_results.get("label_encoder")
    class_names = None if label_encoder is None else label_encoder.classes_

    # per-class mean |SHAP| (one row per SHAP output of the model)
    mean_shap_values = analysis_results["mean_shap_values"]
    if class_names is not None:
        names = shap_class_names(len(mean_shap_values), class_names)
        for name, shap_class in zip(names, mean_shap_values):
            result_df.loc[f"shap_importance: class {name}"] = pd.Series(
                np.abs(shap_class).mean(axis=0), index=columns
            )

    if analysis_results["reduced_features"] is not None:
        reduced_features = analysis_results["reduced_features"]
        result_df.loc["reduced_shap_feature_importance", reduced_features] = analysis_results[
            "reduced_shap_feature_importance"
        ]
        mean_shap_values = analysis_results["reduced_mean_shap_values"]
        if class_names is not None:
            names = shap_class_names(len(mean_shap_values), class_names)
            for name, shap_class in zip(names, mean_shap_values):
                result_df.loc[f"reduced shap_importance: class {name}", reduced_features] = (
                    np.abs(shap_class).mean(axis=0)
                )

        result_df = result_df.sort_values(
            "reduced_shap_feature_importance", axis=1, ascending=False
        )
    else:
        result_df = result_df.sort_values("shap_feature_importance", axis=1, ascending=False)

    result_df.to_csv(os.path.join(folder, "importance_results.csv"))


def classify_pairwise(  # pylint: disable=too-many-locals
    features,
    features_info,
    model="XG",
    graph_removal=0.3,
    interpretability=1,
    n_top_features=5,
    reduce_set=False,
    reduced_set_size=100,
    reduced_set_max_correlation=0.5,
    n_repeats=1,
    n_splits=None,
    analysis_type="classification",
):
    """Classify all possible pairs of clases with kfold and returns top features.

    The top features for each pair with high enough accuracies are collected in a list,
    for later analysis.

    Args:
        features (dataframe): extracted features
        features_info (dataframe): features information
        model (str): model to preform analysis
        graph_removal (float): remove samples with more than graph_removal % bad values
        n_top_features (int): number of top features to save
        reduce_set (bool): is True, the classification will be rerun
            on a reduced set of top features (from shapely analysis)
        reduce_set_size (int): number of features to keep for reduces set
        reduced_set_max_correlation (float): to discared highly correlated top features
            in reduced set of features
        n_repeats (int): number of k-fold repeats
        n_splits (int): numbere of split for k-fold, None=automatic estimation
        analysis_type (str): 'classification' or 'regression'

    Returns:
        (dataframe, list, int): accuracies dataframe, list of top features, number of top pairs
    """
    features, features_info, _ = _preprocess_features(
        features, features_info, graph_removal, interpretability
    )

    classifier = _get_model(model, analysis_type=analysis_type)
    classes = features.label.unique()
    class_pairs = list(itertools.combinations(classes, 2))
    accuracy_matrix = pd.DataFrame(columns=classes, index=classes)

    top_features = {}
    for pair in tqdm(class_pairs):
        L.info("Pairwise classification between classes %s and %s", str(pair[0]), str(pair[1]))
        features_pair = features.loc[(features.label == pair[0]) | (features.label == pair[1])]
        le = LabelEncoder()
        features_pair.label = le.fit_transform(features_pair.label)
        analysis_results = fit_model_kfold(
            features_pair,
            classifier,
            analysis_type=analysis_type,
            reduce_set=reduce_set,
            reduced_set_size=reduced_set_size,
            reduced_set_max_correlation=reduced_set_max_correlation,
            n_repeats=n_repeats,
            n_splits=n_splits,
        )
        if "reduced_acc_scores" in analysis_results:
            accuracy_matrix.loc[pair[0], pair[1]] = np.round(
                np.mean(analysis_results["reduced_acc_scores"]), 3
            )
        else:
            accuracy_matrix.loc[pair[0], pair[1]] = np.round(
                np.mean(analysis_results["acc_scores"]), 3
            )
        accuracy_matrix.loc[pair[1], pair[0]] = accuracy_matrix.loc[pair[0], pair[1]]

        if "reduced_shap_feature_importance" in analysis_results:
            top_feat_idx = analysis_results["reduced_shap_feature_importance"].argsort()[
                -n_top_features:
            ]
            top_features_raw = analysis_results["X"][analysis_results["reduced_features"]].columns[
                top_feat_idx
            ]
        else:
            top_feat_idx = analysis_results["shap_feature_importance"].argsort()[-n_top_features:]
            top_features_raw = analysis_results["X"].columns[top_feat_idx]

        top_features[pair] = [f_class + "_" + f for f_class, f in top_features_raw]

    return accuracy_matrix.astype(float), top_features
