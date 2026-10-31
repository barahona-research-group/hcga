"""Tests for the fixes in hcga/analysis.py and hcga/plotting.py.

They use small synthetic feature matrices (no graph extraction) and small models.
"""

import os
import subprocess
import sys
import time
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from xgboost import XGBClassifier, XGBRegressor  # noqa: E402

from hcga import analysis as A  # noqa: E402
from hcga import plotting as P  # noqa: E402
from hcga.feature_class import InterpretabilityScore  # noqa: E402


def _xgb():
    return XGBClassifier(n_estimators=20, max_depth=3, random_state=0)


def _make_features(n_per_class=20, n_noise=10, labels=(0, 1, 2), seed=0, shift=3.0):
    """Feature frame with MultiIndex columns: feature 'informative' is shifted by class."""
    rng = np.random.default_rng(seed)
    y = np.repeat(np.array(labels, dtype=object), n_per_class)
    class_idx = np.repeat(np.arange(len(labels)), n_per_class)
    n = len(y)
    data = {("C", "informative"): class_idx * shift + rng.normal(size=n)}
    for i in range(n_noise):
        data[("N", f"noise_{i}")] = rng.normal(size=n)
    features = pd.DataFrame(data)
    features.columns = pd.MultiIndex.from_tuples(
        features.columns, names=["feature_class", "feature_name"]
    )
    features[("label", "")] = list(y)
    return features


def _make_info(features, score=5):
    cols = features.drop(columns=["label"]).columns
    info = pd.DataFrame(
        {
            col: [col[1], col[1], col[1], "desc", InterpretabilityScore(score)]
            for col in cols
        },
        index=["name", "shortname", "fullname", "description", "interpretability"],
    )
    info.columns = cols
    return info


# 1. SHAP importance axes


@pytest.mark.parametrize("model", [_xgb(), RandomForestClassifier(n_estimators=30, random_state=0)])
def test_shap_importance_multiclass_ranks_informative_first(model):
    features = _make_features()
    res = A.fit_model_kfold(features, model, reduce_set=False, n_splits=3)
    n_features = features.shape[1] - 1
    assert np.shape(res["shap_feature_importance"]) == (n_features,)
    assert res["X"].columns[np.argmax(res["shap_feature_importance"])] == ("C", "informative")
    assert len(res["mean_shap_values"]) == 3
    assert res["mean_shap_values"][0].shape == (len(features), n_features)


def test_shap_importance_binary_rf_not_zero():
    """Binary RF: per-class SHAP values are opposite, so abs must be taken before averaging."""
    features = _make_features(labels=(0, 1))
    model = RandomForestClassifier(n_estimators=30, random_state=0)
    res = A.fit_model_kfold(features, model, reduce_set=False, n_splits=3)
    imp = res["shap_feature_importance"]
    assert imp.shape == (features.shape[1] - 1,)
    assert imp.max() > 0.05
    assert res["X"].columns[np.argmax(imp)] == ("C", "informative")


def test_shap_feature_importance_formats_agree():
    rng = np.random.default_rng(1)
    sv = rng.normal(size=(15, 4, 3))
    as_list = [sv[:, :, c] for c in range(3)]
    mean_a, imp_a = A._get_shap_feature_importance([sv, sv])
    mean_b, imp_b = A._get_shap_feature_importance([as_list, as_list])
    expected = np.mean([np.abs(sv[:, :, c]).mean(axis=0) for c in range(3)], axis=0)
    assert np.allclose(imp_a, expected) and np.allclose(imp_b, expected)
    assert len(mean_a) == len(mean_b) == 3
    # regression / binary xgboost: one 2-D output per fold
    mean_c, imp_c = A._get_shap_feature_importance([sv[:, :, 0]])
    assert len(mean_c) == 1 and np.allclose(imp_c, np.abs(sv[:, :, 0]).mean(axis=0))


# 2. _save_to_csv


def test_save_to_csv_per_class_mean_abs_shap(tmp_path):
    features = _make_features(labels=("a", "b", "c"))
    info = _make_info(features)
    res = A.fit_model_kfold(features, _xgb(), reduced_set_size=3, n_splits=3)
    A._save_to_csv(info, res, tmp_path)
    df = pd.read_csv(tmp_path / "importance_results.csv", header=[0, 1], index_col=0)
    for i, name in enumerate(["a", "b", "c"]):
        row = df.loc[f"shap_importance: class {name}"].astype(float)
        expected = pd.Series(np.abs(res["mean_shap_values"][i]).mean(axis=0), index=res["X"].columns)
        assert np.allclose(row[expected.index].values, expected.values)
        assert f"reduced shap_importance: class {name}" in df.index
    assert "shap_importance: class 0" not in df.index


# 3. Label encoding in the main path


@pytest.mark.parametrize("labels", [(1, 2, 3), ("ER", "BA", "WS")])
def test_analysis_label_encoding(tmp_path, labels):
    features = _make_features(labels=labels)
    info = _make_info(features)
    res = A.analysis(
        features,
        info,
        folder=str(tmp_path),
        model=_xgb(),
        reduce_set=False,
        plot=False,
        n_splits=3,
        save_model=True,
    )
    assert np.mean(res["acc_scores"]) > 0.8
    assert list(res["label_encoder"].classes_) == sorted(labels)
    assert set(res["y"]) == {0, 1, 2}

    # the saved model predicts in the original labels
    trained = A.load_fitted_model(str(tmp_path / "fitted_model"))
    pred = A.analysis(
        features.drop(columns=["label"]), info, folder=str(tmp_path), trained_model=trained
    )
    assert np.mean(pred == features["label"].values) > 0.9
    assert set(pred) <= set(labels)


# 4. Plotting with seaborn 0.13 / pandas 3


def _assert_no_deprecation(record):
    bad = [
        str(w.message)
        for w in record
        if issubclass(w.category, (FutureWarning, DeprecationWarning, PendingDeprecationWarning))
        or "Pandas4Warning" in w.category.__name__
    ]
    bad = [m for m in bad if any(k in m for k in ("palette", "bw", "MultiIndex", "lexsort"))]
    assert not bad, bad


def test_plot_shap_violin_binary(tmp_path):
    features = _make_features(labels=(0, 1))
    X, y = A.features_to_Xy(features)
    imp = np.arange(X.shape[1], dtype=float)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        P._plot_shap_violin(imp, X, y, tmp_path, 4, label_names=["neg", "pos"])
    assert (tmp_path / "shap_violins.png").exists()
    _assert_no_deprecation(record)
    plt.close("all")


class _FakeGraph:
    def __init__(self, seed):
        self.graph = nx.erdos_renyi_graph(8, 0.4, seed=seed)

    def get_graph(self, _):
        return self.graph


class _FakeCollection:
    def __init__(self, n):
        self.graphs = [_FakeGraph(i) for i in range(n)]


@pytest.mark.parametrize("labels", [(0, 1, 2), (0, 1)])
def test_plot_analysis_classification(tmp_path, labels):
    features = _make_features(labels=labels, n_per_class=10, n_noise=5)
    res = A.fit_model_kfold(features, _xgb(), reduced_set_size=4, n_splits=3)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        P.plot_analysis(res, tmp_path, _FakeCollection(len(features)), "classification", 3, 6)
    _assert_no_deprecation(record)
    assert (tmp_path / "analysis_report.pdf").exists()
    assert (tmp_path / "shap_violins.svg").exists()
    n_outputs = 1 if len(labels) == 2 else len(labels)  # binary xgboost: one SHAP output
    assert len(list(tmp_path.glob("shap_class_*_summary.svg"))) == n_outputs
    plt.close("all")


# 5. Saved-model prediction reuses the stored scaler


def test_saved_model_prediction_independent_of_batch(tmp_path):
    features = _make_features(labels=(0, 1, 2), n_noise=5)
    info = _make_info(features)
    A.analysis(
        features,
        info,
        folder=str(tmp_path),
        model=_xgb(),
        compute_shap=False,
        plot=False,
        n_splits=3,
        save_model=True,
    )
    trained = A.load_fitted_model(str(tmp_path / "fitted_model"))
    assert trained[1].n_features_in_ == features.shape[1] - 1  # scaler fitted without label

    X = features.drop(columns=["label"])
    pred_all = A.analysis(X, info, folder=str(tmp_path), trained_model=trained)
    batch = (features["label"] == 2).values
    pred_batch = A.analysis(X[batch], info, folder=str(tmp_path), trained_model=trained)
    assert np.array_equal(pred_batch, pred_all[batch])
    assert np.mean(pred_batch == 2) > 0.9


# 6. Regression folds, fit_model(kfold=False), train_all, RF settings


def test_regression_kfold_uses_5_splits():
    rng = np.random.default_rng(0)
    features = _make_features(labels=(0, 1), n_noise=3)
    features[("label", "")] = features[("C", "informative")] + 0.1 * rng.normal(size=len(features))
    res = A.fit_model_kfold(
        features, XGBRegressor(n_estimators=20), analysis_type="regression", reduce_set=False
    )
    assert len(res["acc_scores"]) == 5
    assert len(res["mean_shap_values"]) == 1


def test_fit_model_no_kfold():
    features = _make_features(labels=(0, 1, 2), n_noise=5)
    # compute_shap=False must not crash
    res = A.fit_model(features, _xgb(), compute_shap=False)
    assert res["shap_feature_importance"] is None
    # the reduced set has one importance per reduced feature
    res = A.fit_model(features, _xgb(), reduced_set_size=3)
    assert np.shape(res["reduced_shap_feature_importance"]) == (len(res["reduced_features"]),)
    # stratified split: 4 test graphs per class
    _, test_index = res["indices"]
    assert np.bincount(res["y"].iloc[test_index]).tolist() == [4, 4, 4]


def test_save_model_does_not_refit_plotted_model(tmp_path):
    features = _make_features(labels=(0, 1, 2), n_noise=5, shift=1.0)
    info = _make_info(features)
    model = _xgb()
    res = A.analysis(
        features,
        info,
        folder=str(tmp_path),
        model=model,
        kfold=False,
        reduce_set=False,
        save_model=True,
        plot=False,
    )
    train_index, _ = res["indices"]
    reference = clone(model).fit(res["X"].iloc[train_index], res["y"].iloc[train_index])
    assert np.allclose(res["model"].predict_proba(res["X"]), reference.predict_proba(res["X"]))


def test_rf_seeded_and_parallel():
    for analysis_type in ["classification", "regression"]:
        model = A._get_model("RF", analysis_type, random_state=3)
        assert model.random_state == 3 and model.n_jobs == -1


# 7. Reduced-set selection nested in the training folds


def test_nested_reduced_set_unbiased_on_noise():
    nested, old = [], []
    for seed in range(3):
        rng = np.random.default_rng(seed)
        X = pd.DataFrame(rng.normal(size=(60, 500)))
        X.columns = pd.MultiIndex.from_tuples([("N", f"f{i}") for i in range(500)])
        features = X.copy()
        features[("label", "")] = rng.integers(0, 2, 60)
        model = XGBClassifier(n_estimators=30, max_depth=3, random_state=0)
        for nested_selection, out in [(True, nested), (False, old)]:
            res = A.fit_model_kfold(
                features,
                model,
                reduced_set_size=10,
                n_splits=5,
                nested_selection=nested_selection,
            )
            out.append(np.mean(res["reduced_acc_scores"]))
    assert abs(np.mean(nested) - 0.5) < 0.15
    assert np.mean(old) > 0.6  # the all-data selection is optimistically biased


# 8. Interpretability filter and import side effects


def _filter_interpretable_reference(features, features_info, interpretability):
    interpretability = min(5, interpretability)
    for feat in list(features_info.keys()):
        if features_info[feat]["interpretability"].score < interpretability:
            if feat in features.columns:
                features = features.drop(columns=[feat])
                features_info = features_info.drop(columns=[feat])
    return features, features_info


def test_filter_interpretable_vectorised():
    rng = np.random.default_rng(0)
    n_feats = 3000
    cols = pd.MultiIndex.from_tuples([("C", f"f{i}") for i in range(n_feats)])
    features = pd.DataFrame(rng.normal(size=(30, n_feats)), columns=cols)
    scores = rng.integers(1, 6, n_feats)
    info = pd.DataFrame(
        [[f"f{i}" for i in range(n_feats)], [InterpretabilityScore(int(s)) for s in scores]],
        index=["name", "interpretability"],
        columns=cols,
    )
    # one info column not in the features: kept in features_info, as before
    info[("C", "extra")] = ["extra", InterpretabilityScore(1)]

    small = features.iloc[:, :200]
    ref_f, ref_i = _filter_interpretable_reference(small, info, 3)
    new_f, new_i = A._filter_interpretable(small, info, 3)
    assert new_f.columns.equals(ref_f.columns) and new_i.columns.equals(ref_i.columns)

    start = time.time()
    new_f, _ = A._filter_interpretable(features, info, 3)
    assert time.time() - start < 10.0  # the old loop took > 5 s here
    assert new_f.shape[1] == int((scores >= 3).sum())


def test_import_does_not_change_global_warnings_or_logging():
    code = (
        "import logging, warnings\n"
        "import hcga.analysis\n"
        "catch_all = [f for f in warnings.filters if f[0] == 'ignore' and f[2] is Warning"
        " and f[1] is None and f[3] is None]\n"
        "print(len(catch_all), len(logging.getLogger().handlers))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "MPLBACKEND": "Agg"},
    )
    assert out.stdout.split()[-2:] == ["0", "0"]

