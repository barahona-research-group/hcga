"""Tests for the core fixes: timeouts, evaluate_feature, summary statistics, extraction, graph."""

import multiprocessing
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import scipy.stats as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixtures import fixture_graph  # noqa: E402

import hcga.extraction as extraction  # noqa: E402
from hcga.feature_class import FeatureClass, _hmean, _mean_repeats  # noqa: E402
from hcga.graph import GraphCollection, convert_graph  # noqa: E402
from hcga.io import load_analysis, load_fitted_model, save_fitted_model  # noqa: E402
from hcga.utils import FeatureTimeout, timeout_eval  # noqa: E402


def _busy(seconds):
    """Pure-Python loop (interruptible by SIGALRM)."""
    end = time.time() + seconds
    count = 0
    while time.time() < end:
        count += 1
    return count


@nx.utils.not_implemented_for("directed")
def _undirected_only(graph):
    return float(graph.number_of_edges())


class _Probe(FeatureClass):
    """Minimal feature class for testing evaluate_feature and the statistics."""

    shortname = "PRB"
    name = "probe"
    encoding = "networkx"
    timeout = None
    normalize_features = False
    statistics_level = "basic"


def _probe(graph=None, **attrs):
    inst = _Probe(graph if graph is not None else fixture_graph("karate"))
    inst.all_features = False
    for key, value in attrs.items():
        setattr(inst, key, value)
    return inst


# --- 1. timeout_eval ------------------------------------------------------------------------


@pytest.mark.parametrize("timeout", [None, 0, 5])
def test_timeout_eval_propagates_exceptions(timeout):
    with pytest.raises(ZeroDivisionError):
        timeout_eval(lambda x: x / 0, (1,), timeout=timeout)


def test_timeout_eval_returns_value_and_large_results():
    big = timeout_eval(lambda n: list(range(n)), (100_000,), timeout=5)
    assert len(big) == 100_000


def test_timeout_eval_raises_timeout():
    start = time.time()
    with pytest.raises(FeatureTimeout) as info:
        timeout_eval(_busy, (5,), timeout=0.2)
    assert time.time() - start < 5
    assert isinstance(info.value, multiprocessing.TimeoutError)
    assert isinstance(info.value, TimeoutError)
    # timer cancelled and handler restored
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
    assert signal.getsignal(signal.SIGALRM) in (signal.SIG_DFL, signal.SIG_IGN, None)


def test_timeout_eval_no_limit_outside_main_thread():
    result = {}

    def run():
        result["value"] = timeout_eval(_busy, (0.3,), timeout=0.05)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    assert result["value"] > 0


# --- 2. evaluate_feature ----------------------------------------------------------------------


def test_directed_fallback_to_undirected():
    digraph = nx.DiGraph([(0, 1), (1, 0), (1, 2), (2, 3), (3, 0)])
    inst = _probe(convert_graph(digraph, label=0))
    assert inst.graph.is_directed()
    assert inst.evaluate_feature(_undirected_only, "undirected_only") == 4.0
    assert not inst.failures


def test_not_implemented_on_undirected_is_recorded():
    @nx.utils.not_implemented_for("undirected")
    def directed_only(graph):
        return 1.0

    inst = _probe()
    assert inst.evaluate_feature(directed_only, "directed_only") is None
    assert "NetworkXNotImplemented" in inst.failures["directed_only"]


def test_failures_recorded():
    inst = _probe()

    def boom(graph):
        raise ValueError("boom")

    assert inst.evaluate_feature(boom, "boom") is None
    assert inst.failures["boom"] == repr(ValueError("boom"))


def test_timeout_recorded():
    inst = _probe(timeout=0.2)
    assert inst.evaluate_feature(lambda graph: _busy(5), "slow") is None
    assert "FeatureTimeout" in inst.failures["slow"]


def test_keyboard_interrupt_propagates():
    def interrupt(graph):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _probe().evaluate_feature(interrupt, "interrupt")


@pytest.mark.parametrize(
    "value, expected",
    [
        (np.bool_(True), 1.0),
        (True, 1.0),
        (np.float16(0.5), 0.5),
        (np.int64(3), 3.0),
        (3, 3.0),
        (np.float32(0.25), 0.25),
    ],
)
def test_scalar_types_accepted(value, expected):
    result = _probe().evaluate_feature(lambda graph: value, "scalar")
    assert isinstance(result, float)
    assert result == expected


@pytest.mark.parametrize(
    "value, statistics",
    [
        (np.complex128(1 + 2j), None),
        ("a string", None),
        ([1, 2], None),
        ([], "clustering"),
        ([1, 2], "clustering"),
        (3.0, "centrality"),
    ],
)
def test_unexpected_types_give_none(value, statistics):
    inst = _probe()
    assert inst.evaluate_feature(lambda graph: value, "odd", statistics=statistics) is None
    assert "TypeError" in inst.failures["odd"]


def test_unexpected_type_does_not_abort_class():
    inst = _probe()
    inst.add_feature("bad", lambda graph: np.complex128(1j), "bad", 1)
    inst.add_feature("good", lambda graph: 2, "good", 1)
    assert inst.features == {"good": 2.0}


# --- 3. summary statistics --------------------------------------------------------------------


def test_advanced_statistics_tmean_tvar():
    dist = [0.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 10.0]
    inst = _probe(statistics_level="advanced")
    inst.add_feature("d", lambda graph: dist, "test distribution", 3, statistics="centrality")
    feats = inst.features
    assert feats["d_tmean"] == pytest.approx(st.tmean(dist))
    assert feats["d_tvar"] == pytest.approx(st.tvar(dist))
    assert feats["d_mean_repeats"] == pytest.approx(2.5)  # value 2 twice, value 3 three times
    assert feats["d_gmean"] == 0.0  # contains 0
    assert feats["d_hmean"] == 0.0
    assert _Probe.feature_descriptions["d_hmean"]["desc"].startswith("H. Mean")
    assert _Probe.feature_descriptions["d_tvar"]["desc"].startswith("Tvariance")


def test_mean_repeats():
    assert _mean_repeats([1, 2, 2, 3, 3, 3]) == pytest.approx(2.5)
    assert np.isnan(_mean_repeats([1, 2, 3]))


def test_gmean_hmean_positive_only():
    inst = _probe(statistics_level="medium")
    inst.add_feature("p", lambda graph: [1.0, 2.0, 4.0], "positive", 3, statistics="centrality")
    assert inst.features["p_gmean"] == pytest.approx(2.0)
    assert inst.features["p_hmean"] == pytest.approx(3 / (1 + 0.5 + 0.25))
    assert _hmean([0.0, 1.0, 2.0]) == 0.0
    assert np.isnan(_hmean([-1.0, 1.0, 2.0]))


# --- 4. pool worker settings ------------------------------------------------------------------


def _small_collection():
    collection = GraphCollection()
    collection.add_graph(nx.karate_club_graph(), label=0)
    collection.add_graph(nx.gnp_random_graph(20, 0.3, seed=1), label=1)
    return collection


def test_init_worker_restores_settings():
    classes, _ = extraction.get_list_feature_classes(
        mode="fast", normalize_features=False, statistics_level="advanced", timeout=7
    )
    settings = extraction._class_settings(classes)
    for cls in classes:
        cls.normalize_features, cls.statistics_level, cls.timeout = True, "basic", None
    extraction._init_worker(settings)
    for cls in classes:
        assert cls.normalize_features is False
        assert cls.statistics_level == "advanced"
        assert cls.timeout == 7


def test_extract_settings_apply_in_pool_workers():
    """With 2 workers (spawn on macOS), the settings reach the workers."""
    features, info = extraction.extract(
        _small_collection(),
        n_workers=2,
        mode="fast",
        normalize_features=False,
        statistics_level="advanced",
        timeout=5,
    )
    names = features.columns.get_level_values(1)
    assert any(name.endswith("_kstat") for name in names)
    assert not any(name.endswith(("_N", "_E")) for name in names)
    assert set(features.columns) - {("label", "")} <= set(info.columns)


def test_single_worker_runs_in_process(monkeypatch):
    def no_pool(*args, **kwargs):
        raise AssertionError("a pool was created for n_workers=1")

    monkeypatch.setattr(extraction, "NestedPool", no_pool)
    classes, _ = extraction.get_list_feature_classes(
        mode="fast", normalize_features=False, timeout=0
    )
    frame = extraction.compute_all_features(_small_collection(), classes[:3], n_workers=1)
    assert list(frame.index) == [0, 1]


def test_check_features_info_warns(caplog):
    columns = pd.MultiIndex.from_tuples([("A", "x"), ("A", "y"), ("label", "")])
    features = pd.DataFrame([[1.0, 2.0, 0]], columns=columns)
    info = pd.DataFrame({("A", "x"): ["x"]})
    extraction._check_features_info(features, info)
    assert "1 feature columns are missing" in caplog.text


# --- 5. frame building ------------------------------------------------------------------------


def _fake_class(shortname, features):
    return type(
        f"Fake{shortname}",
        (FeatureClass,),
        {
            "shortname": shortname,
            "name": shortname,
            "get_features": lambda self, all_features=False: dict(features),
        },
    )


def _old_frame(graph, classes):
    """The previous column-by-column frame building."""
    columns = pd.MultiIndex(levels=[[], []], codes=[[], []], names=["feature_class", "feature_name"])
    frame = pd.DataFrame(columns=columns)
    for cls in classes:
        inst = cls(graph)
        feats = pd.DataFrame(inst.get_features(), index=[graph.id])
        frame[[(inst.shortname, col) for col in feats.columns]] = feats
    return frame


def test_frame_building_unchanged():
    graph = fixture_graph("karate")
    classes = [
        _fake_class("F1", {"a": 1.0, "b": np.float64(2.5), "c": np.nan}),
        _fake_class("F2", {"a": 3.0, "d": None, "e": 4}),
    ]
    new = extraction.feature_extraction(graph, classes)
    pd.testing.assert_frame_equal(new, _old_frame(graph, classes))
    assert list(new.columns.names) == ["feature_class", "feature_name"]


def test_features_info_unchanged():
    classes, info = extraction.get_list_feature_classes(
        mode="fast", normalize_features=True, statistics_level="medium", timeout=0
    )
    columns = pd.MultiIndex(levels=[[], []], codes=[[], []], names=["feature_class", "feature_name"])
    old_info = pd.DataFrame(columns=columns)
    for cls in classes:
        inst = cls()
        class_info = pd.DataFrame()
        for feature in cls.setup_class(
            normalize_features=True, statistics_level="medium", timeout=0
        ).columns:
            class_info[feature] = pd.Series(inst.get_feature_info(feature))
        old_info[[(cls.shortname, col) for col in class_info.columns]] = class_info
    pd.testing.assert_frame_equal(info, old_info)


# --- 6. labels, iteration, node features, cache --------------------------------------------------


@pytest.mark.parametrize("labels", [["a", "b"], [3, 7]])
def test_graph_labels_survive(labels):
    collection = GraphCollection()
    for label in labels:
        collection.add_graph(nx.karate_club_graph(), label=label)
    assert [graph.label for graph in collection.graphs] == labels
    columns = pd.MultiIndex.from_tuples([("A", "x")], names=["feature_class", "feature_name"])
    features = pd.DataFrame([[1.0], [2.0]], index=[0, 1], columns=columns)
    extraction._set_graph_labels(features, collection)
    assert features["label"].tolist() == labels
    if isinstance(labels[0], int):
        assert pd.api.types.is_integer_dtype(features["label"])


def test_iteration_is_reentrant():
    collection = GraphCollection()
    for seed in range(3):
        collection.add_graph(nx.gnp_random_graph(10, 0.5, seed=seed))
    pairs = [(a.id, b.id) for a in collection for b in collection]
    assert len(pairs) == 9


def test_remove_node_features():
    graph = nx.path_graph(5)
    for node in graph:
        graph.nodes[node]["features"] = [float(node), 1.0]
    collection = GraphCollection()
    collection.add_graph(graph)
    hcga_graph = collection.graphs[0]
    assert hcga_graph.n_node_features == 2
    hcga_graph.get_graph("networkx")  # build the cache
    collection.remove_node_features()
    assert hcga_graph.n_node_features == 0
    assert not {"features", "attributes", "labels"} & set(hcga_graph.nodes.columns)
    nx_graph = hcga_graph.get_graph("networkx")
    assert all(data["feat"] == [0] for _, data in nx_graph.nodes(data=True))
    assert collection.get_n_node_features() == 0


def test_maximal_subgraph_invalidates_cache():
    graph = nx.disjoint_union(nx.complete_graph(5), nx.path_graph(3))
    hcga_graph = convert_graph(graph)
    assert hcga_graph.get_graph("networkx").number_of_nodes() == 8
    hcga_graph.maximal_subgraph()
    assert hcga_graph.get_graph("networkx").number_of_nodes() == 5
    assert hcga_graph.get_graph("networkx").number_of_edges() == 10


def test_remove_weights_invalidates_cache():
    graph = nx.Graph()
    graph.add_weighted_edges_from([(0, 1, 3.0), (1, 2, 5.0), (2, 0, 2.0)])
    hcga_graph = convert_graph(graph)
    assert hcga_graph.get_graph("networkx").size(weight="weight") == 10.0
    hcga_graph.remove_weights()
    assert hcga_graph.get_graph("networkx").size(weight="weight") == 3.0


# --- 7. convert_graph (issue #110) ------------------------------------------------------------


def test_convert_graph_self_loop_weight():
    graph = nx.Graph()
    graph.add_weighted_edges_from([(0, 1, 2.0), (1, 2, 3.0), (2, 2, 5.0), (2, 3, 1.0)])
    hcga_graph = convert_graph(graph)
    assert len(hcga_graph.edges) == 4  # each undirected edge stored once
    nx_graph = hcga_graph.get_graph("networkx")
    assert nx_graph[2][2]["weight"] == 5.0
    assert sorted(w for _, _, w in nx_graph.edges(data="weight")) == [1.0, 2.0, 3.0, 5.0]


def test_convert_graph_digraph_reciprocal_weights():
    graph = nx.DiGraph()
    graph.add_weighted_edges_from([(0, 1, 1.0), (1, 0, 7.0), (1, 2, 2.0), (2, 0, 4.0)])
    hcga_graph = convert_graph(graph)  # graph_type inferred
    assert hcga_graph.graph_type == "directed"
    nx_graph = hcga_graph.get_graph("networkx")
    assert nx_graph.is_directed()
    assert nx_graph[0][1]["weight"] == 1.0
    assert nx_graph[1][0]["weight"] == 7.0
    assert nx_graph.number_of_edges() == 4


def test_convert_graph_non_integer_node_labels():
    graph = nx.Graph()
    graph.add_weighted_edges_from([("a", "b", 2.0), ("b", "c", 3.0), ("c", "c", 1.0)])
    nx_graph = convert_graph(graph).get_graph("networkx")
    assert nx_graph[0][1]["weight"] == 2.0
    assert nx_graph[2][2]["weight"] == 1.0


# --- 8. io, app, warnings -----------------------------------------------------------------------


def test_fitted_model_paths(tmp_path):
    save_fitted_model("model", "scaler", "info", folder=tmp_path)
    assert (tmp_path / "model.pkl").exists()
    assert load_fitted_model(tmp_path) == ["model", "scaler", "info"]
    assert load_fitted_model(filename=str(tmp_path / "model.pkl")) == ["model", "scaler", "info"]


def test_load_analysis_default_folder(tmp_path, monkeypatch):
    import pickle  # pylint: disable=import-outside-toplevel

    monkeypatch.chdir(tmp_path)
    with open("analysis_results.pkl", "wb") as f:
        pickle.dump([1, 2, 3], f)
    assert load_analysis() == [1, 2, 3]


def test_app_options(monkeypatch):
    from click.testing import CliRunner  # pylint: disable=import-outside-toplevel

    import hcga.io  # pylint: disable=import-outside-toplevel
    from hcga.app import cli, feature_analysis  # pylint: disable=import-outside-toplevel

    n_splits = next(p for p in feature_analysis.params if p.name == "n_splits")
    assert n_splits.type.name == "integer"

    called = {}

    def fake_extract(graphs, n_workers, **kwargs):
        called["n_workers"] = n_workers
        return None, None

    monkeypatch.setattr(extraction, "extract", fake_extract)
    monkeypatch.setattr(hcga.io, "load_dataset", lambda filename: None)
    result = CliRunner().invoke(
        cli, ["extract_features", "dummy.pkl", "-n", "-1", "--runtimes"]
    )
    assert result.exit_code == 0, result.output
    assert called["n_workers"] == os.cpu_count()


def test_import_does_not_silence_warnings():
    """Importing hcga must not install a blanket "ignore" filter (libraries add targeted ones)."""
    code = (
        "import warnings; before = list(warnings.filters); import hcga.feature_class; "
        "added = [f for f in warnings.filters if f not in before]; "
        "print(any(f[0] == 'ignore' and f[1] is None and f[2] is Warning for f in added))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    assert out.strip() == "False"
