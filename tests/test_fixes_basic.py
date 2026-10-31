"""Tests for the accuracy fixes in the basic feature classes and the dataset generators.

Each test compares a feature with a closed-form value or a direct networkx computation on a
small graph, or checks determinism of a generator.
"""

import sys
import warnings
import zipfile
from math import comb
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixtures import fixture_graph, fixture_specs  # noqa: E402

from hcga.extraction import get_list_feature_classes  # noqa: E402
from hcga.features import assortativity, node_features, rich_club  # noqa: E402
from hcga.features.nestedness import nestedness  # noqa: E402
from hcga.graph import GraphCollection  # noqa: E402

approx = pytest.approx


@pytest.fixture(scope="module")
def classes():
    class_list, _ = get_list_feature_classes(
        mode="all", normalize_features=False, statistics_level="basic", timeout=0
    )
    return {cls.__name__: cls for cls in class_list}


def _features(classes, class_name, graph):
    return classes[class_name](graph).get_features()


def _nx_fixture(name):
    return dict((n, g) for n, g, _ in fixture_specs())[name]


def _graph_from_nx(graph, graph_type=None):
    collection = GraphCollection()
    collection.add_graph(graph, label=0, graph_type=graph_type)
    return collection.graphs[0]


def _lcc(graph):
    graph = graph.to_undirected() if graph.is_directed() else graph
    return graph.subgraph(max(nx.connected_components(graph), key=len))


# --- 1. ScaleFree ----------------------------------------------------------------------------


def test_scale_free_s_metric(classes):
    graph = _nx_fixture("karate")
    expected = sum(graph.degree(u) * graph.degree(v) for u, v in graph.edges)
    feats = _features(classes, "ScaleFree", fixture_graph("karate"))
    assert feats["s_metric"] == approx(expected)


# --- 2. BasicStats ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["karate", "er40", "dir_er30", "disconnected", "complete8"])
def test_basic_stats_density(classes, name):
    feats = _features(classes, "BasicStats", fixture_graph(name))
    assert feats["density"] == approx(nx.density(_nx_fixture(name)))


@pytest.mark.parametrize("name", ["karate", "dir_er30", "disconnected", "path7"])
def test_basic_stats_diameter_radius(classes, name):
    """Diameter and radius of the (undirected) largest connected component, also for digraphs."""
    lcc = _lcc(_nx_fixture(name))
    feats = _features(classes, "BasicStats", fixture_graph(name))
    assert feats["diameter"] == nx.diameter(lcc)
    assert feats["radius"] == nx.radius(lcc)


# --- 3. Clustering ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["karate", "complete8", "er40", "dir_er30", "ring12"])
def test_clustering_num_triangles_is_total_count(classes, name):
    graph = _nx_fixture(name).to_undirected()
    expected = sum(nx.triangles(graph).values()) // 3
    feats = _features(classes, "Clustering", fixture_graph(name))
    assert feats["num_triangles"] == expected


def test_clustering_num_triangles_closed_form(classes):
    assert _features(classes, "Clustering", fixture_graph("complete8"))["num_triangles"] == comb(
        8, 3
    )
    assert _features(classes, "Clustering", fixture_graph("karate"))["num_triangles"] == 45


def test_clustering_directed_uses_undirected_graph(classes):
    undirected = _nx_fixture("dir_er30").to_undirected()
    feats = _features(classes, "Clustering", fixture_graph("dir_er30"))
    assert feats["transitivity"] == approx(nx.transitivity(undirected))
    square = list(nx.square_clustering(undirected).values())
    assert feats["square_clustering_mean"] == approx(np.mean(square))


# --- 4. Eulerian -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "graph",
    [nx.cycle_graph(5), nx.path_graph(5), nx.star_graph(4)],
    ids=["C5", "P5", "star4"],
)
def test_eulerian_features(classes, graph):
    feats = _features(classes, "Eulerian", _graph_from_nx(graph))
    assert feats["eulerian"] == nx.is_eulerian(graph)
    assert feats["semi_eulerian"] == nx.is_semieulerian(graph)
    assert feats["has_eulerian_path"] == nx.has_eulerian_path(graph)


# --- 5. InOutDegrees -------------------------------------------------------------------------


def _digraph_with_sink_and_isolated():
    graph = nx.DiGraph([(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (3, 4), (2, 4), (4, 0)])
    graph.add_edges_from([(1, 5), (2, 5)])  # node 5 is a sink
    graph.add_node(6)  # isolated
    return graph


def test_in_out_degrees_sinks_and_isolated_nodes(classes):
    graph = _digraph_with_sink_and_isolated()
    feats = _features(classes, "InOutDegrees", _graph_from_nx(graph, "directed"))
    k_in = np.array([graph.in_degree(n) for n in graph])
    k_out = np.array([graph.out_degree(n) for n in graph])
    k = k_in + k_out
    has_out = k_out > 0
    assert feats["in_out_degree_mean"] == approx(np.mean(k_in[has_out] / k_out[has_out]))
    assert feats["in_out_degree_max"] == approx(np.max(k_in[has_out] / k_out[has_out]))
    assert feats["in_degree_normed_mean"] == approx(np.mean(k_in[k > 0] / k[k > 0]))
    assert feats["out_degree_normed_mean"] == approx(np.mean(k_out[k > 0] / k[k > 0]))
    assert feats["in_degree_mean"] == approx(np.mean(k_in))


def test_in_out_degrees_none_on_undirected(classes):
    assert _features(classes, "InOutDegrees", fixture_graph("karate")) == {}


# --- 6. ChemicalTheory -----------------------------------------------------------------------


def test_wiener_on_largest_component(classes):
    feats = _features(classes, "ChemicalTheory", fixture_graph("disconnected"))
    lcc = _lcc(_nx_fixture("disconnected"))
    assert np.isfinite(feats["wiener index"])
    assert feats["wiener index"] == nx.wiener_index(lcc)


def test_wiener_directed_uses_undirected_graph(classes):
    feats = _features(classes, "ChemicalTheory", fixture_graph("dir_er30"))
    assert feats["wiener index"] == nx.wiener_index(_lcc(_nx_fixture("dir_er30")))


@pytest.mark.parametrize("name", ["karate", "er40", "ring12", "dir_er30"])
def test_estrada_matches_networkx(classes, name):
    graph = _nx_fixture(name).to_undirected()
    expected = nx.estrada_index(graph)
    feats = _features(classes, "ChemicalTheory", fixture_graph(name))
    assert feats["estrada_index"] == approx(expected, rel=1e-9)
    assert feats["log_estrada_index"] == approx(np.log(expected), rel=1e-12)


def test_estrada_overflow_gives_inf_and_finite_log():
    """K_n has eigenvalues n-1 (once) and -1 (n-1 times): log EE = n-1 + log(1 + (n-1)e^-n).

    exp overflows above ~709.8, so n = 750 overflows the Estrada index itself. (The functions are
    called directly: the class would also compute the Wiener index, which is slow on K_750.)
    """
    from hcga.features.chemical_theory import estrada_index, log_estrada_index

    n = 750
    graph = nx.complete_graph(n)
    assert log_estrada_index(graph) == approx((n - 1) + np.log1p((n - 1) * np.exp(-n)))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert estrada_index(graph) == np.inf


# --- 7. Nestedness ---------------------------------------------------------------------------


def _nestedness_loop(graph):
    """The original O(N^2) implementation, as the reference."""
    neighbors = [set(graph.neighbors(node)) for node in graph.nodes]
    sum_n_ij = sum_n_m = 0
    for j in range(1, len(neighbors)):
        for i in range(j):
            sum_n_ij += len(neighbors[i] & neighbors[j])
            sum_n_m += min(len(neighbors[i]), len(neighbors[j]))
    return 0 if sum_n_m == 0 else sum_n_ij / sum_n_m


@pytest.mark.parametrize("name", [name for name, _, _ in fixture_specs()])
def test_nestedness_matches_loop(name):
    graph = fixture_graph(name).get_graph("networkx")
    assert nestedness(graph) == approx(_nestedness_loop(graph), abs=1e-12)


def test_nestedness_with_selfloop_and_empty():
    graph = nx.path_graph(5)
    graph.add_edge(2, 2)
    assert nestedness(graph) == approx(_nestedness_loop(graph), abs=1e-12)
    assert nestedness(nx.empty_graph(4)) == 0


# --- 8. RichClub, NodeFeatures, Assortativity ------------------------------------------------


def test_rich_club_directed_uses_undirected_graph(classes):
    graph = _nx_fixture("dir_er30").to_undirected()
    expected = list(nx.rich_club_coefficient(graph, normalized=False).values())
    # explicit conversion in the feature function, not reliant on the base-class fallback
    assert rich_club.eval_rich_club(_nx_fixture("dir_er30")) == approx(expected)
    feats = _features(classes, "RichClub", fixture_graph("dir_er30"))
    assert feats["rich_club_k_1"] == approx(expected[0])
    assert feats["rich_club_mean"] == approx(np.mean(expected))


def test_lru_caches_hold_one_graph():
    assert rich_club.eval_rich_club.cache_info().maxsize == 1
    assert node_features.get_feature_matrix.cache_info().maxsize == 1
    assert node_features.get_conv_matrix.cache_info().maxsize == 1


def _node_feature_classes():
    class_list, _ = get_list_feature_classes(
        mode="all", normalize_features=False, statistics_level="basic", timeout=0, n_node_features=2
    )
    return {cls.__name__: cls for cls in class_list}


def test_node_features_sparse_conv_matches_dense():
    rng = np.random.default_rng(0)
    graph = nx.gnp_random_graph(30, 0.2, seed=0)
    feat = rng.normal(size=(30, 2))
    collection = GraphCollection()
    collection.add_graph(graph, node_features=feat, label=0)
    hcga_graph = collection.graphs[0]

    nx_graph = hcga_graph.get_graph("networkx")
    conv = nx.to_numpy_array(nx_graph) + np.eye(30)
    feats = _node_feature_classes()["NodeFeatures"](hcga_graph).get_features()
    try:
        expected_values = {
            "conv_node_feature": conv @ feat,
            "conv2_node_feature": conv @ conv @ feat,
        }
        for name, expected in expected_values.items():
            for col in range(2):
                assert feats[f"{name}{col}_mean"] == approx(expected[:, col].mean(), rel=1e-12)
                assert feats[f"{name}{col}_max"] == approx(expected[:, col].max(), rel=1e-12)
    finally:
        # restore the class set-up used by the other tests in this module
        get_list_feature_classes(
            mode="all", normalize_features=False, statistics_level="basic", timeout=0
        )


def test_node_features_skipped_without_node_features(classes):
    assert _features(classes, "NodeFeatures", fixture_graph("karate")) == {}


@pytest.mark.parametrize(
    "graph", [nx.cycle_graph(3), nx.cycle_graph(12), nx.complete_graph(5)], ids=["C3", "C12", "K5"]
)
def test_assortativity_regular_graph_nan_without_raising(classes, graph):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        feats = _features(classes, "Assortativity", _graph_from_nx(graph))
    for name in [
        "degree_assortativity_coeff",
        "degree_assortativity_coeff_weighted",
        "degree_assortativity_coeff_pearson",
        "degree_assortativity_coeff_pearson_weighted",
    ]:
        assert np.isnan(feats[name])
    assert feats["degree assortativity_mean"] == approx(graph.degree(0))
    # also directly (the base class may silence some warnings itself)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.isnan(assortativity.degree_assortativity_coefficient(graph))
        assert np.isnan(assortativity.degree_pearson_correlation_coefficient(graph))


def test_assortativity_average_neighbour_degree_description(classes):
    _features(classes, "Assortativity", fixture_graph("karate"))
    desc = classes["Assortativity"].feature_descriptions["degree assortativity_mean"]["desc"]
    assert "average neighbor degree" in desc.lower()


# --- 9. Reciprocity --------------------------------------------------------------------------


def test_reciprocity(classes):
    assert _features(classes, "Reciprocity", fixture_graph("karate")) == {}
    feats = _features(classes, "Reciprocity", fixture_graph("dir_er30"))
    assert feats["reciprocity"] == approx(nx.overall_reciprocity(_nx_fixture("dir_er30")))


# --- 10. Dataset generators ------------------------------------------------------------------


def _edge_lists(collection):
    return [
        (graph.label, graph.edges[["start_node", "end_node"]].to_numpy().tolist())
        for graph in collection.graphs
    ]


def test_make_synthetic_sbm_reproducible():
    from hcga.dataset_creation import make_synthetic

    first = make_synthetic(write_to_file=False)
    second = make_synthetic(write_to_file=False)
    assert len(first.graphs) == 40
    assert sorted({int(np.ravel(graph.label)[0]) for graph in first.graphs}) == [1, 2, 3, 4]
    assert _edge_lists(first) == _edge_lists(second)
    assert _edge_lists(first) != _edge_lists(make_synthetic(write_to_file=False, seed=1))


def test_deprecated_synthetic_generators_reproducible():
    from hcga.dataset_creation import synthetic_datasets as sd

    for generator in [
        sd.synthetic_data_watts_strogatz,
        sd.synthetic_data_powerlaw_cluster,
        sd.synthetic_data_sbm,
    ]:
        first, labels_first = generator(4)
        second, labels_second = generator(4)
        assert [sorted(g.edges) for g in first] == [sorted(g.edges) for g in second]
        assert np.array_equal(labels_first, labels_second)


def test_test_dataset_directed_graphs():
    from hcga.dataset_creation import make_test_dataset

    collection = make_test_dataset(write_to_file=False, n_graphs=1)
    types = [graph.graph_type for graph in collection.graphs]
    assert types.count("directed") == 2
    assert [g.label for g in collection.graphs] == [
        g.label for g in make_test_dataset(write_to_file=False, n_graphs=1).graphs
    ]


TOY = {
    "TOY_A.txt": "1, 2\n2, 1\n2, 3\n3, 2\n1, 3\n3, 1\n4, 5\n5, 4\n5, 6\n6, 5\n6, 7\n7, 6\n",
    "TOY_graph_indicator.txt": "1\n1\n1\n2\n2\n2\n2\n",
    "TOY_graph_labels.txt": "0\n1\n",
    "TOY_node_labels.txt": "0\n1\n0\n2\n2\n1\n0\n",
    "TOY_node_attributes.txt": "0.1,1.0\n0.2,2.0\n0.3,3.0\n0.4,4.0\n0.5,5.0\n0.6,6.0\n0.7,7.0\n",
}


def _write_toy(folder):
    folder.mkdir(parents=True, exist_ok=True)
    for name, text in TOY.items():
        (folder / name).write_text(text)


def _check_toy(collection):
    assert [graph.label for graph in collection.graphs] == [0, 1]
    assert [len(graph.nodes) for graph in collection.graphs] == [3, 4]
    assert [len(graph.edges) for graph in collection.graphs] == [6, 6]
    # 3 one-hot node labels + 2 attributes
    assert [graph.n_node_features for graph in collection.graphs] == [5, 5]
    assert list(collection.graphs[1].nodes.features.iloc[0]) == [0, 0, 1, 0.4, 4.0]


def test_benchmark_parser_on_toy_tu_dataset(tmp_path):
    from hcga.dataset_creation.benchmark_datasets import extract_benchmark_graphs

    _write_toy(tmp_path / "TOY")
    _check_toy(extract_benchmark_graphs(tmp_path / "TOY", "TOY"))


def test_benchmark_make_downloads_from_tu_url(tmp_path, monkeypatch):
    """make() fetches <TU_URL>/<name>.zip with urllib, then parses and saves the dataset."""
    import urllib.request

    from hcga.dataset_creation import benchmark_datasets
    from hcga.io import load_dataset

    _write_toy(tmp_path / "src" / "TOY")
    archive = tmp_path / "src" / "TOY.zip"
    with zipfile.ZipFile(archive, "w") as zip_file:
        for name in TOY:
            zip_file.write(tmp_path / "src" / "TOY" / name, f"TOY/{name}")

    requested = []

    def fake_urlretrieve(url, filename):
        requested.append(url)
        Path(filename).write_bytes(archive.read_bytes())

    monkeypatch.setattr(urllib.request, "urlretrieve", fake_urlretrieve)
    monkeypatch.chdir(tmp_path)
    benchmark_datasets.make("TOY", folder=str(tmp_path / "datasets"))

    assert requested == ["https://www.chrsmrrs.com/graphkerneldatasets/TOY.zip"]
    _check_toy(load_dataset(tmp_path / "datasets" / "TOY.pkl"))
    assert not (tmp_path / "TOY").exists() and not (tmp_path / "TOY.zip").exists()
