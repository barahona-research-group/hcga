"""Value-level tests against closed-form results on small named graphs.

Tests marked xfail(strict=True) document bugs found in the 2026-10 audit. When a bug is fixed
the test will XPASS and fail the run, which is the reminder to delete the marker.
"""

import sys
from math import comb
from pathlib import Path

import networkx as nx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixtures import fixture_graph, fixture_specs  # noqa: E402

from hcga.extraction import get_list_feature_classes  # noqa: E402
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


# --- complete graph K8 ---------------------------------------------------------------------


def test_complete_basic_stats(classes):
    feats = _features(classes, "BasicStats", fixture_graph("complete8"))
    assert feats["num_nodes"] == 8
    assert feats["num_edges"] == 28
    assert feats["diameter"] == 1
    assert feats["radius"] == 1


def test_complete_connectance(classes):
    feats = _features(classes, "Connectance", fixture_graph("complete8"))
    assert feats["connectance"] == approx(1.0)


@pytest.mark.xfail(strict=True, reason="BasicStats.density divides by N(E-1) instead of N(N-1)")
def test_complete_density(classes):
    feats = _features(classes, "BasicStats", fixture_graph("complete8"))
    assert feats["density"] == approx(1.0)


def test_complete_clustering(classes):
    feats = _features(classes, "Clustering", fixture_graph("complete8"))
    assert feats["transitivity"] == approx(1.0)
    assert feats["clustering_mean"] == approx(1.0)
    assert feats["clustering_std"] == approx(0.0)


@pytest.mark.xfail(strict=True, reason="num_triangles is the mean triangles per node, not the count")
def test_complete_num_triangles(classes):
    feats = _features(classes, "Clustering", fixture_graph("complete8"))
    assert feats["num_triangles"] == comb(8, 3)


def test_complete_centralities(classes):
    feats = _features(classes, "CentralitiesBasic", fixture_graph("complete8"))
    assert feats["degree centrality_mean"] == approx(1.0)
    assert feats["degree centrality_std"] == approx(0.0)
    assert feats["betweenness centrality_max"] == approx(0.0)
    assert feats["closeness centrality_min"] == approx(1.0)


def test_complete_cliques(classes):
    feats = _features(classes, "Cliques", fixture_graph("complete8"))
    assert feats["graph_clique_number"] == 8
    assert feats["num_max_cliques"] == 1


def test_complete_core_number(classes):
    feats = _features(classes, "CoreNumber", fixture_graph("complete8"))
    assert feats["core number_min"] == 7
    assert feats["core number_max"] == 7


def test_complete_spectrum(classes):
    feats = _features(classes, "Spectrum", fixture_graph("complete8"))
    assert feats["eigenvalues_adjacency_max"] == approx(7.0)
    assert feats["eigenvalues_adjacency_min"] == approx(-1.0)
    assert feats["eigenvalues_laplacian_max"] == approx(8.0)


# --- cycle C12 -----------------------------------------------------------------------------


def test_ring_basic_stats(classes):
    feats = _features(classes, "BasicStats", fixture_graph("ring12"))
    assert feats["diameter"] == 6
    assert feats["radius"] == 6


def test_ring_cycle_basis(classes):
    feats = _features(classes, "CycleBasis", fixture_graph("ring12"))
    assert feats["num_cycles"] == 1
    assert feats["minimum_cycle_length"] == 12


def test_ring_spectrum(classes):
    feats = _features(classes, "Spectrum", fixture_graph("ring12"))
    assert feats["eigenvalues_laplacian_sum"] == approx(24.0)  # 2E
    assert feats["eigenvalues_laplacian_max"] == approx(4.0)
    assert feats["eigenvalues_laplacian_min"] == approx(0.0, abs=1e-9)
    assert feats["eigenvalues_adjacency_max"] == approx(2.0)


def test_ring_is_vertex_transitive(classes):
    feats = _features(classes, "CentralitiesBasic", fixture_graph("ring12"))
    for name in ["betweenness centrality", "closeness centrality", "degree centrality"]:
        assert feats[f"{name}_std"] == approx(0.0, abs=1e-12)


def test_ring_eulerian(classes):
    feats = _features(classes, "Eulerian", fixture_graph("ring12"))
    assert bool(feats["eulerian"]) is True


def test_ring_clustering(classes):
    feats = _features(classes, "Clustering", fixture_graph("ring12"))
    assert feats["transitivity"] == approx(0.0)


# --- star S10 (11 nodes) --------------------------------------------------------------------


def test_star_centralities(classes):
    feats = _features(classes, "CentralitiesBasic", fixture_graph("star10"))
    assert feats["betweenness centrality_max"] == approx(1.0)
    assert feats["degree centrality_max"] == approx(1.0)
    assert feats["closeness centrality_max"] == approx(1.0)
    assert feats["degree centrality_min"] == approx(0.1)


def test_star_distances(classes):
    feats = _features(classes, "DistanceMeasures", fixture_graph("star10"))
    assert feats["center_size"] == 1
    assert feats["periphery"] == 10
    assert feats["eccentricity_min"] == 1
    assert feats["eccentricity_max"] == 2
    basic = _features(classes, "BasicStats", fixture_graph("star10"))
    assert basic["diameter"] == 2
    assert basic["radius"] == 1


def test_star_assortativity(classes):
    feats = _features(classes, "Assortativity", fixture_graph("star10"))
    assert feats["degree_assortativity_coeff"] == approx(-1.0)


def test_star_cliques(classes):
    feats = _features(classes, "Cliques", fixture_graph("star10"))
    assert feats["graph_clique_number"] == 2
    assert feats["num_max_cliques"] == 10


def test_star_wiener(classes):
    feats = _features(classes, "ChemicalTheory", fixture_graph("star10"))
    assert feats["wiener index"] == approx(10 + 2 * comb(10, 2))


# --- path P7 -------------------------------------------------------------------------------


def test_path_basic(classes):
    basic = _features(classes, "BasicStats", fixture_graph("path7"))
    assert basic["diameter"] == 6
    assert basic["radius"] == 3
    cycles = _features(classes, "CycleBasis", fixture_graph("path7"))
    assert cycles["num_cycles"] == 0
    comps = _features(classes, "Components", fixture_graph("path7"))
    assert bool(comps["is_connected"]) is True
    assert comps["num_connected_components"] == 1


def test_path_wiener(classes):
    feats = _features(classes, "ChemicalTheory", fixture_graph("path7"))
    assert feats["wiener index"] == approx(sum((7 - d) * d for d in range(1, 7)))


def test_path_distances(classes):
    feats = _features(classes, "DistanceMeasures", fixture_graph("path7"))
    assert feats["center_size"] == 1
    assert feats["periphery"] == 2


# --- components -----------------------------------------------------------------------------


def test_disconnected_component_count(classes):
    nx_graph = _nx_fixture("disconnected")
    feats = _features(classes, "Components", fixture_graph("disconnected"))
    assert feats["num_connected_components"] == nx.number_connected_components(nx_graph)
    assert bool(feats["is_connected"]) is False


@pytest.mark.xfail(strict=True, reason="largest_connected_component takes the first component, not the largest")
def test_largest_component_is_largest(classes):
    nx_graph = nx.Graph([(0, 1)])
    nx_graph.add_edges_from(nx.complete_graph(range(2, 12)).edges)
    nx_graph.add_edges_from([(20, 21), (21, 22)])
    feats = _features(classes, "Components", _graph_from_nx(nx_graph))
    assert feats["largest_connected_component"] == 10


# --- directed -------------------------------------------------------------------------------


def test_directed_reciprocity(classes):
    nx_graph = _nx_fixture("dir_er30")
    feats = _features(classes, "Reciprocity", fixture_graph("dir_er30"))
    assert feats["reciprocity"] == approx(nx.overall_reciprocity(nx_graph))


def test_directed_in_out_degree_sums(classes):
    nx_graph = _nx_fixture("dir_er30")
    feats = _features(classes, "InOutDegrees", fixture_graph("dir_er30"))
    assert feats["in_degree_sum"] == nx_graph.number_of_edges()
    assert feats["out_degree_sum"] == nx_graph.number_of_edges()
