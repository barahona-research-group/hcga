"""Tests for the accuracy fixes in the centralities, paths, distances, spectral and flow classes.

Each test fails on the code before the fix and passes after it.
"""

import sys
import threading
import warnings
from math import comb
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixtures import fixture_graph, fixture_specs  # noqa: E402

from hcga.features import efficiency as efficiency_module  # noqa: E402
from hcga.features import rbc as rbc_module  # noqa: E402
from hcga.features.centralities_basic import CentralitiesBasic  # noqa: E402
from hcga.features.distance_measures import DistanceMeasures  # noqa: E402
from hcga.features.efficiency import Efficiency  # noqa: E402
from hcga.features.flow_hierarchy import FlowHierarchy  # noqa: E402
from hcga.features.k_components import KComponents  # noqa: E402
from hcga.features.link_analysis_hits import Hits  # noqa: E402
from hcga.features.minimum_cuts import MinimumCuts  # noqa: E402
from hcga.features.node_connectivity import NodeConnectivity  # noqa: E402
from hcga.features.rbc import RolesimilarityBasedComparison  # noqa: E402
from hcga.features.shortest_paths import ShortestPaths  # noqa: E402
from hcga.features.small_worldness import SmallWorldness  # noqa: E402
from hcga.features.spectrum import Spectrum  # noqa: E402
from hcga.features.vitality import Vitality  # noqa: E402
from hcga.graph import GraphCollection  # noqa: E402

approx = pytest.approx

CLASSES = [
    CentralitiesBasic,
    DistanceMeasures,
    Efficiency,
    FlowHierarchy,
    Hits,
    KComponents,
    MinimumCuts,
    NodeConnectivity,
    RolesimilarityBasedComparison,
    ShortestPaths,
    SmallWorldness,
    Spectrum,
    Vitality,
]


@pytest.fixture(scope="module", autouse=True)
def setup_classes():
    for cls in CLASSES:
        cls.setup_class(normalize_features=False, statistics_level="basic", timeout=0)


def _features(cls, graph):
    return cls(graph).get_features()


def _nx_fixture(name):
    return dict((n, g) for n, g, _ in fixture_specs())[name]


def _graph_from_nx(graph, graph_type=None):
    collection = GraphCollection()
    collection.add_graph(graph, label=0, graph_type=graph_type)
    return collection.graphs[0]


def _hcga_nx(name):
    """The networkx graph hcga builds for a fixture (weights included)."""
    return fixture_graph(name).get_graph("networkx")


def _weak_lcc(graph):
    return graph.subgraph(max(nx.weakly_connected_components(graph), key=len))


# --- Hits -----------------------------------------------------------------------------------


def test_hits_hubs_computed():
    graph = _hcga_nx("karate")
    hubs, _ = nx.hits(graph, max_iter=1000)
    feats = _features(Hits, fixture_graph("karate"))
    assert feats["Hits_max"] == approx(max(hubs.values()))
    assert feats["Hits_min"] == approx(min(hubs.values()))


def test_hits_authorities_on_digraph():
    graph = _weak_lcc(_hcga_nx("dir_er30"))
    hubs, authorities = nx.hits(graph, max_iter=1000)
    feats = _features(Hits, fixture_graph("dir_er30"))
    assert feats["hits_authorities_max"] == approx(max(authorities.values()))
    assert feats["hits_authorities_std"] == approx(np.std(list(authorities.values())))
    assert feats["Hits_max"] == approx(max(hubs.values()))


# --- ShortestPaths --------------------------------------------------------------------------


def test_shortest_paths_per_node_features_on_karate():
    graph = nx.Graph(nx.karate_club_graph().edges())  # unweighted, so lengths are hop counts
    feats = _features(ShortestPaths, _graph_from_nx(graph))
    # the mean over nodes of each node's mean distance is the average shortest path length
    assert feats["mean_shortest_path_mean"] == approx(nx.average_shortest_path_length(graph))
    assert feats["largest_shortest_path_max"] == nx.diameter(graph)
    assert feats["largest_shortest_path_min"] == nx.radius(graph)
    assert feats["radius_directed"] == nx.radius(graph)


def test_shortest_paths_path_graph_closed_form():
    feats = _features(ShortestPaths, fixture_graph("path7"))
    # end node: (1 + ... + 6) / 6 = 3.5; middle node: 2 * (1 + 2 + 3) / 6 = 2
    assert feats["mean_shortest_path_max"] == approx(3.5)
    assert feats["mean_shortest_path_min"] == approx(2.0)
    assert feats["largest_shortest_path_max"] == 6
    assert feats["largest_shortest_path_min"] == 3
    assert feats["number_shortest_paths_directed"] == 7 * 6
    assert feats["radius_directed"] == 3


def test_shortest_paths_weighting_is_consistent():
    graph = _hcga_nx("er40_weighted")
    weighted = dict(nx.all_pairs_dijkstra_path_length(graph, weight="weight"))
    # nodes that reach no other node (isolated or sinks) have no eccentricity
    ecc = [max(d.values()) for d in weighted.values() if len(d) > 1]
    feats = _features(ShortestPaths, fixture_graph("er40_weighted"))
    assert feats["diameter_directed"] == approx(max(ecc))
    assert feats["largest_shortest_path_max"] == approx(max(ecc))
    assert feats["radius_directed"] == approx(min(ecc))
    mean_dist = [
        np.mean([x for v, x in d.items() if v != u]) for u, d in weighted.items() if len(d) > 1
    ]
    assert feats["mean_shortest_path_mean"] == approx(np.mean(mean_dist))


# --- Spectrum -------------------------------------------------------------------------------


def test_spectrum_eigenvalue_ratios_use_sorted_eigenvalues():
    graph = _nx_fixture("er40")
    eigenvalues = np.sort(np.linalg.eigvalsh(nx.to_numpy_array(graph)))[::-1]
    feats = _features(Spectrum, fixture_graph("er40"))
    for i, j in [(1, 0), (5, 0), (9, 3)]:
        assert feats[f"eigenvalue_ratio_{i}_{j}"] == approx(eigenvalues[i] / eigenvalues[j])


def test_spectrum_complete_graph_ratio():
    # K8 has eigenvalues 7, -1 (x7)
    feats = _features(Spectrum, fixture_graph("complete8"))
    assert feats["eigenvalue_ratio_1_0"] == approx(-1 / 7)
    assert feats["eigenvalue_ratio_7_1"] == approx(1.0)


def test_spectrum_directed_laplacian_real_parts():
    graph = _hcga_nx("dir_er30")
    laplacian = nx.laplacian_matrix(graph).toarray()
    real_parts = np.linalg.eigvals(laplacian).real
    feats = _features(Spectrum, fixture_graph("dir_er30"))
    # out-degree Laplacian eigenvalues have non-negative real parts (Gershgorin)
    assert feats["eigenvalues_laplacian_min"] >= -1e-9
    assert feats["eigenvalues_laplacian_min"] == approx(real_parts.min(), abs=1e-9)
    assert feats["eigenvalues_laplacian_max"] == approx(real_parts.max())


def test_spectrum_directed_adjacency_sorted_by_real_part():
    graph = _hcga_nx("dir_er30")
    eigenvalues = np.linalg.eigvals(nx.to_numpy_array(graph))
    real_parts = np.sort(eigenvalues.real)[::-1]
    feats = _features(Spectrum, fixture_graph("dir_er30"))
    assert feats["eigenvalue_ratio_1_0"] == approx(real_parts[1] / real_parts[0])


# --- CentralitiesBasic ----------------------------------------------------------------------


def test_weighted_eigenvector_centrality_uses_weights():
    graph = _hcga_nx("er40_weighted")
    graph = graph.subgraph(max(nx.connected_components(graph), key=len))
    weighted = nx.eigenvector_centrality_numpy(graph, weight="weight")
    unweighted = nx.eigenvector_centrality_numpy(graph)
    feats = _features(CentralitiesBasic, fixture_graph("er40_weighted"))
    assert feats["eigenvector centrality weighted_max"] == approx(max(weighted.values()))
    assert feats["eigenvector centrality_max"] == approx(max(unweighted.values()))
    weighted_max = feats["eigenvector centrality weighted_max"]
    assert weighted_max != approx(feats["eigenvector centrality_max"])


def test_pagerank_unweighted():
    graph = _hcga_nx("er40_weighted")
    unweighted = nx.pagerank(graph, weight=None)
    feats = _features(CentralitiesBasic, fixture_graph("er40_weighted"))
    assert feats["pagerank_max"] == approx(max(unweighted.values()))
    assert feats["pagerank_std"] == approx(np.std(list(unweighted.values())))
    assert feats["pagerank_max"] != approx(feats["pagerank weighted_max"])


def test_katz_centrality_positive_on_dense_graph():
    # lambda_max ~ 24 > 1 / 0.1, where the old fixed alpha = 0.1 gives negative centralities
    graph = nx.gnp_random_graph(60, 0.4, seed=0)
    lambda_max = np.max(np.abs(np.linalg.eigvals(nx.to_numpy_array(graph))))
    katz = nx.katz_centrality_numpy(graph, alpha=0.5 / lambda_max)
    feats = _features(CentralitiesBasic, _graph_from_nx(graph))
    assert feats["katz centrality_min"] > 0
    assert feats["katz centrality_min"] == approx(min(katz.values()))
    assert feats["katz centrality_max"] == approx(max(katz.values()))


def test_centralities_on_digraph():
    graph = _hcga_nx("dir_er30")
    feats = _features(CentralitiesBasic, fixture_graph("dir_er30"))

    subgraph = nx.subgraph_centrality(graph.to_undirected())
    assert feats["subgraph centrality_mean"] == approx(np.mean(list(subgraph.values())))

    second_order = nx.second_order_centrality(_weak_lcc(graph).to_undirected())
    assert feats["second order centrality_mean"] == approx(np.mean(list(second_order.values())))

    scc = graph.subgraph(max(nx.strongly_connected_components(graph), key=len))
    eigenvector = nx.eigenvector_centrality_numpy(scc)
    assert feats["eigenvector centrality_max"] == approx(max(eigenvector.values()))
    assert "eigenvector centrality weighted_max" in feats


# --- NodeConnectivity -----------------------------------------------------------------------


def test_node_connectivity_pairs():
    graph = _nx_fixture("karate")
    average = nx.average_node_connectivity(graph)
    feats = _features(NodeConnectivity, fixture_graph("karate"))
    assert feats["node_conn_mean"] == approx(average)
    assert feats["node_conn_sum"] == approx(average * comb(34, 2))
    assert feats["node_conn_min"] == nx.node_connectivity(graph)
    assert feats["node_connectivity"] == nx.node_connectivity(graph)
    assert feats["avg_node_connectivity"] == approx(average)


def test_node_connectivity_directed_uses_ordered_pairs():
    graph = _hcga_nx("dir_er30")
    feats = _features(NodeConnectivity, fixture_graph("dir_er30"))
    assert feats["node_conn_mean"] == approx(nx.average_node_connectivity(graph))
    assert feats["node_conn_sum"] == approx(nx.average_node_connectivity(graph) * 30 * 29)


# --- KComponents ----------------------------------------------------------------------------


def test_k_components_largest_component():
    # k_components lists the 4-cycle before the 9-cycle at k = 2 (the largest k)
    graph = nx.disjoint_union(nx.cycle_graph(4), nx.cycle_graph(9))
    feats = _features(KComponents, _graph_from_nx(graph))
    assert feats["size_2_component"] == 9
    assert feats["size_max_k_component"] == 9


def test_k_components_tree_has_no_2_component():
    feats = _features(KComponents, fixture_graph("path7"))
    assert feats["size_2_component"] == 0
    assert feats["size_max_k_component"] == 7


# --- FlowHierarchy --------------------------------------------------------------------------


def test_flow_hierarchy_undefined_on_undirected():
    feats = _features(FlowHierarchy, fixture_graph("ring12"))
    assert "flow_hierarchy" not in feats
    assert "flow_hierarchy_weighted" not in feats


def test_flow_hierarchy_directed():
    graph = _hcga_nx("dir_er30")
    feats = _features(FlowHierarchy, fixture_graph("dir_er30"))
    assert feats["flow_hierarchy"] == approx(nx.flow_hierarchy(graph))


# --- Vitality -------------------------------------------------------------------------------


def test_vitality_path_graph():
    # removing an end of P7 leaves P6: W(P7) - W(P6) = 56 - 35 = 21; the 5 inner nodes
    # are articulation points (vitality -inf, dropped)
    feats = _features(Vitality, fixture_graph("path7"))
    assert feats["vitality_min"] == approx(21.0)
    assert feats["vitality_max"] == approx(21.0)
    assert feats["n_articulation_points"] == 5


def test_vitality_matches_networkx_on_lcc():
    graph = _nx_fixture("disconnected")
    lcc = graph.subgraph(max(nx.connected_components(graph), key=len))
    values = np.array(list(nx.closeness_vitality(lcc).values()))
    finite = values[np.isfinite(values)]
    feats = _features(Vitality, fixture_graph("disconnected"))
    assert np.isfinite(feats["vitality_mean"])
    assert feats["vitality_mean"] == approx(finite.mean())
    assert feats["vitality_min"] == approx(finite.min())
    assert feats["n_articulation_points"] == len(list(nx.articulation_points(lcc)))


# --- DistanceMeasures -----------------------------------------------------------------------


def test_distance_measures_on_digraph():
    graph = _weak_lcc(_hcga_nx("dir_er30")).to_undirected()
    feats = _features(DistanceMeasures, fixture_graph("dir_er30"))
    assert feats["eccentricity_max"] == nx.diameter(graph)
    assert feats["extrema_bounding"] == nx.diameter(graph)
    assert feats["center_size"] == len(nx.center(graph))
    assert feats["periphery"] == len(nx.periphery(graph))
    assert feats["barycenter_size"] == len(nx.barycenter(graph))


def test_distance_measures_all_pairs_computed_once(monkeypatch):
    calls = []
    original = nx.shortest_path_length

    def counting(G, source=None, target=None, weight=None, method="dijkstra"):
        calls.append(weight)
        return original(G, source=source, target=target, weight=weight, method=method)

    monkeypatch.setattr(nx, "shortest_path_length", counting)
    feats = _features(DistanceMeasures, fixture_graph("karate"))
    assert feats["eccentricity_max"] == 5
    # one unweighted all-pairs call shared by all features (previously one more per node for
    # each of center, periphery and eccentricity); the weighted barycenter makes its own call
    assert calls.count(None) == 1


# --- Efficiency and MinimumCuts on digraphs -------------------------------------------------


def test_efficiency_on_digraph():
    # called directly, so the test does not depend on the base class's undirected fallback
    digraph = _hcga_nx("dir_er30")
    graph = digraph.to_undirected()
    assert efficiency_module.global_efficiency(digraph) == approx(nx.global_efficiency(graph))
    assert efficiency_module.local_efficiency(digraph) == approx(nx.local_efficiency(graph))
    feats = _features(Efficiency, fixture_graph("dir_er30"))
    assert feats["global_efficiency"] == approx(nx.global_efficiency(graph))


def test_minimum_cuts_on_digraph():
    graph = _weak_lcc(_hcga_nx("dir_er30")).to_undirected()
    feats = _features(MinimumCuts, fixture_graph("dir_er30"))
    assert feats["min_node_cut_size"] == nx.node_connectivity(graph)
    assert feats["min_edge_cut_size"] == nx.edge_connectivity(graph)


# --- SmallWorldness -------------------------------------------------------------------------


def test_small_worldness_deterministic():
    first = _features(SmallWorldness, fixture_graph("karate"))["omega"]
    second = _features(SmallWorldness, fixture_graph("karate"))["omega"]
    assert first == second


def test_small_worldness_disconnected_uses_lcc():
    feats = _features(SmallWorldness, fixture_graph("disconnected"))
    assert np.isfinite(feats["omega"])


def test_small_worldness_star_does_not_hang():
    result = {}

    def run():
        result["feats"] = _features(SmallWorldness, _graph_from_nx(nx.star_graph(10)))

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout=20)
    assert not thread.is_alive(), "nx.omega hangs on a star graph"
    assert "omega" not in result["feats"]


# --- RBC ------------------------------------------------------------------------------------


def test_rbc_beta_is_real():
    # a digraph with complex adjacency eigenvalues: the old max(np.linalg.eig(a)[0]) gave a
    # complex beta, and assigning the walk counts into a float array raised ComplexWarning
    graph = _hcga_nx("dir_er30")
    rbc_module.rbc.cache_clear()
    with warnings.catch_warnings():
        warnings.simplefilter("error", np.exceptions.ComplexWarning)
        similarity_graph = rbc_module.rbc(graph)
    assert similarity_graph.number_of_nodes() == 30


def test_rbc_similarity_graph_has_no_self_loops():
    rbc_module.rbc.cache_clear()
    assert nx.number_of_selfloops(rbc_module.rbc(_hcga_nx("dir_er30"))) == 0
    feats = _features(RolesimilarityBasedComparison, fixture_graph("dir_er30"))
    assert feats["number_of_edges"] == feats["number_of_edges_no_selfloops"]


def test_rbc_descriptions_do_not_say_jaccard():
    descriptions = str(RolesimilarityBasedComparison.feature_descriptions)
    assert "Jaccard" not in descriptions


def test_spectrum_ratio_nan_for_zero_eigenvalue():
    # C12 has adjacency eigenvalues 2cos(2*pi*k/12); the 6th and 7th largest are exactly 0,
    # so ratios with those denominators are rounding noise and are reported as NaN
    feats = _features(Spectrum, fixture_graph("ring12"))
    assert np.isnan(feats["eigenvalue_ratio_6_5"])
    assert np.isnan(feats["eigenvalue_ratio_7_6"])
    assert feats["eigenvalue_ratio_1_0"] == approx(np.sqrt(3) / 2)
    # a 7-node graph has no 8th eigenvalue, so that ratio is not emitted at all
    assert "eigenvalue_ratio_7_3" not in _features(Spectrum, fixture_graph("path7"))


def test_constant_distribution_statistics_are_exact():
    # every node of K8 has the same subgraph centrality; the summary statistics must not
    # carry BLAS rounding noise (kstatvar of a constant vector came out as ~1e-7)
    feats = _features(CentralitiesBasic, fixture_graph("complete8"))
    assert feats["subgraph centrality_std"] == 0.0
    assert feats["subgraph centrality_min"] == feats["subgraph centrality_max"]
