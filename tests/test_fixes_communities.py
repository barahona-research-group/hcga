"""Tests for the fixes to the community, clique, cycle, component and set-cover feature classes.

Each test failed before the corresponding fix (2026-10 audit, audit/REPORT_communities.md).
"""

import sys
import time
from collections import Counter
from math import comb
from pathlib import Path

import networkx as nx
import pytest
from networkx.algorithms.approximation import min_weighted_dominating_set

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixtures import fixture_graph, fixture_specs  # noqa: E402

from hcga.extraction import get_list_feature_classes  # noqa: E402
from hcga.features import (  # noqa: E402
    cliques,
    communities_bisection,
    communities_labelprop,
    covering,
    cycle_basis,
    independent_sets,
    looplessness,
    maximal_matching,
)
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


def _nx_of(graph):
    """The networkx graph that hcga's feature functions receive."""
    return graph.get_graph("networkx")


# --- CommunitiesAsyn -------------------------------------------------------------------------


def test_asyn_all_features_computed_with_integer_names(classes):
    feats = _features(classes, "CommunitiesAsyn", fixture_graph("er40"))
    for c in [2, 4, 6, 8, 10]:
        for name in ["sum_density", "ratio_density", "len_most_dense", "len_least_dense"]:
            assert f"{name}_c={c}" in feats
        for quality in ["modularity", "coverage", "performance"]:
            assert f"partition_c={c}_{quality}" in feats
    assert not any("c=2.0" in name for name in feats)


def test_asyn_matches_networkx_seeded(classes):
    graph = fixture_graph("er40")
    feats = _features(classes, "CommunitiesAsyn", graph)
    comms = list(nx.community.asyn_fluidc(_nx_of(graph), 4, seed=0))
    sizes = sorted(len(c) for c in comms)
    assert feats["sum_density_c=4"] == approx(sum(1 / s for s in sizes))
    assert feats["len_most_dense_c=4"] == sizes[0]
    assert feats["len_least_dense_c=4"] == sizes[-1]
    assert feats["partition_c=4_modularity"] == approx(
        nx.community.modularity(_nx_of(graph), comms)
    )


def test_asyn_deterministic(classes):
    first = _features(classes, "CommunitiesAsyn", fixture_graph("er40"))
    second = _features(classes, "CommunitiesAsyn", fixture_graph("er40"))
    assert first == approx(second)


def test_asyn_disconnected_and_directed(classes):
    feats = _features(classes, "CommunitiesAsyn", fixture_graph("disconnected"))
    assert "partition_c=4_modularity" in feats  # valid partition: singletons outside the LCC
    feats = _features(classes, "CommunitiesAsyn", fixture_graph("dir_er30"))
    assert "sum_density_c=4" in feats


# --- Looplessness -------------------------------------------------------------------------------


def test_looplessness_directed_chain(classes):
    chain = _graph_from_nx(nx.DiGraph([(0, 1), (1, 2), (2, 3)]), graph_type="directed")
    feats = _features(classes, "Looplessness", chain)
    assert feats["trophic_levels_min"] == approx(1.0)
    assert feats["trophic_levels_max"] == approx(4.0)
    assert feats["trophic_levels_mean"] == approx(2.5)
    assert feats["incoherence_parameter"] == approx(0.0)


def test_looplessness_food_web(classes):
    nx_web = nx.DiGraph([(0, 1), (0, 2), (1, 2), (2, 3), (1, 3)])
    feats = _features(classes, "Looplessness", _graph_from_nx(nx_web, graph_type="directed"))
    # levels 1, 2, 2.5, 3.25
    assert feats["trophic_levels_mean"] == approx((1 + 2 + 2.5 + 3.25) / 4)
    assert feats["trophic_levels_max"] == approx(3.25)
    assert feats["incoherence_parameter"] == approx(
        nx.trophic_incoherence_parameter(nx_web, weight=None)
    )
    assert feats["incoherence_parameter"] == approx(0.3535534)


def test_looplessness_undefined_cases(classes):
    assert looplessness.looplessness(nx.Graph([(0, 1), (1, 2)])) is None
    assert looplessness.looplessness(nx.DiGraph([(0, 1), (1, 2), (2, 0)])) is None  # no basal
    feats = _features(classes, "Looplessness", fixture_graph("path7"))
    assert "trophic_levels_mean" not in feats


# --- Components ------------------------------------------------------------------------------


def test_components_sorted_by_size(classes):
    nx_graph = nx.Graph([(0, 1)])
    nx_graph.add_edges_from(nx.complete_graph(range(2, 12)).edges)
    nx_graph.add_edges_from([(20, 21), (21, 22)])
    feats = _features(classes, "Components", _graph_from_nx(nx_graph))
    assert feats["largest_connected_component"] == 10
    assert feats["ratio_largest_connected_components"] == approx(3 / 10)
    assert feats["ratio_maxmin_connected_components"] == approx(10 / 2)


# --- DominatingSets -----------------------------------------------------------------------------


def test_min_dominating_set_uses_networkx(classes):
    graph = fixture_graph("er40")
    feats = _features(classes, "DominatingSets", graph)
    assert feats["size_min_dominating_set"] == len(min_weighted_dominating_set(_nx_of(graph)))


# --- seeds -------------------------------------------------------------------------------------


def test_bisection_seeded(classes):
    graph = fixture_graph("er40")
    first = _features(classes, "CommunitiesBisection", graph)
    second = _features(classes, "CommunitiesBisection", fixture_graph("er40"))
    assert first == approx(second)
    parts = nx.community.kernighan_lin_bisection(_nx_of(graph), seed=0)
    assert first["partition_modularity"] == approx(nx.community.modularity(_nx_of(graph), parts))


def test_independent_set_seeded(classes):
    graph = fixture_graph("er40")
    first = _features(classes, "IndependentSets", graph)
    second = _features(classes, "IndependentSets", fixture_graph("er40"))
    assert first == second
    assert first["size_max_indep_set"] == len(nx.maximal_independent_set(_nx_of(graph), seed=0))


# --- SimpleCycles --------------------------------------------------------------------------


def test_simple_cycles_bounded(classes):
    graph = fixture_graph("er40")
    start = time.perf_counter()
    feats = _features(classes, "SimpleCycles", graph)
    assert time.perf_counter() - start < 15.0  # unbounded enumeration never finishes
    lengths = Counter(len(c) for c in nx.simple_cycles(_nx_of(graph), length_bound=6))
    assert feats["number_simple_cycles"] == sum(lengths.values())
    assert feats["simple_cycles_sizes_max"] == 6


# --- Cliques ---------------------------------------------------------------------------------


def test_clique_counts_match_networkx(classes):
    graph = fixture_graph("karate")
    feats = _features(classes, "Cliques", graph)
    sizes = [len(c) for c in nx.enumerate_all_cliques(_nx_of(graph)) if 2 <= len(c) <= 4]
    assert feats["num_cliques"] == len(sizes)
    assert feats["clique_sizes_mean"] == approx(sum(sizes) / len(sizes))
    assert feats["graph_clique_number"] == 5
    assert feats["num_max_cliques"] == sum(1 for _ in nx.find_cliques(_nx_of(graph)))


def test_complete_graph_clique_counts(classes):
    feats = _features(classes, "Cliques", fixture_graph("complete8"))
    assert feats["num_cliques"] == comb(8, 2) + comb(8, 3) + comb(8, 4)


def test_cliques_dense_graph_fast(classes):
    graph = _graph_from_nx(nx.gnp_random_graph(150, 0.5, seed=0))
    start = time.perf_counter()
    feats = _features(classes, "Cliques", graph)
    assert time.perf_counter() - start < 30.0  # was about 17 s on an idle machine
    assert "num_cliques" in feats


# --- CommunitiesModularity / LabelPropagation / Bisection ----------------------------------------


def test_unweighted_partition_modularity_ignores_weights(classes):
    graph = fixture_graph("er40_weighted")
    nx_graph = _nx_of(graph)
    feats = _features(classes, "CommunitiesModularity", graph)
    comms = nx.community.greedy_modularity_communities(nx_graph, weight=None)
    assert feats["communities_modularity"] == approx(
        nx.community.modularity(nx_graph, comms, weight=None)
    )
    assert feats["communities_modularity"] != approx(
        nx.community.modularity(nx_graph, comms, weight="weight")
    )


def test_undirected_only_functions_convert_directed():
    # explicit conversion in the feature functions, independent of the base-class fallback
    digraph = _nx_fixture("dir_er30")
    undirected = digraph.to_undirected()
    assert len(communities_labelprop.eval_labelprop(digraph)) >= 1
    assert sum(len(c) for c in communities_bisection.eval_bisection(digraph)) == len(digraph)
    assert covering.min_edge_cover(digraph) == len(nx.min_edge_cover(undirected))
    assert maximal_matching.maximal_matching(digraph) >= 1
    assert independent_sets.size_max_indep_set(digraph) >= 1
    assert cliques.eval_maximal_cliques(digraph) == [len(c) for c in nx.find_cliques(undirected)]
    assert cycle_basis.num_cycles(digraph) == len(nx.cycle_basis(undirected))


def test_undirected_only_classes_run_on_directed(classes):
    graph = fixture_graph("dir_er30")
    undirected = _nx_of(graph).to_undirected()
    feats = _features(classes, "CommunitiesLabelPropagation", graph)
    assert "largest_commsize" in feats
    feats = _features(classes, "CommunitiesBisection", graph)
    assert "partition_modularity" in feats
    feats = _features(classes, "Covering", graph)
    assert feats["min_edge_cover"] == len(nx.min_edge_cover(undirected))
    feats = _features(classes, "MaximalMatching", graph)
    assert "maximal_matching" in feats


# --- JaccardSimilarity ---------------------------------------------------------------------------


def test_jaccard_star(classes):
    # Jaccard graph of a star: the 10 leaves form a clique (they share the hub); hub isolated
    feats = _features(classes, "JaccardSimilarity", fixture_graph("star10"))
    assert feats["number_of_edges"] == comb(10, 2)
    assert feats["number_of_edges_no_selfloops"] == comb(10, 2)
    assert feats["largest_connected_component"] == 10
    assert feats["num_connected_components"] == 2
    assert feats["diameter"] == 1
    assert feats["radius"] == 1


def test_jaccard_connectance_no_selfloops(classes):
    feats = _features(classes, "JaccardSimilarity", _graph_from_nx(nx.cycle_graph(3)))
    assert feats["connectance"] == approx(1.0)


# --- CycleBasis / Covering -----------------------------------------------------------------------


def test_minimum_cycle_length_is_girth():
    graph = nx.random_regular_graph(3, 12, seed=65)
    assert min(len(c) for c in nx.cycle_basis(graph)) == 4  # fundamental basis misses it
    assert cycle_basis.minimum_cycle_length(graph) == nx.girth(graph) == 3


def test_covering_isolated_node_returns_none():
    graph = nx.path_graph(4)
    graph.add_node(10)
    assert covering.min_edge_cover(graph) is None
