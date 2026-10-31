"""Fixture graphs used by the value-level tests.

Everything here is deterministic: the random graphs are built with fixed seeds, so the
baseline files in tests/data describe exactly these graphs.
"""

import networkx as nx
import numpy as np

from hcga.graph import GraphCollection


def _weighted_er(n, p, seed):
    graph = nx.gnp_random_graph(n, p, seed=seed)
    rng = np.random.default_rng(seed)
    for _, _, data in graph.edges(data=True):
        data["weight"] = float(rng.uniform(0.5, 3.0))
    return graph


def _disconnected(seed):
    graph = nx.disjoint_union(nx.gnp_random_graph(25, 0.2, seed=seed), nx.cycle_graph(10))
    graph.add_nodes_from([35, 36])
    return graph


def fixture_specs():
    """Return (name, networkx graph, graph_type) for each fixture graph."""
    return [
        ("karate", nx.karate_club_graph(), None),
        ("ring12", nx.cycle_graph(12), None),
        ("star10", nx.star_graph(10), None),
        ("complete8", nx.complete_graph(8), None),
        ("path7", nx.path_graph(7), None),
        ("er40", nx.gnp_random_graph(40, 0.15, seed=1), None),
        ("er40_weighted", _weighted_er(40, 0.15, seed=2), None),
        ("ba40", nx.barabasi_albert_graph(40, 2, seed=3), None),
        ("dir_er30", nx.gnp_random_graph(30, 0.12, seed=4, directed=True), "directed"),
        ("dir_gn20", nx.gn_graph(20, seed=6), "directed"),
        ("disconnected", _disconnected(seed=5), None),
    ]


def fixture_collection():
    """Return the fixture graphs as a GraphCollection, plus the list of names.

    Graph ids follow the order of fixture_specs(), and labels are the same ids.
    """
    collection = GraphCollection()
    names = []
    for i, (name, graph, graph_type) in enumerate(fixture_specs()):
        collection.add_graph(graph, label=i, graph_type=graph_type)
        names.append(name)
    return collection, names


def fixture_graph(name):
    """Return the hcga Graph object for one fixture by name."""
    collection, names = fixture_collection()
    return collection.graphs[names.index(name)]
