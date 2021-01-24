"""Utils functions."""
import logging

import pandas as pd

from hcga.graph import Graph

L = logging.getLogger(__name__)


class TimeoutError(Exception):
    """Timeout exception."""


def timeout_handler(signum, frame):
    """Function to raise timeout exception."""
    raise TimeoutError


def get_trivial_graph(n_node_features=0):
    """Generate a trivial graph for internal purposes."""
    nodes = pd.DataFrame([0, 1, 2])
    if n_node_features > 0:
        nodes["features"] = 3 * [n_node_features * [0.0]]
    edges = pd.DataFrame()
    edges["start_node"] = [0, 1, 2]
    edges["end_node"] = [1, 2, 0]
    return Graph(nodes, edges, 0)
