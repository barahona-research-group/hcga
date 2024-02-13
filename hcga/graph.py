"""
Classes for creating graph objects compatible with hcga.

Graphs can be constructed in various formats in Python, from Networkx Graph objects
to numpy matrices.  To provide a consistent and reliable format for hcga we have
constructed a Graph class and a GraphCollection class.

When loading graphs into hcga, the Graph classes will attempt to convert the input
graph type into a generic hcga Graph object.

"""

import logging

import networkx as nx
import numpy as np
import pandas as pd
import scipy as sc

MIN_NUM_NODES = 2
MIN_NUM_EDGES = 1
L = logging.getLogger(__name__)


class GraphCollection:
    """A collection of Graph objects (see Graph class)."""

    def __init__(self):
        """Initialise an empty list of graphs."""
        self.graphs = []

    def add_graph(self, graph, node_features=None, label=None, graph_type=None):
        """Add a graph to the list.

        Args:
            graph (graph-like object): valid data representig graph (see convert_graph)
            node_feature (array): node feature matrix
            label (int): label of the graph
            graph_type (str): set to 'directed' for directed graphs
        """
        if not isinstance(graph, Graph):
            graph = convert_graph(graph, node_features, label, graph_type)
        graph.id = len(self.graphs)
        self.graphs.append(graph)

    def add_graph_list(
        self, graph_list, node_features_list=None, graph_labels=None, graph_type=None
    ):
        """Add a list of graphs.

        Args:
            graph_list (list(graph-like object)): valid data representig graphs (see convert_graph)
            node_feature_list (list(array)): node feature matrices
            graph_labels (list(int)): label of the graphs
            graph_type (str): set to 'directed' for directed graphs
        """
        for i, graph in enumerate(graph_list):
            if not isinstance(graph, Graph):
                if not graph_labels:
                    graph_label = None
                else:
                    graph_label = graph_labels[i]

                if not node_features_list:
                    node_features = None
                else:
                    node_features = node_features_list[i]

                graph = convert_graph(graph, node_features, graph_label, graph_type)
            graph.id = len(self.graphs)
            self.graphs.append(graph)

    def __iter__(self):
        self.current_graph = -1
        return self

    def __next__(self):
        self.current_graph += 1
        if self.current_graph >= len(self.graphs):
            raise StopIteration
        while self.graphs[self.current_graph].disabled:
            self.current_graph += 1
            if self.current_graph >= len(self.graphs):
                raise StopIteration
        return self.graphs[self.current_graph]

    def __len__(self):
        return sum(1 for graph in self.graphs if not graph.disabled)

    def get_n_node_features(self):
        """Get the number of features of the nodes."""
        n_node_features = self.graphs[0].n_node_features
        for graph in self.graphs:
            assert n_node_features == graph.n_node_features
        return n_node_features

    def get_num_disabled_graphs(self):
        """Get the number of disabled graphs."""
        return len(self.graphs) - len(self)

    def remove_node_features(self):
        """Remove the node features."""
        for graph in self.graphs:
            for node_id, node in enumerate(graph.nodes):
                graph.nodes[node_id] = node[0]
            graph.set_node_features()

    def remove_edge_weights(self):
        """Remove edge weights."""
        for graph in self.graphs:
            graph.remove_weights()

    def get_graph_ids(self):
        """Get the list of active graph ids."""
        return [graph.id for graph in self.graphs if not graph.disabled]

    def maximal_subgraphs(self):
        """Overwrites each graph with its maximal subgraph."""
        L.warning("Returning the maximal subgraph for each graph")

        for graph in self.graphs:
            graph.maximal_subgraph()


class Graph:
    """
    Class to encode a generic graph structure for hcga.

    A graph can be attributed nodes, edges, a graph label and node features.
    """

    def __init__(self, nodes, edges, label, graph_type=None, label_name=None):
        """Defining the main graphs quantities.

        Args:
            nodes (DataFrame): node dataframe, index as node id, and optional
                label and attributes columns (with lists elements)
            edges (DataFrame): edge dataframe, with two columns 'start_node' and 'end_node'
                with id corresponding to indices in nodes. And a third optional column 'weight'
                which if absent all edges will default to weight 1.
            label (int): label of the graph, it has to be an integer
            graph_type (str): set to 'directed' for directed graphs
            label_name (any): name or other information on the graph label
        """
        nodes["new_index"] = np.arange(0, len(nodes.index))
        edges["start_node"] = nodes.new_index[edges["start_node"].to_list()].to_list()
        edges["end_node"] = nodes.new_index[edges["end_node"].to_list()].to_list()

        if "weight" not in edges:
            edges["weight"] = 1.0

        self.nodes = nodes.set_index("new_index")
        self.edges = edges.reset_index()

        self.label = label
        self.graph_type = graph_type
        self.label_name = label_name
        self.disabled = False
        self.id = -1

        self._check_length()
        self.set_node_features()

    def set_node_features(self):
        """Set the node features and get the dimension of features."""
        if "features" in self.nodes:
            self.n_node_features = len(self.nodes.features.iloc[0])
        else:
            if "labels" in self.nodes:
                features_lab = np.asarray(self.nodes["labels"].to_list())
            if "attributes" in self.nodes:
                features_attr = np.asarray(self.nodes["attributes"].to_list())
            if "labels" in self.nodes and "attributes" in self.nodes:
                features = np.concatenate((features_lab, features_attr), axis=1)
            if "labels" in self.nodes and "attributes" not in self.nodes:
                features = features_lab
            if "labels" not in self.nodes and "attributes" in self.nodes:
                features = features_attr
            if "labels" in self.nodes or "attributes" in self.nodes:
                self.nodes["features"] = list(features)
                self.n_node_features = len(features[0])
            else:
                self.n_node_features = 0

    def remove_weights(self):
        """Set edge weights to one."""
        self.edges["weight"] = 1.0

    def _check_length(self):
        """Verify if the graph is large enough to be considered."""
        if len(self.nodes.index) <= MIN_NUM_NODES:
            self.disabled = True
        if len(self.edges.index) <= MIN_NUM_EDGES:
            self.disabled = True

    def get_graph(self, encoding=None):
        """Get usable graph structure with given encoding.

        For now, only networkx is implemented.
        """
        if encoding is None:
            return self
        if encoding == "networkx":
            if not hasattr(self, "_graph_networkx"):
                self.set_networkx()
            return self._graph_networkx
        if encoding == "igraph":
            raise Exception("Igraph is not yet implemented")
        raise Exception("Graph encoding not understood")

    def set_networkx(self):
        """Set the networkx graph encoding."""
        if self.graph_type is None:
            self._graph_networkx = nx.Graph()
        elif self.graph_type == "directed":
            self._graph_networkx = nx.DiGraph()
        else:
            raise Exception("graph type not recognised or not yet implemented")
        if self.n_node_features == 0:
            nodes = [(node, {"feat": [0]}) for node, node_data in self.nodes.iterrows()]
        else:
            nodes = [
                (node, {"feat": node_data["features"]}) for node, node_data in self.nodes.iterrows()
            ]
        self._graph_networkx.add_nodes_from(nodes)

        edges = [
            (edge["start_node"], edge["end_node"], edge["weight"])
            for _, edge in self.edges.iterrows()
        ]
        self._graph_networkx.add_weighted_edges_from(edges)

    def maximal_subgraph(self):
        """Overwrite the graph with its maximal subgraph."""
        n_nodes = len(self.nodes.index)
        n_edges = len(self.edges.index)
        adj = sc.sparse.coo_matrix(
            (
                np.ones(n_edges),
                (self.edges.start_node.to_numpy(), self.edges.end_node.to_numpy()),
            ),
            shape=[n_nodes, n_nodes],
        )

        n_components, labels = sc.sparse.csgraph.connected_components(
            csgraph=adj, return_labels=True
        )
        if n_components == 1:
            return

        largest_cc_label = np.argmax(np.unique(labels, return_counts=True)[1])

        drop_nodes = np.where(labels != largest_cc_label)[0]
        self.nodes = self.nodes.drop(drop_nodes)

        drop_edges = [
            edge_id
            for edge_id, edge in self.edges.iterrows()
            if edge.start_node in drop_nodes and edge.end_node in drop_nodes
        ]
        self.edges = self.edges.drop(drop_edges)


def convert_graph(  # pylint: disable=too-many-branches
    graph, node_features=None, label=None, graph_type=None
):
    """Function to convert different graph types to the class Graph.

    Args:
        graph (graph like object): valid graph data (NetowrkX or np.ndarray)
        label (int): label of the graph
        graph_type (str): set to 'directed' for directed graphs

    Returns:
        (Graph): converted Graph object
    """
    if node_features is not None:
        if isinstance(node_features, np.ndarray):
            node_features = node_features.tolist()

    if label is not None:
        label = [label]

    if isinstance(graph, nx.Graph):
        A = nx.to_scipy_sparse_array(graph, weight="weight", format="coo")

        if not node_features:
            try:
                node_features = list(nx.get_node_attributes(graph, "features").values())
                if isinstance(node_features[0], np.ndarray):
                    node_features = [x.tolist() for x in node_features]
            except Exception:  # pylint: disable=broad-except
                node_features = None

        if not label:
            try:
                graph_label = list(nx.get_node_attributes(graph, "label").values())
                if not isinstance(graph_label, list):
                    graph_label = [graph_label]
            except Exception:  # pylint: disable=broad-except
                label = None

        edges = np.array([A.row, A.col, A.data]).T
        edges_df = pd.DataFrame(edges, columns=["start_node", "end_node", "weight"])

        nodes = np.arange(0, A.shape[0])
        nodes_df = pd.DataFrame(index=nodes)

        if node_features is not None:
            nodes_df["attributes"] = node_features

        graph = Graph(nodes_df, edges_df, label, graph_type)

    if isinstance(graph, np.ndarray):
        graph = sc.sparse.coo_matrix(graph)
        edges = np.array([graph.row, graph.col, graph.data]).T
        edges_df = pd.DataFrame(edges, columns=["start_node", "end_node", "weight"])

        nodes = np.arange(0, graph.shape[0])
        nodes_df = pd.DataFrame(index=nodes)

        if node_features is not None:
            nodes_df["attributes"] = node_features

        graph = Graph(nodes_df, edges_df, label, graph_type)

    return graph
