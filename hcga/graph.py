"""Classes representing single and a collection of graphs."""
import networkx as nx
import numpy as np

MIN_NUM_NODES = 2


class GraphCollection:
    """Contain a list of graphs."""

    def __init__(self):
        """Set empty list of graphs."""
        self.graphs = []

    def add_graph(self, graph):
        """Add a graph to the list."""
        if not isinstance(graph, Graph):
            raise Exception("Only add class Graph to GraphCollection")
        graph.id = len(self.graphs)
        self.graphs.append(graph)

    def __iter__(self):
        """Makes this class iterable over the graphs."""
        self.current_graph = -1
        return self

    def __next__(self):
        """Get the next enabled graph."""
        self.current_graph += 1
        if self.current_graph < len(self.graphs):
            while self.graphs[self.current_graph].disabled:
                self.current_graph += 1
            return self.graphs[self.current_graph]
        raise StopIteration

    def __len__(self):
        """Overrites the len() function to get number of enabled graphs."""
        if not hasattr(self, "len"):
            self.len = sum([1 for graph in self.graphs if not graph.disabled])
        return self.len

    def get_n_node_features(self):
        """Get the number of features of the nodes."""
        n_node_features = self.graphs[0].n_node_features
        for graph in self.graphs:
            assert n_node_features == graph.n_node_features
        return n_node_features

    def get_num_disabled_graphs(self):
        """Get the number of disabled graphs."""
        return len(self.graphs) - self.__len__()


class Graph:
    """Class to encode various graph structures."""

    def __init__(self, nodes, edges, label):
        """Set main graphs quantities."""
        self.nodes = nodes
        self.edges = edges
        self.label = label
        self.disabled = False
        self.id = -1

        self._check_length()
        self._get_n_node_features()

    def _get_n_node_features(self):
        """Get the number of node features."""
        if len(np.shape(self.nodes)) == 1:
            self.n_node_features = 0
        elif len(np.shape(self.nodes)) == 2:
            self.n_node_features = len(self.nodes[0][1])
        else:
            raise Exception("Too many elements for node data")

    def _check_length(self):
        """Verify if the graph is large enough to be considered."""
        if len(self.nodes) <= MIN_NUM_NODES:
            self.disabled = True
        if len(self.edges) == 0:
            self.disabled = True

    def get_graph(self, encoding=None):
        """Get usable graph structure with given encoding."""
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
        self._graph_networkx = nx.Graph()
        if self.n_node_features == 0:
            nodes = [(node, {"feat": [0]}) for node in self.nodes]
        else:
            nodes = [(node, {"feat": feat}) for node, feat in self.nodes]
        self._graph_networkx.add_nodes_from(nodes)

        if len(self.edges[0]) == 2:
            edges = [(edge[0], edge[1], 1.0) for edge in self.edges]
        elif len(self.edges) == 3:
            edges = self.edges
        else:
            raise Exception("Too many elements for edge data")
        self._graph_networkx.add_weighted_edges_from(edges)
