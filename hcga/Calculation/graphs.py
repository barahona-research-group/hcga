import numpy as np
import networkx as nx




class Graphs():

    """
        Takes a list of graphs
    """

    def __init__(self, graphs, graph_metadata = [], node_meta_data = [], graph_class = []):
        self.graphs = graphs # A list of networkx graphs
        self.graph_metadata = graph_metadata # A list of vectors with additional feature data describing the graph
        self.node_metadata = node_metadata # A list of arrays with additional feature data describing nodes on the graph
        self.graph_class = graph_class # A list of class IDs - A single class ID for each graph.
        self.is_directed = [] # A list of 0 or 1 if the graph is directed or not.

    def is_directed(self):
