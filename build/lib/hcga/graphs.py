import numpy as np
import networkx as nx

from hcga.utils import read_graphfile
from hcga.Operations.operations import Operations

from tqdm import tqdm


class Graphs():

    """
        Takes a list of graphs
    """

    def __init__(self, graphs = [], graph_meta_data = [], node_meta_data = [], graph_class = []):
        self.graphs = graphs # A list of networkx graphs
        self.graph_labels = graph_class # A list of class IDs - A single class ID for each graph.

        self.graph_metadata = graph_meta_data # A list of vectors with additional feature data describing the graph
        self.node_metadata = node_meta_data # A list of arrays with additional feature data describing nodes on the graph

        if not graphs:
            self.load_graphs()




    def load_graphs(self, graph_data_set = ''):


        directory = '/home/robert/Documents/PythonCode/hcga/hcga/TestData'
        dataname = 'ENZYMES'

        graphs,graph_labels = read_graphfile(directory,dataname)

        self.graphs = graphs
        self.graph_labels = graph_labels


    def calculate_features(self):
        """
        Calculation the features for each graph in the set of graphs

        """

        graph_feature_set = []

        for G in tqdm(self.graphs):
            G_operations = Operations(G)
            G_operations.feature_extraction()
            graph_feature_set.append(G_operations)


        self.calculated_graph_features = graph_feature_set
