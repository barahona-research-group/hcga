"""Testing suite for hcga"""

import unittest
import random
import numpy as np
import pandas as pd
import networkx as nx
import hcga.graphs as hcga_graphs
import zipfile


class TestingGraphMethods(unittest.TestCase):

    def setUp(self):
        rgraphs, rlabels = synthetic_data_sbm(N=4, seed=42)
        self.g = hcga_graphs.Graphs(graphs=rgraphs, graph_class=rlabels)
    
    def test_calculate_features(self):
        """
        Testing of the method calculate_features from the class Graphs
        of hcga.graphs with synthetic data
        """
        hashsum_ref = -505139337749376602
        
        self.g.n_processes = 4
        self.g.calculate_features(calc_speed='veryfast', parallel = True)
        hashsum = pd.util.hash_pandas_object(self.g.graph_feature_matrix).sum()
        self.assertEqual(hashsum, hashsum_ref)
        
    def test_normalise_feature_data(self):
        np.random.seed(42)
        hashsum_ref = 8994831284871540922
        feature_matrix = pd.DataFrame(np.random.uniform(1,100,size=(4, 400)))
        self.g.graph_feature_matrix = feature_matrix
        self.g.normalise_feature_data()
        hashsum = pd.util.hash_pandas_object(pd.DataFrame(self.g.X_norm)).sum()
        self.assertEqual(hashsum, hashsum_ref)
        
    def test_graph_classification(self):
        final_acc_ref = 0.7
        np.random.seed(42)
        hashsum_ref = -6561133957446616237
        n_graphs = 10
        feature_matrix = pd.DataFrame(np.random.uniform(1,100,size=(n_graphs, 400)))
        labels = [1]*(3) + [2]*(n_graphs-3)
        self.g.graph_feature_matrix = feature_matrix
        self.g.graph_labels = labels
        final_acc = self.g.graph_classification(reduc_threshold = 0.5, plot=False) 
        self.assertEqual(final_acc, final_acc_ref)

class TestingGraphLoading(unittest.TestCase):

    def test_load_graphs(self):
        summary_ref = 7406729
        directory = '../examples/datasets'
        with zipfile.ZipFile(directory + '/PROTEINS.zip',"r") as zip_ref:
            zip_ref.extractall(directory)
        g2 = hcga_graphs.Graphs(directory=directory, dataset='PROTEINS')
        summary = sum(x.number_of_edges() * x.number_of_nodes() for x in g2.graphs)
        self.assertEqual(summary, summary_ref)
    
def synthetic_data_sbm(N=1000, seed=None):
    if seed:
        random.seed(seed)
        np.random.seed(seed)

    graphs = []
    graph_labels = []
    prob_in = 0.9
    prob_out = 0.25

    for i in range(int(N/2)):
        G = nx.stochastic_block_model(
            [random.randint(2,3),random.randint(2,3),random.randint(2,3)],
            [[prob_in,prob_out,prob_out],[prob_out,prob_in,prob_out],[prob_out,prob_out,prob_in]])

        for u,v in G.edges:
            if len(G[u][v]) == 0:
                G[u][v]['weight'] = 1.
        graphs.append(G)
        graph_labels.append(1)

    for i in range(int(N/2)):
        G = nx.stochastic_block_model(
            [random.randint(2,3),random.randint(2,3)],
            [[prob_in, prob_out],[prob_out, prob_in]])
        for u,v in G.edges:
            if len(G[u][v]) == 0:
                G[u][v]['weight'] = 1.
        graphs.append(G)
        graph_labels.append(2)

    return graphs, np.asarray(graph_labels)


if __name__ == '__main__':
    unittest.main()

