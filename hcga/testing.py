"""Testing suite for hcga"""

import unittest
import random
import numpy as np
import pandas as pd
import networkx as nx
import hcga.graphs as hcga_graphs

class TestingGraphMethods(unittest.TestCase):

    def test_calculate_features(self):
        """
        Testing of the method calculate_features from the class Graphs
        of hcga.graphs with synthetic data
        """
        hashsum_ref = -505139337749376602
        rgraphs, rlabels = synthetic_data_sbm(N=4, seed=42)
        g = hcga_graphs.Graphs(graphs=rgraphs, graph_class=rlabels)
        g.n_processes = 4
        g.calculate_features(calc_speed='veryfast', parallel = True)
        hashsum = pd.util.hash_pandas_object(g.graph_feature_matrix).sum()
        self.assertEqual(hashsum, hashsum_ref)


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

