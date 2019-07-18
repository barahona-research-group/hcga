

import networkx as nx


class Efficiency():
    """
    Efficiency class
    """
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = []

    def feature_extraction(self):

        """
        Compute the efficiency of the network
        
        The efficiency of two nodes is the reciprocal of the length of the 
        shortest path between them.


        Parameters
        ----------
        G : graph
          A networkx graph

        Returns
        -------
        feature_list : list
           List of features related to efficiency.
        
        Notes
        -----
        Degree centrality calculations using networkx:
            https://networkx.github.io/documentation/stable/reference/algorithms/efficiency.html
        
        """


        feature_names = ['local_efficiency','global_efficiency']

        G = self.G

        feature_list = []
        
        #Efficiency calculations
        feature_list.append(nx.local_efficiency(G))
        feature_list.append(nx.global_efficiency(G))
        
        self.feature_names = feature_names
        self.features = feature_list
