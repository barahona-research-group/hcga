

import networkx as nx

class MaximalMatching():
    """
    Maximal matching class
    """
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = {}

    def feature_extraction(self):

        """Compute the maximal matching of the network
        
        A matching is a subset of edges such that no edges are connected to
        the same node. It is maximal if the addition of another edge to this
        subset no longer makes it a matching.
        


        Parameters
        ----------
        G : graph
          A networkx graph

        Returns
        -------
        feature_list : list
           List of features related to the maximal matching.
           
        Notes
        -----
        Maximal matching calculations using networkx:
            `Networkx_maximal_matching <https://networkx.github.io/documentation/stable/reference/algorithms/matching.html>`_


        """

        """
        feature_names = ['num_edges','ratio']
        """

        G = self.G

        feature_list = {}
        
        E=nx.number_of_edges(G)
        # Calculate number of edges in maximal matching
        feature_list['num_edges']=len(nx.maximal_matching(G))
        # Calculate the total ratio of edges in the maximal matching to the
        #in the network
        feature_list['ratio']=len(nx.maximal_matching(G))/E
        
        """
        self.feature_names=feature_names
        """
        self.features = feature_list