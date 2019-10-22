

import networkx as nx
import numpy as np

class Efficiency():
    """
    Efficiency class
    """
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = {}

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
        feature_list : dict
           Dictionary of features related to efficiency.
        
        Notes
        -----
        Degree centrality calculations using networkx:
            `Networkx_efficiency <https://networkx.github.io/documentation/stable/reference/algorithms/efficiency.html>`_
        
        """


        G = self.G

        feature_list = {}
        
        if not nx.is_directed(G):            
            #Efficiency calculations
            feature_list['local_efficiency']=nx.local_efficiency(G)
            feature_list['global_efficiency']=nx.global_efficiency(G)
        else:
            feature_list['local_efficiency']=np.nan
            feature_list['global_efficiency']=np.nan
            

        self.features = feature_list
