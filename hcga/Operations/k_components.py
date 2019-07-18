

import networkx as nx
import numpy as np

class kComponents():
    """
    k components class
    """
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = []

    def feature_extraction(self):

        """
        Compute features related to the k components of the network
        
        A k_component is a subgraph such that every node in it is connected to
        at least k other nodes in the subgraph.

        Parameters
        ----------
        G : graph
          A networkx graph

        Returns
        -------
        feature_list : list
           List of features related to k components.
        
        Notes
        -----
        K components calculations using networkx:
            https://networkx.github.io/documentation/stable/reference/algorithms/approximation.html            
        """
        
        
        feature_names = ['num_k_components','max_k']

        G = self.G

        feature_list = []
        
        # Calculate the k_components
        k_components=nx.k_components(G)
        k_components_keys=np.asarray(list(k_components.keys()))
        k_components_vals=np.asarray(list(k_components.values()))
        
        # Calculate basic features related to k_components
        num_k=0
        for i in range(len(k_components_vals)):
            num_k=num_k+len(k_components_vals[i])
        
        max_k=max(k_components_keys)
        feature_list.append(num_k) 
        feature_list.append(max_k) 
        
        # Define other feature names
        s1='num_{}_components'.format(max_k)
        s2='mean_{}_component_size'.format(max_k)
        s3='largest_{}_component'.format(max_k)
        s4='smallest_{}_component'.format(max_k)
        feature_names.append(s1)
        feature_names.append(s2)
        feature_names.append(s3)
        feature_names.append(s4)
        
        # Calculate basic feature related to largest k component
        feature_list.append(len(k_components_vals[0]))
        l=[len(i) for i in k_components_vals[0]]
        feature_list.append(np.mean(l))
        feature_list.append(max(l))
        feature_list.append(min(l))
        
        self.feature_names = feature_names
        self.features = feature_list
