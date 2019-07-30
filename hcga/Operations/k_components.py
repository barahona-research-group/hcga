

import networkx as nx
import numpy as np

class kComponents():
    """
    k components class
    """
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = {}

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
            `Networkx_kcomponents <https://networkx.github.io/documentation/stable/reference/algorithms/approximation.html>`_            
        """
        
        """
        feature_names = ['num_k_components','max_k']
        """

        G = self.G

        feature_list = {}
        if not nx.is_directed(G):            
            # Calculate the k_components
            k_components=nx.k_components(G)
            k_components_keys=np.asarray(list(k_components.keys()))
            k_components_vals=np.asarray(list(k_components.values()))
            
            # Calculate basic features related to k_components
            num_k=0
            for i in range(len(k_components_vals)):
                num_k=num_k+len(k_components_vals[i])
            
            max_k=max(k_components_keys)
            feature_list['num_k_components']=num_k
            feature_list['max_k']=max_k
            
            # Calculate basic feature related to largest k component
            feature_list['num_max_k_components']=len(k_components_vals[0])
            l=[len(i) for i in k_components_vals[0]]
            feature_list['mean_max_k_component_size']=np.mean(l)
            feature_list['largest_max_k_component']=max(l)
            feature_list['smallest_max_k_component']=min(l)
        
        else:
            feature_list['num_k_components']=np.nan
            feature_list['max_k']=np.nan
            feature_list['num_max_k_components']=np.nan
            feature_list['mean_max_k_component_size']=np.nan
            feature_list['largest_max_k_component']=np.nan
            feature_list['smallest_max_k_component']=np.nan
            
        """
        self.feature_names = feature_names
        """
        self.features = feature_list
