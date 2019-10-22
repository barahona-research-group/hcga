import numpy as np
import networkx as nx


class IndependentSets():
    """
    Independent sets class
    """
    
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = {}

    def feature_extraction(self):

        """Compute independent set measures.

        An independent set is a set of nodes such that the subgraph
        of G induced by these nodes contains no edges. A maximal
        independent set is an independent set such that it is not possible
        to add a new node and still get an independent set.

        Parameters
        ----------
        G : graph
           A networkx graph

        Returns
        -------
        feature_list :list
           List of features related to independent sets.


        Notes
        -----
        Eccentricity using networkx:
            `Networkx_maximal_indpendent_sets <https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.mis.maximal_independent_set.html#networkx.algorithms.mis.maximal_independent_set>`_   
        


        """
        

        G = self.G

        feature_list = {}
        
        if not nx.is_directed(G):            
    
    
            ind_set = nx.maximal_independent_set(G)
            
            feature_list['num_ind_nodes_norm']=len(ind_set)
            
            feature_list['ratio__ind_nodes_norm']=len(ind_set)/len(G)
        else:
            feature_list['num_ind_nodes_norm']=np.nan
            feature_list['ratio__ind_nodes_norm']=np.nan

        self.features = feature_list
