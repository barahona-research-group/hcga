
import numpy as np
import networkx as nx

class Cycles():
    """
    Cycles class
    """
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = {}
        
    def feature_extraction(self):
        
        """Compute some simple cycle features of the network


        Parameters
        ----------
        G : graph
          A networkx graph

        Returns
        -------
        feature_list : dict
           Dictionary of features related to cycles.
           
           
        Notes
        -----
        Cycles calculations using networkx:
            `Networkx_cycles <https://networkx.github.io/documentation/stable/reference/algorithms/cycles.html>`_
        """
        

        G=self.G
        feature_list={}
        if not nx.is_directed(G) and nx.cycle_basis(G):
            # Find list of cycles for graph
            cycles=nx.cycle_basis(G)
            # Add basic cycle features 
            feature_list['num_cycles']=len(cycles)
            l=[len(i) for i in cycles]
            feature_list['mean_cycle_length']=np.mean(l)
            feature_list['shortest_cycle']=min(l)
            feature_list['longest_cycle']=max(l)
        else:
            feature_list['num_cycles']=np.nan
            feature_list['mean_cycle_length']=np.nan
            feature_list['shortest_cycle']=np.nan
            feature_list['longest_cycle']=np.nan
            

        self.features = feature_list
        