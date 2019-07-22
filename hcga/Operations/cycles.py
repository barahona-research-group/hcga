
import pandas as pd
import numpy as np
import networkx as nx

class Cycles():
    """
    Cycles class
    """
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = []
        
    def feature_extraction(self):
        
        """Compute some simple cycle features of the network


        Parameters
        ----------
        G : graph
          A networkx graph

        Returns
        -------
        feature_list : list
           List of features related to cycles.
           
           
        Notes
        -----
        Cycles calculations using networkx:
            https://networkx.github.io/documentation/stable/reference/algorithms/cycles.html
        """
        
        """
        feature_names=['num_cycles','mean_cycle_length','shortest_cycle','longest_cycle']
        """
        
        G=self.G
        feature_list={}
        # Find list of cycles for graph
        cycles=nx.cycle_basis(G)
        # Add basic cycle features 
        feature_list['num_cycles']=len(cycles)
        l=[len(i) for i in cycles]
        feature_list['mean_cycle_length']=np.mean(l)
        feature_list['shortest_cycle']=min(l)
        feature_list['longest_cycle']=max(l)
        
        """
        self.feature_names = feature_names
        """
        
        self.features = feature_list
        