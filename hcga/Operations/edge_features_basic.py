
import networkx as nx

from hcga.Operations.utils import summary_statistics

class EdgeFeaturesBasic():
    """
    Edge features class
    """    
    
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = {}
        
    def feature_extraction(self):
        """
        Compute basic statistics of the edge features.
        
        The edge features (usually just edge weights) provide an additional dimension to describe our graph/
        
        Parameters
        ----------
        G : graph
          A networkx graph


        Returns
        -------
        feature_list : dict
           Dictionary of features related to edge features.


        Notes
        -----
       
            
        

        
        """        
        G = self.G
        
        
        feature_list = {}
        

        edge_attributes = list(nx.get_edge_attributes(G,'weight').values())
        feature_list = summary_statistics(feature_list,edge_attributes,'edge_weights')       
        
        
        self.features=feature_list
