
import networkx as nx

class Diameter():
    """
    Diameter class
    """
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = []

    def feature_extraction(self):
        """
        Compute diameter and radius of graph
        
        Parameters
        ----------
        G : graph
          A networkx graph

        Returns
        -------
        feature_list : list
           List containing diameter and radius of graph.
           
        Notes
        -----
        Diameter/radius calculations using networkx:
            https://networkx.github.io/documentation/stable/reference/algorithms/distance_measures.html
        """
        
        """
        feature_names = ['diameter','radius']
        """
        
        G = self.G
        feature_list = {}
        #Adding diameter and radius 
        feature_list['diameter']=nx.diameter(G)
        feature_list['radius']=nx.radius(G)
        
        """
        self.feature_names = feature_names
        """
        self.features = feature_list
        
        
