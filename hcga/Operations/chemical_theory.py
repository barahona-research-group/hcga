import networkx as nx

class ChemicalTheory():
    """
    Chemical theory class
    """
 

    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = {}

    def feature_extraction(self):

        """Compute measures related to chemical theory.

        Parameters
        ----------
        G : graph
           A networkx graph

        Returns
        -------
        feature_list : dict
           Dictionary of features related to node connectivity.


        Notes
        -----


        """


        G = self.G

        feature_list = {}      

        # calculate wiener index using networkx
        feature_list['wiener_index'] = nx.wiener_index(G)   
        
        

        self.features = feature_list
