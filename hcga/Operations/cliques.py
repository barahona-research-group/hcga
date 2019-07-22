
from networkx.algorithms import clique
from hcga.Operations import utils
import numpy as np


class Cliques():
    """
    Cliques class
    """
 
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = []

    def feature_extraction(self):
        """
        Compute the clique number for the graph and the number of maximal 
        cliques
        
        A clique is the maximal subset of nodes such that all the nodes in 
        this set are connected to each other. 
        The clique number is the size of the largest clique in a graph.
        
        
        Parameters
        ----------
        G : graph
          A networkx graph


        Returns
        -------
        feature_list :list
           List of features related to cliques.


        Notes
        -----
        Clique number calculations using networkx:
            https://networkx.github.io/documentation/stable/reference/algorithms/clique.html
        """        

        """
        # Defining featurenames
        feature_names = ['clique_number','number_of_cliques',]
        """
        
        G = self.G

        feature_list = {}
        
        #Calculate the clique number for the graph
        feature_list['clique_number']=clique.graph_clique_number(G)
        
        #Calculate the number of maximal cliques in the graph
        feature_list['number_of_cliques']=clique.graph_number_of_cliques(G)
        
        """
        self.feature_names=feature_names
        """
        self.features = feature_list