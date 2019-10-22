

import numpy as np
import networkx as nx
from hcga.Operations import utils


class Sparsification():
    """
    Sparsification class
    """
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = {}

    def feature_extraction(self):

        r"""
        
        Parameters
        ----------
        G : graph
           A networkx graph

        Returns
        -------
        feature_list : dict
           Dictionary of features related to sparsification.


        Notes
        -----

        References
        ----------


        """

        
        G = self.G

        feature_list = {}


       
        self.features = feature_list
