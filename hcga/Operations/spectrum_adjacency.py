#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 10:41:52 2019

@author: robert
"""


import numpy as np
import networkx as nx


class SpectrumAdjacency():
    """
    Force centrality class
    """
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = {}

    def feature_extraction(self):

        """Compute ...



        Parameters
        ----------
        G : graph
          A networkx graph




        Returns
        -------



        Notes
        -----



        """

        
        
        G = self.G

        feature_list = {}         
        
        
        
        
        
        # laplacian spectrum
        eigenvals_A = np.real(nx.linalg.spectrum.adjacency_spectrum(G))
        
        if len(eigenvals_A) < 10:
            eigenvals_A = np.concatenate((eigenvals_A,np.zeros(10-len(eigenvals_A))))        
            
        for i in range(10):               
            feature_list['A_eigvals_'+str(i)]=eigenvals_A[i]
        
        
        for i in range(10):
            for j in range(10):
                try:
                    feature_list['A_eigvals_ratio_'+str(i)+'_'+str(j)] = eigenvals_A[i]/eigenvals_A[j]
                except:
                    feature_list['A_eigvals_ratio_'+str(i)+'_'+str(j)] = np.nan
                
        feature_list['A_eigvals_min'] = min(eigenvals_A)
        


        

        self.features = feature_list
