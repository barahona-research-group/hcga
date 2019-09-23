#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 10:41:52 2019

@author: robert
"""


import numpy as np
import networkx as nx


class SpectrumLaplacian():
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
        eigenvals_L = np.real(nx.linalg.spectrum.laplacian_spectrum(G))
        
        if len(eigenvals_L) < 10:
            eigenvals_L = np.concatenate((eigenvals_L,np.zeros(5-len(eigenvals_L))))        
            
        for i in range(10):               
            feature_list['L_eigvals_'+str(i)]=eigenvals_L[i]
        
        
        for i in range(10):
            for j in range(10):
                try:
                    feature_list['L_eigvals_ratio_'+str(i)+'_'+str(j)] = eigenvals_L[i]/eigenvals_L[j]
                except:
                    feature_list['L_eigvals_ratio_'+str(i)+'_'+str(j)] = np.nan
                
        feature_list['L_eigvals_min'] = min(eigenvals_L)




        

        self.features = feature_list
