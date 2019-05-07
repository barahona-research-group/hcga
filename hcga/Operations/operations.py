# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 18:30:46 2019

@author: Rob
"""

import pandas as pd
import numpy as np
import yaml
from importlib import import_module
import os
import networkx as nx

from scipy.sparse import linalg
import scipy as sp

from tqdm import tqdm

class Operations():

    """
        Class that extracts all time-series features in a chosen YAML file
    """

    def __init__(self, G, YAMLfilename = 'operations.yaml'):
        
        self.G = G
        self.operations_dict = []
        self.YAMLfilename = YAMLfilename
        self.pre_computations = []
        self.feature_names = []
        self.feature_vals = []
        
        
        # pre computed values
        self.eigenvalues = []
        self.eigenvectors = []
        
        
        # functins to run automatically
        if not self.operations_dict:
            self.load_yaml()
            
        if not self.pre_computations:
            self.pre_compute()

    def load_yaml(self):

        module_dir = os.path.dirname(os.path.abspath(__file__))
        YAML_file_path = os.path.join(module_dir, self.YAMLfilename)

        with open(YAML_file_path, 'r') as stream:
            operations_dict = yaml.load(stream)

        self.operations_dict = operations_dict


    def pre_compute(self):
        
        """
        This function pre-computes a set of calculations that are often reused
        by various functions. This saves time by not pre-computing the same data
        for multiple functions.
        
        """
        if not self.eigenvalues:            
            self.precompute_eigenvectors()        

        


    def feature_extraction(self):
        operations_dict = self.operations_dict

        feature_names = []
        feature_vals = []

        # loop over the feature classes defined in the YAML file
        
        for i, key in enumerate(operations_dict.keys()):
            operation = operations_dict[key]

            #Extract the filename and class name
            main_params = ['filename','classname','shortname','keywords','calculation_speed','precomputed']
            filename = operation[main_params[0]]
            classname = operation[main_params[1]]
            symbolic_name = operation[main_params[2]]
            keywords = operation[main_params[3]]
            calculation_speed = operation[main_params[4]]
            precomputed = operation[main_params[5]]


            # Extracting all additional arguments if they exist
            params = []
            args = list(operation.keys() - main_params)
            for arg in args:
                params.append(operation[arg])


            # import the class from the file
            feature_obj = getattr(import_module('hcga.Operations.'+filename), classname)
            
                        
            if precomputed:
                feature_obj = feature_obj(self.G,self)
            else:
                feature_obj = feature_obj(self.G)

            if not params:
                feature_obj.feature_extraction()
            else:
                feature_obj.feature_extraction(params)


            # Alter the feature feature_names
            f_names = feature_obj.feature_names
            f_names_updated = [symbolic_name + '_' + f_name for f_name in f_names]

            # appending the parameter list onto the feature name
            param_string = '_'.join(map(str, params))
            if params:
                f_names_updated_params = [f_name + '_' + param_string for f_name in f_names_updated]
                f_names_updated = f_names_updated_params

            # Append the altered feature names and the feature list of values
            feature_names.append(f_names_updated)
            feature_vals.append(feature_obj.features)

        self.feature_vals = feature_vals
        self.feature_names = feature_names

    def _extract_data(self):

        features = self.feature_vals # features as list of lists
        feature_names = self.feature_names # feature names as list of lists

        features_flat = [item for sublist in features for item in sublist] # features as single list
        feature_names_flat = [item for sublist in feature_names for item in sublist] # feature names as single list

        return feature_names_flat, features_flat


    def precompute_eigenvectors(self,weight=None, max_iter=50, tol=0):        

        try:
            M = nx.to_scipy_sparse_matrix(self.G, nodelist=list(self.G), weight=weight,
                                      dtype=float)
            
            eigenvalues, eigenvectors = linalg.eigs(M.T, k = self.G.number_of_nodes() - 2, which='LR',
                                              maxiter=max_iter, tol=tol)            
            
            self.eigenvalues = eigenvalues
            self.eigenvectors = eigenvectors
        
        except:
            self.eigenvalues = []
            self.eigenvectors = []
            pass
        
        
        
        return

        