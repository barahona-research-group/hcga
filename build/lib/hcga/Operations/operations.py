# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 18:30:46 2019

@author: Rob
"""

import pandas as pd
import numpy as np
import csv
from pprint import pprint
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

    def __init__(self, G, CSVfilename = 'operations.csv'):

        self.G = G
        self.operations_dict = []
        self.CSVfilename = CSVfilename
        self.pre_computations = []
        self.feature_names = []
        self.feature_vals = []
        self.feature_dict = {}


        # pre computed values
        self.eigenvalues = []
        self.eigenvectors = []


        # functins to run automatically
        if not self.operations_dict:
            self.load_csv()

        if not self.pre_computations:
            self.pre_compute()

    def load_csv(self):
        
        module_dir = os.path.dirname(os.path.abspath(__file__))
        CSV_file_path = os.path.join(module_dir, self.CSVfilename)


        with open(CSV_file_path, 'r', newline='') as csv_file:
            reader = csv.reader(line.replace('  ', ',') for line in csv_file)
            operations_dict = list(reader)
            operations_dict.pop(0)

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

        feature_dict = {}

        # loop over the feature classes defined in the YAML file

        for i in range(len(operations_dict)):
            operation = operations_dict[i]

            #Extract the filename and class name
            main_params = ['filename','classname','shortname','keywords','calculation_speed','precomputed']
            filename = operation[1]
            classname = operation[2]
            symbolic_name = operation[3]
            keywords = operation[4]
            calculation_speed = operation[5]
            precomputed = operation[6]


            # Extracting all additional arguments if they exist
            params = []   
            for arg in range(7,len(operation)):
                params.append(operation[arg])
                
            
            #print("Running file {} [{}/{}].".format(filename, i, len(operations_dict)))
            
            # import the class from the file            
            feature_class = getattr(import_module('hcga.Operations.'+filename), classname)           



            """# import the class from the file
            feature_class = getattr(import_module('hcga.Operations.'+filename), classname)"""



                        

            if precomputed=='True ':
                feature_obj = feature_class(self.G,(self.eigenvalues,self.eigenvectors))
            else:
                feature_obj = feature_class(self.G)

            if not params:
                feature_obj.feature_extraction()
            else:
                feature_obj.feature_extraction(params)

                        
            
            """
            # Alter the feature feature_names
            f_names = feature_obj.feature_names
            f_names_updated = [symbolic_name + '_' + f_name for f_name in f_names]
            """
            """
            # appending the parameter list onto the feature name
            param_string = '_'.join(map(str, params))
            if params:
                f_names_updated_params = [f_name + '_' + param_string for f_name in f_names_updated]
                f_names_updated = f_names_updated_params"""

            """
            # Append the altered feature names and the feature list of values
            feature_names.append(list(feature_obj.features.keys()))
            feature_vals.append(list(feature_obj.features.values()))
            """
            
            # Store features as a dictionary of dictionaries
            feature_dict[symbolic_name] = feature_obj.features
        
        
        self.feature_dict = feature_dict
        self.feature_vals = []
        self.feature_names = []

    def _extract_data(self):
        
        feature_dict = self.feature_dict
        feature_vals = [] 
        feature_names = [] 
        
        # Seperate out names and values from dictionary of features into 
        # feature_names and feature_vals
        symbolic_names=list(feature_dict.keys())
        features=list(feature_dict.values())
        for k in range(len(feature_dict)):
            names=list(features[k].keys())
            values=list(features[k].values())
            for l in range(len(names)):
                feature_names.append(symbolic_names[k]+'_'+names[l])
                feature_vals.append(values[l])
        
        """
        features_flat = [item for sublist in features for item in sublist] # features as single list
        feature_names_flat = [item for sublist in feature_names for item in sublist] # feature names as single list
        """
        
        return feature_names, feature_vals
        

    def precompute_eigenvectors(self,weight=None, max_iter=50, tol=0):

        try:
            M = nx.to_scipy_sparse_matrix(self.G, nodelist=list(self.G), weight=weight,
                                      dtype=float)

            eigenvalues, eigenvectors = linalg.eigs(M.T, k = self.G.number_of_nodes() - 2, which='LR',
                                              maxiter=max_iter, tol=tol)

            self.eigenvalues = eigenvalues
            self.eigenvectors = eigenvectors

        except:
            self.eigenvalues = np.array([1,1])
            self.eigenvectors = np.array([[1,1],[1,1]])
            pass




        return

