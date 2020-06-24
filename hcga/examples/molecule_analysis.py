#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 11:24:08 2020

@author: robert
"""

from hcga.io import load_dataset

graphs = load_dataset('./datasets/custom_dataset_molecules.pkl')
    
#import hcga object
from hcga.hcga import Hcga

# define an object
h = Hcga()

#assigning the graphs field to the recently created dataset
h.graphs = graphs

# extracting all features here
h.extract(mode='slow',n_workers=12,timeout=20)

# saving all features into a pickle
h.save_features('./results/custom_dataset_molecules/all_features.pkl')
