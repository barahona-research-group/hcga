#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 13:22:14 2020

@author: robert
"""

import sys
from make_data import make_benchmark_dataset, make_test_data

if len(sys.argv) > 1:
    dataset = sys.argv[1]    
else:
    sys.exit("Please provide name of dataset to build")
    
    
if dataset=='TESTDATA':
    print("--- Building test dataset and creating pickle ---")
    make_test_data()
else:
    print("---Downloading and creating pickle for {}---".format(dataset))
    make_benchmark_dataset(data_name=dataset)