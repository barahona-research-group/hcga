#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 18:31:41 2019

@author: robert
"""

# test script that loads some enzymes data and runs it

from hcga.graphs import Graphs
import numpy as np
import os


cwd = os.getcwd()
g = Graphs(directory=cwd+'/TestData',dataset='COLLAB')

g.n_processes = 40

g.calculate_features(calc_speed='fast')

g.normalise_feature_data()

g.graph_classification()
