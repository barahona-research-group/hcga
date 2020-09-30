#!/bin/bash

export OMP_NUM_THREADS=1  # set to one to prevent numpy to run in parallel
hcga get_data $1
hcga extract_features datasets/$1.pkl -m fast -n -1 --timeout 10
hcga feature_analysis $1
