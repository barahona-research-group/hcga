#!/bin/bash

export OMP_NUM_THREADS=1  # set to one to prevent numpy to run in parallel

echo 'Getting data'
#hcga -v get_data $1

echo 'Extracting features'
hcga -vvv extract_features datasets/$1.pkl -m fast -n 5  --timeout 10.0

echo 'Run classification'
hcga -v feature_analysis $1
