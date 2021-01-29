#!/bin/bash

export OMP_NUM_THREADS=1  # set to one to prevent numpy to run in parallel

#echo 'Getting data'
#hcga get_data $1

echo 'Extracting features'
hcga extract_features datasets/$1.pkl -m fast -n 4 --timeout 0.1

echo 'Run classification'
hcga feature_analysis $1
