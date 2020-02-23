#!/bin/bash

export OMP_NUM_THREADS=1 #set to one to prevent numpy to run in parallel

cd ./datasets/
unzip -o $1 
cd ..

hcga extract_features $1 -n 4

hcga feature_analysis
