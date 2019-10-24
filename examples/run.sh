#!/bin/bash

set -e

#need this is a virtualenv called hcga is used
module purge all
. ~/hcga/bin/activate

export OMP_NUM_THREADS=1 #set to one to prevent numpy to run in parallel

cd ./datasets/
unzip -o $1 
cd ..

# first extract features
python run_feature_extraction.py $1

# then run classification (this can be run separately, 
# as the features are saved from the previous script)
python run_classification.py $1
