#!/bin/bash


#need this is a virtualenv called hcga is used
#set -e
#module purge all
#. ~/hcga/bin/activate

export OMP_NUM_THREADS=1 #set to one to prevent numpy to run in parallel

# then run classification (this can be run separately, 
# as the features are saved from the previous script)
python3 run_classification.py $1
