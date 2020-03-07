#!/bin/bash

hcga feature_analysis -ff $1 -m 'shap' -c 'RF' --no-kfold
