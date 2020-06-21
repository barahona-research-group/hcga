# *hcga*: Highly comparative graph analysis

This is the official repository of *hcga*, a highly comparative graph analysis toolbox. It performs a massive feature extraction from a set of graphs, and applies supervised classification methods. 

<p align="center">
  <img src="doc/artwork/hcga_workflow.png" width="800" />
</p>

## Installation

Navigate to the main folder simply type:

```pip install .```

## Documentation

See https://barahona-research-group.github.io/hcga/ for the documentation. 

## Cite

Please cite our paper if you use this code in your own work:

```
R. Peach, H. Palasciano, A. Arnaudon, M. Barahona, et al. “hcga: Highly Comparative Graph Analysis for graph phenotyping”, In preparation, 2019

```

## Run test file

In the example folder, the script ``run.py`` can be used to run the examples of the paper, simply get the dataset
``` hcga get_data DATASET```
then extract features with
```./run.sh .datasets/DATASET```
and run analysis suite with
```./analysis.sh DATASET```

In the paper, we used ``DATASET`` as one of 
* ``ENZYMES``
* ``DD``
* ``COLLAB``
* ``PROTEINS``
* ``REDDIT-MULTI-12K``

More comments are in the scripts for some parameters choices. 

