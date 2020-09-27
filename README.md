# *hcga*: Highly comparative graph analysis

This is the official repository of *hcga*, a highly comparative graph analysis toolbox. It performs a massive feature extraction from a set of graphs, and applies supervised classification methods. 

<p align="center">
  <img src="doc/artwork/hcga_workflow.png" width="800" />
</p>

## Installation

Navigate to the main folder simply type:

```pip install .```

## Documentation

Head over to our [documentation](https://barahona-research-group.github.io/hcga/) to find out more about installation, data handling, creation of datasets and a full list of implemented features, transforms, and datasets.
For a quick start, check out our [examples](https://github.com/barahona-research-group/hcga/tree/master/examples) in the `examples/` directory.

## Cite

Please cite our paper if you use this code in your own work:

```
hcga: Highly Comparative Graph Analysis for network phenotyping
Robert L Peach, Alexis Arnaudon, Julia A Schmidt, Henry Palasciano, Nathan R Bernier, Kim Jelfs, Sophia Yaliraki, Mauricio Barahona
bioRxiv 2020.09.25.312926; doi: https://doi.org/10.1101/2020.09.25.312926 

```

## Run example

In the example folder, the script ``run_example.sh`` can be used to run the examples of the paper:
```./run_example.sh DATASET```
where ``DATASET`` is one of 
* ``ENZYMES``
* ``DD``
* ``COLLAB``
* ``PROTEINS``
* ``REDDIT-MULTI-12K``
