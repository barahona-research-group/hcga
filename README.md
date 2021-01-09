# *hcga*: Highly comparative graph analysis

This is the official repository of *hcga*, a highly comparative graph analysis toolbox. It performs a massive feature extraction from a set of graphs, and applies supervised classification methods. 

<p align="center">
  <img src="doc/artwork/hcga_workflow.png" width="800" />
</p>

Networks are widely used as mathematical models of complex systems across many scientific disciplines, not only in biology and medicine but also in the social sciences, physics, computing and engineering. Decades of work have produced a vast corpus of research characterising the topological, combinatorial, statistical and spectral properties of graphs. Each graph property can be thought of as a feature that captures important (and some times overlapping) characteristics of a network. In the analysis of real-world graphs, it is crucial to integrate systematically a large number of diverse graph features in order to characterise and classify networks, as well as to aid network-based scientific discovery. Here, we introduce *hcga*, a framework for highly comparative analysis of graph data sets that computes several thousands of graph features from any given network. *hcga* also offers a suite of statistical learning and data analysis tools for automated identification and selection of important and interpretable features underpinning the characterisation of graph data sets.

## Installation

For users that are not familiar with python and would like to use this code, we apologise that it isn't available in other languages. However, to help set you up with have provided a description of the steps required to setup python and its necessary dependencies if you scroll down to the bottom.

Navigate to the main folder simply type:

```
pip install .
```

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

The bibtex reference:
```
@article{peach2020hcga,
  title={hcga: Highly Comparative Graph Analysis for network phenotyping},
  author={Peach, Robert L and Arnaudon, Alexis and Schmidt, Julia A and Palasciano, Henry and Bernier, Nathan R and Jelfs, Kim and Yaliraki, Sophia and Barahona, Mauricio},
  journal={bioRxiv},
  year={2020},
  publisher={Cold Spring Harbor Laboratory}
}

```

## Run example

In the example folder, the script ``run_example.sh`` can be used to run the benchmark examples in the paper:
```
./run_example.sh DATASET
```
where ``DATASET`` is one of 
* ``ENZYMES``
* ``DD``
* ``COLLAB``
* ``PROTEINS``
* ``REDDIT-MULTI-5K``

Other examples can be found as jupyter-notebooks in `examples/` directory. We have included six examples:
* Example 1: Classification on synthetic data
* Example 2: Regression on synthetic data
* Example 3: Large Molecule dataset and regression
* Example 4: Training on labelled data, saving the fitted model, and predicting on unseen unlabelled data.
* Example 5: Pairwise classification. Exploring the similarity of classes.
* Example 6: Loading data in different ways.



## Our other available packages

If you are interested in trying our other packages, see the below list:
* [GDR](https://github.com/barahona-research-group/GDR) : Graph diffusion reclassification. A methodology for node classification using graph semi-supervised learning.
* [MSC](https://github.com/barahona-research-group/MultiscaleCentrality) : MultiScale Centrality: A scale dependent metric of node centrality.


## Python, Anaconda and hcga installation

The simplest setup will be to install [Anaconda](https://docs.anaconda.com/anaconda/install/). Anaconda is a package manager and contains useful IDEs for writing and viewing python scripts and notebooks. Choose from one of the following links below depending on your operating system:
* [Windows users](https://docs.anaconda.com/anaconda/install/windows/). Simply download the installer and make sure to register Anaconda3 as the default Python.
* [Mac users](https://docs.anaconda.com/anaconda/install/mac-os/). Perform the standard installation.
* [Linux users](https://docs.anaconda.com/anaconda/install/linux/). Linux will often require dependencies depending on your Linux distribution - these are described in the link.


Once Anaconda is installed you can open your command terminal (in any operating system), navigate into the *hcga* folder and simply type:

```
pip install .
```

The hcga package should now be installed directly into your Anaconda packages alongside other dependencies. 

If you want to run the example scripts then you need to open jupyter-notebook. Alternatively, you can run the example python scripts directly from the command line (see Run Example above). Thankfully jupyter-notebook is automatically installed with Anaconda. To open jupyter-notebook open a command terminal and type:
```
jupyter-notebook
```
You can then navigate to the examples folder and open the notebook of your choosing.







