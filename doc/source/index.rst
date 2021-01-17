..  -*- coding: utf-8 -*-

.. _contents:

*hcga*: Highly Comparative Graph Analysis
#########################################

*hcga* is inspired by *hctsa*, the time serie massive feature extraction framework of [Hctsa]_ and instead extracts features from graphs to classify or regress them.  
This code comes with the companion paper [Hcga]_  containing more details and examples of applications. 

Installation
************

To install the dev version from `GitHub <https://github.com/imperialcollegelondon/hcga/>`_ with the commands::

       $ git clone git@github.com:barahona-research-group/hcga.git 
       $ cd hcga
       $ pip install .

There is no PyPi release version yet, but stay tuned for more!

Usage
*****

*hcga* has three main steps:

1. create dataset of graphs
2. extract features from them
3. use the features for classification of other analysis

1. Create a dataset
--------------------

Benchmarks datasets from `Graphkernel <https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets>`_ can be loaded directly with::

    $ hcga get_data DATASET

where for example, ``DATASET`` can be one of 
    * ENZYMES
    * DD
    * COLLAB
    * PROTEINS
    * REDDIT-MULTI-12K

To create custom dataset, one must load it in the ``hcga.graph.GraphCollection`` data structure as follow::

    import pandas as pd

    from hcga.graph import Graph, GraphCollection
    from hcga.io import save_dataset

    # create edges dataframe with columns names (required)
    edges = [[0, 1], [1, 2], [2, 0]]  # list of edges with node indices
    edges_df = pd.DataFrame(edges, columns = ['start_node', 'end_node'])

    # create nodes dataframe
    nodes = [0, 1, 2]
    nodes_df = pd.DataFrame(index=nodes)

    # optional: node labels and attributes
    # set labels, use one-hot vector (should be a list for each node)
    nodes_df['labels'] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  
    # any list for each node (should be of the same length)
    nodes_df['attributes'] = [[0.1, 0.2, 0.3], [0.2, 0.3, 0.1], [0.3, 0.2, 0.1]]

    # create graph colllection object
    graphs = GraphCollection()

    # add new graph to the collection
    graph_label = 1  # graph_label should be an integer
    graphs.add_graph(Graph(nodes_df, edges_df, graph_label))

    # save dataset to file './dataset.pkl'
    save_dataset(graphs, 'dataset', folder='.')

2. Extract features
--------------------

Once a dataset is created, features can be extracted using for example::

    $ hcga extract_features dataset.pkl --mode fast --timeout 10 --n-workers 4 

we refer to the hcga app documentation for more details, but the options here mean:

- ``--mode fast``: only extract simple features (other options include ``medium/slow`` 
- ``--timeout 10``: stop features computation after 10 seconds (this prevents some features to get stuck)
- ``--n-workers 4``: set the number of workers in multiprocessing
- ``--runtime``: this option runs a small set of graphs and ouput estimated times for each feature


3. Classifiy graphs
--------------------

Finally, to use the extracted features to classify graphs with respect to their labels, one use::

    $ hcga feature_analysis dataset --interpretability 1

where ``dataset`` is the name of the dataset, and ``--interpretability 1`` selects the features with all interpretabilities. Choices range from ``1-5``, where ``5`` only uses most interpretable features.

Citing
******

To cite *hcga*, please use [Hcga]_. 

Credits
*******

The code is still a preliminary version, and written by us.

Original authors:
*****************

- Robert Peach, GitHub: `peach-lucien <https://github.com/peach-lucien>`_
- Alexis Arnaudon, GitHub: `arnaudon <https://github.com/arnaudon>`_

Contributors:
*************

- Henry Palasciano, GitHub: `henrypalasciano <https://github.com/henrypalasciano>`_
- Nathan Bernier, GitHub: `nrbernier <https://github.com/nrbernier>`_

Any contributors are welcome, please contact us if interested. 

Bibliography
************

.. [Hcga]  R. Peach, H. Palasciano, A. Arnaudon, M. Barahona, 
                “hcga: Highly Comparative Graph Analysis for graph phenotyping”, In preparation, 2019
.. [Hctsa]  B. D. Fulcher and N. S. Jones, 
                “hctsa:  A computational framework for automated time-series phenotyping using massive feature extraction,” Cell systems, vol. 5, no. 5, pp. 527–531, 2017

API documentation
*****************

Documentation of the API of *hcga*. 

.. toctree::
   :maxdepth: 3

   app
   hcga
   graph
   feature_class 
   extraction
   analysis
   Features


Indices and tables
******************

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
