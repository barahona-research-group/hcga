"""Classes representing single and a collection of graphs."""
import networkx as nx
import numpy as np
import scipy as sc
import pandas as pd

MIN_NUM_NODES = 2
MIN_NUM_EDGES = 1


class GraphCollection:
    """Contain a list of graphs."""

    def __init__(self):
        """Set empty list of graphs."""
        self.graphs = []

    def add_graph(self, graph, node_features=None, label=None):
        """Add a graph to the list."""
        if not isinstance(graph, Graph):
            graph = convert_graph(graph, node_features, label)
            #raise Exception("Only add class Graph to GraphCollection")
        graph.id = len(self.graphs)
        self.graphs.append(graph)

    def add_graph_list(self, graph_list, node_features_list=None, graph_labels=None):
        """ Add a list of graphs """
        for i, graph in enumerate(graph_list):
            if not isinstance(graph, Graph):
                graph = convert_graph(graph, node_features_list[i], graph_labels[i])                
                #raise Exception("Only add class Graph to GraphCollection")      
            self.graphs.append(graph)

    def __iter__(self):
        """Makes this class iterable over the graphs."""
        self.current_graph = -1
        return self

    def __next__(self):
        """Get the next enabled graph."""
        self.current_graph += 1
        if self.current_graph >= len(self.graphs):
            raise StopIteration
        while self.graphs[self.current_graph].disabled:
            self.current_graph += 1
            if self.current_graph >= len(self.graphs):
                raise StopIteration
        return self.graphs[self.current_graph]

    def __len__(self):
        """Overrites the len() function to get number of enabled graphs."""
        return sum([1 for graph in self.graphs if not graph.disabled])

    def get_n_node_features(self):
        """Get the number of features of the nodes."""
        n_node_features = self.graphs[0].n_node_features
        for graph in self.graphs:
            assert n_node_features == graph.n_node_features
        return n_node_features

    def get_num_disabled_graphs(self):
        """Get the number of disabled graphs."""
        return len(self.graphs) - self.__len__()

    def remove_node_features(self):
        """Remove the node features."""
        for graph in self.graphs:
            for node_id, node in enumerate(graph.nodes):
                graph.nodes[node_id] = node[0]
            graph.set_n_node_features()

    def remove_edge_weights(self):
        """ remove edge weights. """        
        for graph in self.graphs:
            graph.remove_weights()

    def get_graph_ids(self):
        """Get the list of active graph ids."""
        return [graph.id for graph in self.graphs if not graph.disabled]

    def maximal_subgraphs(self):
        """Overwrites each graph with its maximal subgraph."""
        print("Returning the maximal subgraph for each graph")

        for graph in self.graphs:
            graph.maximal_subgraph()


class Graph:
    """Class to encode a generic graph structure for hcga."""

    def __init__(self, nodes, edges, label, graph_type=None, label_name=None):
        """Set main graphs quantities.

        Args:
            nodes (DataFrame): node dataframe, index as node id, and optional
                label and attributes columns (with lists elements)
            edges (DataFrame): edge dataframe, with two columns 'start_node' and 'end_node'
                with id corresponding to indices in nodes. And a third optional column 'weight'
                which if absent all edges will default to weight 1.
            label (int): label of the graph, it has to be an integer
            label_name (any): name or other information on the graph label
        """
        nodes["new_index"] = np.arange(0, len(nodes.index))
        edges["start_node"] = nodes.new_index[edges["start_node"].to_list()].to_list()
        edges["end_node"] = nodes.new_index[edges["end_node"].to_list()].to_list()

        if "weight" not in edges:
            edges["weight"] = 1.0

        self.nodes = nodes.set_index("new_index")
        self.edges = edges.reset_index()

        self.label = label
        self.graph_type = graph_type
        self.label_name = label_name
        self.disabled = False
        self.id = -1

        self._check_length()
        self.set_node_features()

    def set_node_features(self):
        """Set the node features and get the dimension of features."""
        if "features" in self.nodes:
            self.n_node_features = len(self.nodes.features.iloc[0])
        else:
            if "labels" in self.nodes:
                features_lab = np.asarray(self.nodes["labels"].to_list())
            if "attributes" in self.nodes:
                features_attr = np.asarray(self.nodes["attributes"].to_list())
            if "labels" in self.nodes and "attributes" in self.nodes:
                features = np.concatenate((features_lab, features_attr), axis=1)
            if "labels" in self.nodes and "attributes" not in self.nodes:
                features = features_lab
            if "labels" not in self.nodes and "attributes" in self.nodes:
                features = features_attr
            if "labels" in self.nodes or "attributes" in self.nodes:
                self.nodes["features"] = list(features)
                self.n_node_features = len(features[0])
            else:
                self.n_node_features = 0

    def remove_weights(self):
        """ set edge weights to one """
        self.edges['weight'] = 1.0
    
    def _check_length(self):
        """Verify if the graph is large enough to be considered."""
        if len(self.nodes.index) <= MIN_NUM_NODES:
            self.disabled = True
        if len(self.edges.index) <= MIN_NUM_EDGES:
            self.disabled = True

    def get_graph(self, encoding=None):
        """Get usable graph structure with given encoding.
        For now, only networkx is implemented."""
        if encoding is None:
            return self
        if encoding == "networkx":
            if not hasattr(self, "_graph_networkx"):
                self.set_networkx()
            return self._graph_networkx
        if encoding == "igraph":
            raise Exception("Igraph is not yet implemented")
        raise Exception("Graph encoding not understood")

    def set_networkx(self):
        """Set the networkx graph encoding."""
        if self.graph_type is None:
            self._graph_networkx = nx.Graph()
        elif self.graph_type == "directed":
            self._graph_networkx = nx.DiGraph()
        else:
            raise Exception("graph type not reconised or not yet implemented")
        if self.n_node_features == 0:
            nodes = [(node, {"feat": [0]}) for node, node_data in self.nodes.iterrows()]
        else:
            nodes = [
                (node, {"feat": node_data["features"]})
                for node, node_data in self.nodes.iterrows()
            ]
        self._graph_networkx.add_nodes_from(nodes)

        edges = [
            (edge["start_node"], edge["end_node"], edge["weight"])
            for _, edge in self.edges.iterrows()
        ]
        self._graph_networkx.add_weighted_edges_from(edges)

    def maximal_subgraph(self):
        """Overwrite the graph with its maximal subgraph."""
        n_nodes = len(self.nodes.index)
        n_edges = len(self.edges.index)
        adj = sc.sparse.coo_matrix(
            (
                np.ones(n_edges),
                (self.edges.start_node.to_numpy(), self.edges.end_node.to_numpy()),
            ),
            shape=[n_nodes, n_nodes],
        )

        n_components, labels = sc.sparse.csgraph.connected_components(
            csgraph=adj, return_labels=True
        )
        if n_components == 1:
            return

        largest_cc_label = np.argmax(np.unique(labels, return_counts=True)[1])

        drop_nodes = np.where(labels != largest_cc_label)[0]
        self.nodes = self.nodes.drop(drop_nodes)

        drop_edges = [
            edge_id
            for edge_id, edge in self.edges.iterrows()
            if edge.start_node in drop_nodes and edge.end_node in drop_nodes
        ]
        self.edges = self.edges.drop(drop_edges)

def convert_graph(g, node_features=None, label=None):
    """ Function to convert different graph types to the class Graph """
    
    if node_features is not None:
        if isinstance(node_features,np.ndarray):
            node_features = node_features.tolist()

    if label is not None:
        label = [label]            
    
    if isinstance(g, nx.Graph):
        A = nx.to_scipy_sparse_matrix(g, weight='weight', format='coo')
        
        if not node_features:
            try:
                node_features = list(nx.get_node_attributes(g,'features').values())
                if isinstance(node_features[0],np.ndarray):
                    node_features = [x.tolist() for x in node_features]
            except:
                node_features = None
                
                
        if not label:
            try:
                graph_label = list(nx.get_node_attributes(g,'label').values())
                if not isinstance(graph_label, list):
                    graph_label = [graph_label]       
            except:
                raise Exception('Graph label not present. Try renaming networkx attribute to \'label\'')
                         
        
        edges = np.array([A.row,A.col,A.data]).T
        edges_df = pd.DataFrame(edges, columns = ['start_node', 'end_node', 'weight'])

        nodes = np.arange(0,A.shape[0])
        nodes_df = pd.DataFrame(index=nodes) 
        
        if node_features is not None:
            nodes_df['attributes'] = node_features
        
        graph = Graph(nodes_df, edges_df, label)
                

    if isinstance(g, np.ndarray):
        g = sc.sparse.coo_matrix(g)
        edges = np.array([g.row,g.col,g.data]).T
        edges_df = pd.DataFrame(edges, columns = ['start_node', 'end_node', 'weight'])
        
        nodes = np.arange(0,g.shape[0])
        nodes_df = pd.DataFrame(index=nodes) 

        if node_features is not None:
            nodes_df['attributes'] = node_features    
            
        graph = Graph(nodes_df, edges_df, label)

    
    return graph

    
    
    
    
    
    
    
    
    
    
    
