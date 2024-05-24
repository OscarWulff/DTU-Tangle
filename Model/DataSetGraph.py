import random
import networkx as nx
from sklearn.cluster import KMeans, SpectralClustering
from Model.Cut import Cut
from Model.DataType import DataType
import numpy as np

class DataSetGraph(DataType):
    def __init__(self, agreement_param, k, cuts=[], search_tree=None):
        super().__init__(agreement_param, cuts, search_tree)
        self.G = None  # Attribute to hold the graph
        self.k = k  # Number of clusters for spectral clustering

    def initialize(self):
        if self.G is None:
            raise ValueError("Graph G is not initialized. Please assign a graph to the G attribute.")
        
        self.generate_multiple_cuts(self.G)
        self.cost_function_Graph()

    def generate_multiple_cuts(self, G , initial_partition_method="K-Means"): 
        """ 
        Generate multiple cuts for the graph.
        
        Parameters:
        G (networkx.Graph): Graph
        """
        # Only generate a few initial cuts based on the size of the graph
        cuts = 10
        self.cuts = []
        unique_cuts = set()
        if initial_partition_method == "K-Means":
            bipartitions = self.generate_kmeans_cut(G)
            if bipartitions:
                for partition in bipartitions:
                    # Create a new cut object
                    cut = Cut()
                    cut.A = partition[0]
                    cut.Ac = partition[1]
                    self.cuts.append(cut)
        elif initial_partition_method == "Kernighan-Lin":
            while len(self.cuts) < cuts:
                partition = nx.algorithms.community.kernighan_lin_bisection(G, max_iter=2, weight='weight', seed=None)
                cut = Cut()
                cut.A = partition[0]
                cut.Ac = partition[1]
                # check if cut is unique
                if (tuple(cut.A), tuple(cut.Ac)) not in unique_cuts and (tuple(cut.Ac), tuple(cut.A)) not in unique_cuts:
                    unique_cuts.add((tuple(cut.A), tuple(cut.Ac)))
                    self.cuts.append(cut)
                else:
                    print("Duplicate cut found.")
        else:
            raise ValueError("Invalid initial partitioning method.")


    def generate_kmeans_cut(self, G):
        """
        Generate cuts using KMeans.
        
        Parameters:
        G (networkx.Graph): Graph
        
        Returns:
        list of tuples: Each tuple represents a bipartition (A, Ac)
        """
        # Convert graph to adjacency matrix
        adjacency_matrix = nx.convert_matrix.to_numpy_array(G)
        
        # Apply KMeans
        kmeans = KMeans(n_clusters=self.k)
        labels = kmeans.fit_predict(adjacency_matrix)
        
        # Generate bipartitions based on centroids
        bipartitions = []
        for i in range(self.k):
            # Nodes in the same cluster as the centroid
            partition_A = set(node for node, label in zip(G.nodes, labels) if label == i)
            # Nodes not in the same cluster as the centroid
            partition_Ac = set(G.nodes) - partition_A
            bipartitions.append((partition_A, partition_Ac))
        
        return bipartitions
    
    def cost_function_Graph(self):
        """ 
        Calculate the cost of cuts for Data Set Graph.
        """
        if self.G is None:
            raise ValueError("Graph G is not initialized.")
        
        for cut in self.cuts:
            # Initialize the cost to 0 before calculating
            cut.cost = self.calculate_cut_cost(cut)

    def calculate_cut_cost(self, cut):
        """ 
        Helper function to calculate the cost of cuts for Data Set Graph.
        
        Parameters:
        cut (Cut): Cut instance

        Returns:
        cost of the cut
        """
        if self.G is None:
            raise ValueError("Graph G is not initialized.")

        # Initialize cost for the current cut
        cut_cost = 0

        # Calculate the cost for the cut based on the described procedure
        A_size = len(cut.A)
        Ac_size = len(cut.Ac)
        total_nodes = A_size + Ac_size

        if A_size == 0 or Ac_size == 0:
            return 0

        # Calculate the sum of edge weights between nodes in A and nodes in Ac
        edge_weight_sum = 0
        for u in cut.A:
            for v in cut.Ac:
                # Check if there is an edge between nodes u and v
                if self.G.has_edge(u, v):
                    edge_weight_sum += 1  # Increment by 1 if edge exists

        # Calculate the cost based on the sum of edge weights
        if total_nodes - A_size != 0:
            cut_cost = edge_weight_sum / (A_size * (total_nodes - A_size))

        return cut_cost

    def order_function(self):
        """Return cuts in list of ascending order of the cost."""
        return sorted(self.cuts, key=lambda x: x.cost)

