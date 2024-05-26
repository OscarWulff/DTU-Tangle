from itertools import permutations
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
        cuts = self.agreement_param // 2
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
        elif initial_partition_method == "K-Means-Half":
            bipartitions = self.generate_kmeans_half_cut(G)
            if bipartitions:
                for partition in bipartitions:
                    # Create a new cut object
                    cut = Cut()
                    cut.A = partition[0]
                    cut.Ac = partition[1]
                    self.cuts.append(cut)
        elif initial_partition_method == "K-Means-Both":
            bipartitions = self.generate_kmeans_both_methods(G)
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

    def generate_kmeans_half_cut(self, G):
        """
        Generate half/half cuts using KMeans.

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

        bipartitions = []
        centroids_list = list(range(self.k))
        
        # Maintain a set to store generated partitions
        generated_partitions = set()
        
        # Generate 10 unique half/half cuts
        for _ in range(10):
            # Select the first half of the centroids
            centroids_A = centroids_list[:self.k // 2]
            
            # Assign nodes associated with selected centroids to one partition
            partition_A = set(node for node, label in zip(G.nodes, labels) if label in centroids_A)
            
            # Assign the rest of the nodes to the complementary partition
            partition_Ac = set(G.nodes) - partition_A
            
            # Convert partitions to tuples for hashing
            partition_tuple = (frozenset(partition_A), frozenset(partition_Ac))
            
            # Check if the partition is unique
            if partition_tuple not in generated_partitions:
                bipartitions.append((partition_A, partition_Ac))
                generated_partitions.add(partition_tuple)
            
            # Rotate the centroids list by moving one position
            centroids_list = centroids_list[1:] + [centroids_list[0]]
        
        return bipartitions

    def generate_kmeans_both_methods(self, G):
        """
        Generate cuts using both KMeans strategies.

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

        # Strategy 1: One centroid on one side and the rest on the other
        for i in range(self.k):
            partition_A = set(node for node, label in zip(G.nodes, labels) if label == i)
            partition_Ac = set(G.nodes) - partition_A
            bipartitions.append((partition_A, partition_Ac))

        # Strategy 2: Half/Half
        generated_partitions = set()
        centroids_list = list(range(self.k))
        # Generate 10 unique half/half cuts
        for _ in range(10):
            # Select the first half of the centroids
            centroids_A = centroids_list[:self.k // 2]
            
            # Assign nodes associated with selected centroids to one partition
            partition_A = set(node for node, label in zip(G.nodes, labels) if label in centroids_A)
            
            # Assign the rest of the nodes to the complementary partition
            partition_Ac = set(G.nodes) - partition_A
            
            # Convert partitions to tuples for hashing
            partition_tuple = (frozenset(partition_A), frozenset(partition_Ac))
            
            # Check if the partition is unique
            if partition_tuple not in generated_partitions:
                bipartitions.append((partition_A, partition_Ac))
                generated_partitions.add(partition_tuple)
            
            # Rotate the centroids list by moving one position
            centroids_list = centroids_list[1:] + [centroids_list[0]]

        return bipartitions
    
    
    def cost_function_Graph(self, cost_function="Kernighan-Lin Cost Function"):
        """ 
        Calculate the cost of cuts for Data Set Graph.
        """
        if self.G is None:
            raise ValueError("Graph G is not initialized.")
        
        for cut in self.cuts:
            if cost_function == "Kernighan-Lin Cost Function":
                cut.cost = self.calculate_kernighan_lin_cost(cut)
            elif cost_function == "Modularity":
                cut.cost = self.calculate_modularity_cost(cut)
            elif cost_function == "Ratio Cut":
                cut.cost = self.calculate_edge_cut_cost(cut)
            else:
                raise ValueError("Invalid cost function.")

    def calculate_kernighan_lin_cost(self, cut):
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
    
    def calculate_edge_cut_cost(self, cut):
        """ 
        Helper function to calculate the edge cut cost for Data Set Graph.
        
        Parameters:
        cut (Cut): Cut instance

        Returns:
        cost of the cut
        """
        if self.G is None:
            raise ValueError("Graph G is not initialized.")

        # Calculate the sum of edge weights between nodes in A and nodes in Ac
        edge_cut_cost = sum(1 for u in cut.A for v in cut.Ac if self.G.has_edge(u, v))

        return edge_cut_cost

    def calculate_modularity_cost(self, cut):
        """ 
        Helper function to calculate the modularity cost for Data Set Graph.
        
        Parameters:
        cut (Cut): Cut instance

        Returns:
        cost of the cut
        """
        if self.G is None:
            raise ValueError("Graph G is not initialized.")

        # Initialize modularity cost
        modularity_cost = 0

        # Calculate total number of edges in the graph
        total_edges = self.G.number_of_edges()

        # Calculate the sum of edge weights between nodes in A and nodes in Ac
        for u in cut.A:
            for v in cut.Ac:
                if self.G.has_edge(u, v):
                    modularity_cost += 1

        # Calculate the expected number of edges between nodes in A and nodes in Ac
        expected_edges = (cut.A_size * cut.Ac_size) / (2 * total_edges)

        # Calculate modularity cost
        modularity_cost -= expected_edges

        return modularity_cost



    def order_function(self):
        """Return cuts in list of ascending order of the cost."""
        return sorted(self.cuts, key=lambda x: x.cost)

