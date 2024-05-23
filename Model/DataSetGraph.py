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

    def generate_multiple_cuts(self, G): 
        """ 
        Generate multiple cuts for the graph.
        
        Parameters:
        G (networkx.Graph): Graph
        """
        # only generate a few initial cuts based on the size of the graph
        cuts = 8
        self.cuts = []
        unique_cuts = set()
        while len(self.cuts) < cuts:
            # Generate partition using Spectral Clustering
            cut = self.generate_spectral_cut(G)
            if cut:
                # Check if the cut is unique
                if (tuple(cut.A), tuple(cut.Ac)) not in unique_cuts and (tuple(cut.Ac), tuple(cut.A)) not in unique_cuts:
                    unique_cuts.add((tuple(cut.A), tuple(cut.Ac)))
                    self.cuts.append(cut)
#        while len(self.cuts) < cuts:
#            #initial_partition = generate_initial_partition(G)
#            partition = nx.algorithms.community.kernighan_lin_bisection(G, max_iter=2, weight='weight', seed=None)
#            cut = Cut()
#            cut.A = partition[0]
#            cut.Ac = partition[1]
#            # check if cut is unique
#            if (tuple(cut.A), tuple(cut.Ac)) not in unique_cuts and (tuple(cut.Ac), tuple(cut.A)) not in unique_cuts:
#                unique_cuts.add((tuple(cut.A), tuple(cut.Ac)))
#                self.cuts.append(cut)


    def generate_kmeans_cut(self, G):
        """
        Generate a cut using KMeans.
        
        Parameters:
        G (networkx.Graph): Graph
        
        Returns:
        cut (Cut): Cut instance
        """
        # Convert graph to adjacency matrix
        adjacency_matrix = nx.to_numpy_array(G)
        
        # Apply KMeans
        kmeans = KMeans(n_clusters=2, random_state=random.randint(0, 100))
        labels = kmeans.fit_predict(adjacency_matrix)
        
        # Create a new cut
        cut = Cut()
        cut.A = [node for node, label in zip(G.nodes, labels) if label == 0]
        cut.Ac = [node for node, label in zip(G.nodes, labels) if label == 1]
        
        if len(cut.A) == 0 or len(cut.Ac) == 0:
            return None  # Invalid cut, skip
        
        return cut
    
    def generate_spectral_cut(self, G):
        """
        Generate a cut using Spectral Clustering.
        
        Parameters:
        G (networkx.Graph): Graph
        
        Returns:
        cut (Cut): Cut instance
        """
        # Convert graph to adjacency matrix
        adjacency_matrix = nx.to_numpy_array(G)
        
        # Apply Spectral Clustering
        sc = SpectralClustering(2, affinity='precomputed', n_init=10, random_state=random.randint(0, 100))
        labels = sc.fit_predict(adjacency_matrix)
        
        # Create a new cut
        cut = Cut()
        cut.A = [node for node, label in zip(G.nodes, labels) if label == 0]
        cut.Ac = [node for node, label in zip(G.nodes, labels) if label == 1]
        
        if len(cut.A) == 0 or len(cut.Ac) == 0:
            return None  # Invalid cut, skip
        
        return cut
    
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

# Helper function to generate initial partition using greedy modularity communities
def generate_initial_partition(G):
        """Generate initial partition randomly."""
        partition = ([], [])
        for node in G.nodes:
            partition[random.randint(0, 1)].append(node)
        return partition
