import random
import networkx as nx
from Model.Cut import Cut
from Model.DataType import DataType
import numpy as np

class DataSetGraph(DataType):
    def __init__(self, agreement_param, cuts=[], search_tree=None):
        super().__init__(agreement_param, cuts, search_tree)
        self.G = None  # Attribute to hold the graph

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
        cuts = (len(G.nodes) // (self.agreement_param-1))
        self.cuts = []
        unique_cuts = set()  # Store unique cuts
        i = 0
        while len(unique_cuts) < cuts:
            if (i <= cuts//2):
                fraction = 0.5
            else:
                fraction = random.uniform(0.5, 1)
            initial_partition = self.initial_partition(G.nodes(), fraction=fraction)  # Adjusting fraction if needed
            partition = nx.algorithms.community.kernighan_lin_bisection(G, max_iter=2, partition=initial_partition, weight='weight', seed=None)
            cut = Cut()
            cut.A = partition[0]
            cut.Ac = partition[1]
            # Check if the cut is unique before adding it
            cut_tuple = (tuple(cut.A), tuple(cut.Ac))
            if cut_tuple not in unique_cuts and (tuple(cut.Ac), tuple(cut.A)) not in unique_cuts:
                unique_cuts.add(cut_tuple)
                self.cuts.append(cut)
            i += 1


    def initial_partition(self, nodes, fraction=0.5):
        """Generate an initial partition of nodes based on a fraction."""
        nb_vertices = len(nodes)
        partition = int(round(fraction * nb_vertices))
        A = np.zeros(nb_vertices, dtype=bool)
        A[np.random.choice(np.arange(nb_vertices), partition, replace=False)] = True
        B = np.logical_not(A)

        # Convert boolean arrays to sets of nodes
        node_list = list(nodes)
        setA = {node_list[i] for i, val in enumerate(A) if val}

        setB = {node_list[i] for i, val in enumerate(B) if val}

        return (setA, setB)

            

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
                if self.G.has_edge(u, v):  # More concise way to check for an edge
                    edge_weight_sum += self.G[u][v].get('weight', 0)

        # Calculate the cost based on the sum of edge weights
        cut_cost = edge_weight_sum / (A_size * (total_nodes - A_size))

        return cut_cost



    def order_function(self):
        """Return cuts in list of ascending order of the cost."""
        return sorted(self.cuts, key=lambda x: x.cost)