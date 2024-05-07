import random
import networkx as nx
from Model.Cut import Cut
from Model.DataType import DataType

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
        # only generate a few initial cuts based on the size of the graph
        cuts = min(10, len(G.nodes))
        self.cuts = []
        unique_cuts = set()
        while len(self.cuts) < cuts:
            #initial_partition = generate_initial_partition(G)
            partition = nx.algorithms.community.kernighan_lin_bisection(G, max_iter=2, weight='weight', seed=None)
            cut = Cut()
            cut.A = partition[0]
            cut.Ac = partition[1]
            # check if cut is unique
            if (tuple(cut.A), tuple(cut.Ac)) not in unique_cuts and (tuple(cut.Ac), tuple(cut.A)) not in unique_cuts:
                unique_cuts.add((tuple(cut.A), tuple(cut.Ac)))
                self.cuts.append(cut)

            

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


def generate_initial_partition(G):
        """Generate initial partition randomly."""
        partition = ([], [])
        for node in G.nodes:
            partition[random.randint(0, 1)].append(node)
        return partition
