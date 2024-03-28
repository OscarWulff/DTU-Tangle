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
        self.cost_function_Graph(self.G)

    def generate_multiple_cuts(self, G): 
        """ 
        Generate multiple cuts for the graph.
        
        Parameters:
        G (networkx.Graph): Graph
        max_iter (int): Maximum number of iterations
        weight (str): The edge attribute that holds the numerical value used as a weight. If None, then each edge has unit weight.
        seed (int): Seed for random number generator

        Return:
        cuts of the dataset
        """
        # cuts set to amount of nodes divided by two
        cuts = len(G.nodes)
        for _ in range(cuts):
            initial_partition = generate_initial_partition(G)
            partition = nx.algorithms.community.kernighan_lin_bisection(G, partition=initial_partition, max_iter=5, weight='weight', seed=None)
            cut = Cut()
            cut.A = partition[0]
            cut.Ac = partition[1]
            self.cuts.append(cut)

    def cost_function_Graph(self, G):
        """ 
        Calculate the cost of cuts for Data Set Graph.
        
        Parameters:
        G (networkx.Graph): Graph
        """
        for cut in self.cuts:
            # Initialize the cost to 0 before calculating
            cut.cost = cost_function_helper(G, [cut.A, cut.Ac])


    def order_function(self):
        """Return cuts in list of ascending order of the cost."""
        return sorted(self.cuts, key=lambda x: x.cost)

def cost_function_helper(G, partition):
    """ 
    Helper function to calculate the cost of cuts for Data Set Graph.
    
    Parameters:
    G (networkx.Graph): Graph
    partition (list): Partition of nodes

    Returns:
    cost of the cut
    """
    cut_cost = 0
    for u, v, weight in G.edges(data='weight'):
        if (u in partition[0] and v in partition[1]) or (u in partition[1] and v in partition[0]):
            # Increment the cut by the weight of the edge if the graph is weighted,
            # otherwise, increment the cut by 1 for each edge
            if weight is not None:
                cut_cost += weight
            else:
                cut_cost += 1
    return cut_cost

# , there is no obvious choice for the initial partitions in the pre-processing step
def generate_initial_partition(G):
    # Initialize an empty partition
    partition = ([], [])

    # Randomly assign each node to one of the two partitions
    for node in G.nodes:
        partition[random.randint(0, 1)].append(node)

    return partition