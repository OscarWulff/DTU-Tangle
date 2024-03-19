import networkx as nx
from Model.Cut import Cut
from Model.DataType import DataType

class DataSetGraph(DataType):
    def __init__(self, agreement_param, cuts=[], search_tree=None):
        super().__init__(agreement_param, cuts, search_tree)

    def initialize(self):
        G = nx.Graph()
        G.add_weighted_edges_from([(1, 2, 0.5), (1, 3, 10), (2, 3, 0.6), (3, 4, 0.7), 
                            (4, 5, 0.9), (5, 6, 0.3), (6, 7, 0.8), (7, 8, 1.2),
                            (8, 9, 0.4), (9, 10, 0.5), (10, 11, 0.6), (11, 12, 0.7),
                            (12, 13, 0.8), (13, 14, 0.9), (14, 15, 1.0), (15, 16, 1.1),
                             (16, 17, 1.2), (17, 18, 1.3), (18, 19, 1.4), (19, 20, 1.5),
                             (20, 21, 1.6), (21, 22, 1.7), (22, 23, 1.8), (23, 24, 1.9)])
        self.generate_multiple_cuts(G, max_iter=1, weight=None, seed=None)
        self.cost_function_Graph(G)

    def generate_multiple_cuts(self, G, max_iter=1, weight='weight', seed=None): 
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
        a = 5  # Adjust this value according to your needs
        for _ in range(a):
            partition = nx.algorithms.community.kernighan_lin_bisection(G, max_iter=max_iter, weight=weight, seed=seed)
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


# Example usage:
data_set_graph = DataSetGraph(agreement_param=3)
data_set_graph.initialize()
ordered_cuts = data_set_graph.order_function()
# for cut in ordered_cuts:
#     print(f"Cut A: {cut.A}, Cut Ac: {cut.Ac}, Cost: {cut.cost}")
