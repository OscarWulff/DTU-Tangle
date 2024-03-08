import networkx as nx

def generate_multiple_cuts(G, max_iter=1, weight='weight', seed=None): 
    """ 
    This function is used to generate the cuts for binary questionnaires data set
    parameters:
    G (networkx.Graph): Graph
    max_iter (int): Maximum number of iterations
    weight (str): The edge attribute that holds the numerical value used as a weight. If None, then each edge has unit weight.
    seed (int): Seed for random number generator

    Return:
    cuts of the dataset
    """
    a = 5  # Adjust this value according to your needs
    cuts = []
    for _ in range(a):
        partition = nx.algorithms.community.kernighan_lin_bisection(G, max_iter=max_iter, weight=weight, seed=seed)
        partition_sets = [{node for node in partition[0]}, {node for node in partition[1]}]
        cuts.append(partition_sets)
    return cuts

def cost_function_helper(G, partition):
    cut = 0
    for u, v in G.edges():
        if (u in partition[0] and v in partition[1]) or (u in partition[1] and v in partition[0]):
            # For unweighted graph, increment the cut by 1 for each edge
            cut += 1
    return cut


def cost_function_Graph(G, cuts):
    """ 
    This function is used to calculate the cost of cuts for Data Set Graph
    parameters:
    G (networkx.Graph): Graph
    cuts (list): List of cuts of the dataset

    Returns:
    cost of each cut
    """

    return [cost_function_helper(G, cut) for cut in cuts]



# Example usage:
G = nx.Graph()
G.add_weighted_edges_from([(1, 2, 0.5), (1, 3, 10), (2, 3, 0.6), (3, 4, 0.7), 
                            (4, 5, 0.9), (5, 6, 0.3), (6, 7, 0.8), (7, 8, 1.2),
                            (8, 9, 0.4), (9, 10, 0.5), (10, 11, 0.6), (11, 12, 0.7),
                            (12, 13, 0.8), (13, 14, 0.9), (14, 15, 1.0), (15, 16, 1.1),
                             (16, 17, 1.2), (17, 18, 1.3), (18, 19, 1.4), (19, 20, 1.5),
                             (20, 21, 1.6), (21, 22, 1.7), (22, 23, 1.8), (23, 24, 1.9)])

G1 = nx.Graph()
G1.add_edges_from([(1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8), (8,9), (9,10), (10,11), (11,12), (12,13), (13,14), (14,15), (15,16), (16,17), (17,18), (18,19), (19,20), (20,21), (21,22), (22,23), (23,24)])

cuts = generate_multiple_cuts(G1)
cost = cost_function_Graph(G1, cuts)

print(cuts)
print(cost)
