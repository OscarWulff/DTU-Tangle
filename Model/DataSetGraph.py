import networkx as nx
import random

def generate_multiple_cuts(G, max_iter=1, weight='weight', seed=None): #Max iter is the number of iterations
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
            cut += G[u][v]['weight']
    return cut

def cost_function(G, cuts):
    return [cost_function_helper(G, cut) for cut in cuts]

# Example usage:
G = nx.Graph()
G.add_weighted_edges_from([(1, 2, 0.5), (1, 3, 10), (2, 3, 0.6), (3, 4, 0.7), 
                            (4, 5, 0.9), (5, 6, 0.3), (6, 7, 0.8), (7, 8, 1.2),
                            (8, 9, 0.4), (9, 10, 0.5), (10, 11, 0.6), (11, 12, 0.7),
                            (12, 13, 0.8), (13, 14, 0.9), (14, 15, 1.0), (15, 16, 1.1),
                             (16, 17, 1.2), (17, 18, 1.3), (18, 19, 1.4), (19, 20, 1.5),
                             (20, 21, 1.6), (21, 22, 1.7), (22, 23, 1.8), (23, 24, 1.9)])

cuts = generate_multiple_cuts(G)
cost = cost_function(G, cuts)

print(cuts)
print(cost)
