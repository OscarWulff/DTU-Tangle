import numpy as np
import networkx as nx
from networkx.algorithms.community.kernighan_lin import kernighan_lin_bisection
from sklearn.metrics.pairwise import pairwise_distances

def cost_function_graph_based(graph, cut):
    """
    This function computes the implicit order of a cut based on Gaussian kernel distance.

    Parameters
    ----------
    graph : networkx.Graph
        The graph object.
    cut : set
        A set of node IDs representing the cut.

    Returns
    -------
    expected_order : float
        The average order for the cut.
    """
    # Convert cut to a list for indexing purposes
    cut_list = list(cut)
    
    # Extract subgraph induced by nodes in the cut
    subgraph = graph.subgraph(cut_list)
    
    # Compute pairwise distances between nodes in the subgraph using node attributes
    distances = []
    for u in cut_list:
        for v in cut_list:
            if u != v:
                distance = nx.shortest_path_length(graph, source=int(u), target=int(v))
                distances.append(distance)
    
    # Compute cost function based on distances
    cost = np.mean(distances)
    
    return cost

    
def cut_costs_for_graph(G, unique_cuts):
    """
    Calculate the costs for each unique cut for a given graph.

    Parameters:
    G (NetworkX graph): The graph dataset.
    unique_cuts (list): List of unique cuts of the dataset.

    Returns:
    list of costs for each unique cut.
    """
    cut_costs = []
    for cut in unique_cuts:
        # Convert frozenset back to set of node IDs
        cut = set().union(*cut)
        cut_costs.append(cost_function_graph_based(G, cut))
    return cut_costs

def cut_generator_graph_based(G):
    """
    This function generates a number of unique cuts for a graph dataset in a smart way,
    avoiding repeating the same cut.

    Parameters:
    G (NetworkX graph): The graph dataset.

    Returns:
    list of unique cuts of the dataset.
    """

    # Set to store unique cuts
    unique_cuts = set()

    # Determine the number of nodes in the graph
    num_nodes = len(G)

    # Set the number of cuts based on the number of nodes
    num_cuts = min(5, max(1, int(num_nodes / 10)))  # Set a maximum of 5 cuts or 10% of nodes, whichever is smaller

    # Set the maximum number of iterations based on the graph size
    iterations = max(1, int(num_nodes / 10))  # Adjust the divisor for more aggressive iteration scaling if needed

    # Iterate to generate unique cuts
    while len(unique_cuts) < num_cuts:
        # Partition the graph using Kernighan-Lin bisection
        if any('weight' in G[u][v] for u, v in G.edges()):
            # If the graph has weighted edges
            partition = kernighan_lin_bisection(G, max_iter=iterations, weight='weight')
        else:
            # If the graph has no weights
            partition = kernighan_lin_bisection(G, max_iter=iterations)
        # Convert partition to a frozenset to make it hashable
        frozen_partition = frozenset(map(frozenset, partition))

        # Add the unique cut to the list
        unique_cuts.add(frozen_partition)

    return list(unique_cuts)

# Create a larger weighted graph
G1 = nx.Graph()
G1.add_weighted_edges_from([(1, 2, 0.5), (1, 3, 10), (2, 3, 0.6), (3, 4, 0.7), 
                            (4, 5, 0.9), (5, 6, 0.3), (6, 7, 0.8), (7, 8, 1.2),
                            (8, 9, 0.4), (9, 10, 0.5), (10, 11, 0.6), (11, 12, 0.7),
                            (12, 13, 0.8), (13, 14, 0.9), (14, 15, 1.0), (15, 16, 1.1),
                             (16, 17, 1.2), (17, 18, 1.3), (18, 19, 1.4), (19, 20, 1.5),
                             (20, 21, 1.6), (21, 22, 1.7), (22, 23, 1.8), (23, 24, 1.9)])
# Generate unique cuts for the graph
unique_cuts1 = cut_generator_graph_based(G1)

cut_costs1 = cut_costs_for_graph(G1, unique_cuts1)

# Print the costs
for i, (cut1, cost1) in enumerate(zip(unique_cuts1, cut_costs1), 1):
    print(f"Cut {i} Cost: {cost1}")
    print(f"Cut {i}: {cut1}")


# Example usage
#G = nx.Graph()
#G.add_weighted_edges_from([(1, 2, 0.5), (1, 3, 10), (2, 3, 0.6), (3, 4, 0.7), (4, 5, 0.9)])

# Generate unique cuts for the graph
#unique_cuts = cut_generator_graph_based(G)

# Calculate costs for each unique cut
#cut_costs = cut_costs_for_graph(G, unique_cuts)

# Print the costs and cuts
#for i, (cut, cost) in enumerate(zip(unique_cuts, cut_costs), 1):
#    print(f"Cut {i} Cost: {cost}")
#    print(f"Cut {i}: {cut}")