import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

class GenerateRandomGraph:
    def __init__(self, num_of_nodes, num_of_clusters):
        self.num_of_nodes = num_of_nodes
        self.num_of_clusters = num_of_clusters
        self.average_edges_to_same_cluster = 5.0 #p
        self.average_edges_to_other_clusters = 0.2 #q

    def generate_random_graph(self):
        same_cluster_prob = self.average_edges_to_same_cluster / (self.num_of_nodes / self.num_of_clusters)
        different_cluster_prob = self.average_edges_to_other_clusters / (self.num_of_nodes - self.num_of_nodes / self.num_of_clusters)

        G = nx.Graph()
        ground_truth_list = []
        added = [False] * self.num_of_nodes
        names_map = [-1] * self.num_of_nodes

        for i in range(self.num_of_nodes):
            centroid = i // (self.num_of_nodes // self.num_of_clusters)
            edges_list = []

            if not added[i] and i > 0 and (centroid != (i+1) // (self.num_of_nodes // self.num_of_clusters) or i+1 >= self.num_of_nodes):
                names_map[i] = len(ground_truth_list)
                ground_truth_list.append(centroid)
                added[i] = True
                weight = max(1, int(random.gauss(5, 3)))
                edges_list.append((names_map[i-1], weight))

            for j in range(i+1, self.num_of_nodes):
                force_add = not added[i] and ((centroid != (j+1) // (self.num_of_nodes // self.num_of_clusters)) or j+1 >= self.num_of_nodes)
                if force_add or (centroid == j // (self.num_of_nodes // self.num_of_clusters) and random.random() <= same_cluster_prob) \
                        or (centroid != j // (self.num_of_nodes // self.num_of_clusters) and random.random() <= different_cluster_prob):
                    if not added[i]:
                        names_map[i] = len(ground_truth_list)
                        ground_truth_list.append(centroid)
                        added[i] = True
                    if not added[j]:
                        names_map[j] = len(ground_truth_list)
                        ground_truth_list.append(j // (self.num_of_nodes // self.num_of_clusters))
                        added[j] = True
                    weight = max(1, int(random.gauss(5 if centroid == j // (self.num_of_nodes // self.num_of_clusters) else 10, 3)))
                    edges_list.append((names_map[j], weight))

            G.add_node(names_map[i])  # Add node to graph with the mapped name
            for edge in edges_list:
                G.add_edge(names_map[i], edge[0], weight=edge[1])  # Add edge to graph

        ground_truth = ground_truth_list
        return G, ground_truth

    def visualize_graph(self, graph, ground_truth=None, visualize_ground_truth=True):
        pos = nx.spring_layout(graph)  # Layout for visualization

        if visualize_ground_truth and ground_truth is not None:
            # Draw nodes with different colors for each cluster
            clusters = sorted(set(ground_truth))
            colors = plt.cm.viridis(np.linspace(0, 1, len(clusters)))
            color_map = {cluster: color for cluster, color in zip(clusters, colors)}

            node_colors = [color_map[cluster] for cluster in ground_truth]
        else:
            node_colors = 'skyblue'

        nx.draw(graph, pos, with_labels=True, node_color=node_colors, node_size=500, edge_color='black', linewidths=1, font_size=10)

        # Draw edge labels
        labels = nx.get_edge_attributes(graph, 'weight')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)

        if visualize_ground_truth and ground_truth is not None:
            plt.title("Random Weighted Graph with Ground Truth Clusters")
        else:
            plt.title("Random Weighted Graph")

        plt.show()

