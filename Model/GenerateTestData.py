import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans, SpectralClustering



class GenerateRandomGraph:
    def __init__(self, num_of_nodes, num_of_clusters):
        self.num_of_nodes = num_of_nodes
        self.num_of_clusters = num_of_clusters
        self.average_edges_to_same_cluster = 5.0 #p
        self.average_edges_to_other_clusters = 0.6 #q

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


class GenerateDataFeatureBased():

    def __init__(self, numb_clusters, std_deviation):
        self.numb_clusters = numb_clusters
        self.std_deviation = std_deviation
        self.points = []
        self.ground_truth = []

        # The parameters for the encirclement 
        self.box_low_x = 0
        self.box_high_x = 20
        self.box_low_y = 0
        self.box_high_y = 20

    def fixed_clusters(self, cluster_points, centroids):
        """
        Creating two clusters in 2 dimension for two fixed centroids.
        The points is created from Gaussian Distribution. 
        """
        points_x = []
        points_y = []

        for truth, (center_x, center_y) in enumerate(centroids):
            # Generate points using Gaussian distribution
            points_x.extend(np.random.normal(loc=center_x, scale=self.std_deviation, size=cluster_points))
            points_y.extend(np.random.normal(loc=center_y, scale=self.std_deviation, size=cluster_points))
            
            for _ in range(cluster_points):
                self.ground_truth.append(truth)

        self.points = [(x, y, z) for z, (x, y) in enumerate(zip(points_x, points_y))]
           

    def random_clusters(self, cluster_points):
        """
        Create a choosen number of clusters from Gaussian Distribution. 
        Standard deviation and centroids are choosen random. 
        """
        # Parameter that controls how much overlap is allowed
        overlap = 0.3

        std_low = 0.1
        std_high = 0.5

        tries = 0
        while(tries < 1000):
            tries += 1
            start_over = False
            centroids = []
            std_deviations = [] 
            for i in range(self.numb_clusters): 
                std_deviation_x = np.random.uniform(std_low, std_high)
                std_deviation_y = np.random.uniform(std_low, std_high)
                center_x = np.random.uniform(self.box_low_x, self.box_high_x)
                center_y = np.random.uniform(self.box_low_y, self.box_high_y)
                
                for i in range(len(centroids)): 
                    x1, y1 = centroids[i]
                    std_x, std_y = std_deviations[i]

                    # Check if it is not too close to another cluster
                    if np.abs(center_x-x1) < (std_deviation_x + std_x) * overlap:
                        start_over = True
                    if np.abs(center_y-y1) < (std_deviation_y + std_y) * overlap:
                        start_over = True
                    # Check clusters are inside the box
                    if center_y - std_deviation_y*3 < self.box_low_y and center_y + std_deviation_y*3 > self.box_high_y:
                        start_over = True
                    if center_x - std_deviation_x*3 < self.box_low_x and center_x + std_deviation_x*3 > self.box_high_x:
                        start_over = True

                if start_over:
                    break

                centroids.append((center_x, center_y))
                std_deviations.append((std_deviation_x, std_deviation_y))


            if not start_over:
                points_x = []
                points_y = []

                for truth, (center_x, center_y) in enumerate(centroids):
                    # Generate points using Gaussian distribution
                    points_x.extend(np.random.normal(loc=center_x, scale=std_deviations[truth][0], size=cluster_points))
                    points_y.extend(np.random.normal(loc=center_y, scale=std_deviations[truth][1], size=cluster_points))
                    
                    for _ in range(cluster_points):
                        self.ground_truth.append(truth)

                self.points = [(x, y, z) for z, (x, y) in enumerate(zip(points_x, points_y))]

                print(tries)
                break
        
        if tries == 1000: 
            print("to many tries")

    def plot_points(self):
        clusters = sorted(set(self.ground_truth))
        colors = plt.cm.viridis(np.linspace(0, 1, len(clusters)))
        color_map = {cluster: color for cluster, color in zip(clusters, colors)}

        # Plot the points with color
        for point, truth in zip(self.points, self.ground_truth):
            plt.scatter(point[0], point[1], color=color_map[truth])

        plt.xlim(self.box_low_x-1, self.box_high_x+1)  # Setting x-axis limits from 0 to 10
        plt.ylim(self.box_low_y-1, self.box_high_y+1) 
        # Add labels and title
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Colorized Clusters')

        # Display the plot
        plt.show()

    def plot_points_prob(self, probability):
        """
        Function to be used if you want to plot the points where the probability 
        is plotted as the transparency.  
        """

        clusters = sorted(set(self.ground_truth))
        colors = plt.cm.viridis(np.linspace(0, 1, len(clusters)))
        color_map = {cluster: color for cluster, color in zip(clusters, colors)}

        # Plot the points with color
        for point, truth in zip(self.points, self.ground_truth):
            plt.scatter(point[0], point[1], color=color_map[truth], alpha=probability[point[2]])

        plt.xlim(self.box_low_x-1, self.box_high_x+1)  # Setting x-axis limits from 0 to 10
        plt.ylim(self.box_low_y-1, self.box_high_y+1) 
        # Add labels and title
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Colorized Clusters')

        # Display the plot
        plt.show()

    
    def nmi_score(self, predicted_tangles):
        """
        Calculates the nmi score of the predicted tangles
        """
        nmi_score = normalized_mutual_info_score(self.ground_truth, predicted_tangles)
        return nmi_score


    def k_means(self, k):
        """
        Clusters the data with K-means algorithm
        """
        kmeans = KMeans(n_clusters=k)

        points = [[x, y] for x, y, _ in self.points]

        kmeans.fit(points)

        return kmeans.labels_
    
    def spectral_clustering(self, k):
        """
        Clusters the data with spectral algorithm
        """
        spectral = SpectralClustering(n_clusters=k)

        points = [[x, y] for x, y, _ in self.points]

        spectral.fit(points)

        return spectral.labels_

            
                
                
             
class GenerateDataBinaryQuestionnaire:
    def __init__(self, num_questions, num_samples_per_cluster, num_clusters):
        self.num_questions = num_questions
        self.num_samples_per_cluster = num_samples_per_cluster
        self.num_clusters = num_clusters

    def generate_data(self):
        data = np.zeros((self.num_clusters * self.num_samples_per_cluster, self.num_questions), dtype=int)
        for cluster in range(self.num_clusters):
            # Generate a pattern for this cluster
            cluster_pattern = np.random.choice([0, 1], size=self.num_questions)
            for i in range(self.num_samples_per_cluster):
                # Use the cluster pattern to generate samples for this cluster
                sample = cluster_pattern.copy()  # Make a copy to avoid modifying the original pattern
                # Introduce randomness by flipping some bits
                for j in range(len(sample)):
                    if random.random() < 0.2:  # Adjust the probability as needed
                        sample[j] = 1 - sample[j]  # Flip the bit
                data[cluster * self.num_samples_per_cluster + i] = sample
        return data



