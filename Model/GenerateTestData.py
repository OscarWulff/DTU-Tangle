import random
from matplotlib.pylab import default_rng
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn.metrics import davies_bouldin_score

from Model.DataSetBinaryQuestionnaire import perform_tsne



class GenerateRandomGraph:
    def __init__(self, num_of_nodes, num_of_clusters, average_edges_to_same_cluster, average_edges_to_other_clusters):
        self.num_of_nodes = num_of_nodes
        self.num_of_clusters = num_of_clusters
        self.average_edges_to_same_cluster = average_edges_to_same_cluster
        self.average_edges_to_other_clusters = average_edges_to_other_clusters
        self.ground_truth = []

    def generate_random_graph(self):
        G = nx.Graph()

        # Create nodes and assign them to clusters
        for i in range(self.num_of_nodes):
            cluster_id = i % self.num_of_clusters  # Assign nodes to clusters
            G.add_node(i, cluster=cluster_id)  # Add node with cluster attribute

        # Generate edges within the same cluster with weight p
        for i in range(self.num_of_nodes):
            for j in range(i + 1, self.num_of_nodes):
                if G.nodes[i]['cluster'] == G.nodes[j]['cluster']:
                    if random.random() < self.average_edges_to_same_cluster:
                        G.add_edge(i, j, weight=self.average_edges_to_same_cluster)

        # Generate edges between different clusters with weight q
        for i in range(self.num_of_nodes):
            for j in range(i + 1, self.num_of_nodes):
                if G.nodes[i]['cluster'] != G.nodes[j]['cluster']:
                    if random.random() < self.average_edges_to_other_clusters:
                        G.add_edge(i, j, weight=self.average_edges_to_other_clusters)

        # Assign ground truth labels
        self.ground_truth = [G.nodes[i]['cluster'] for i in range(self.num_of_nodes)]

        return G, self.ground_truth
    
    def nmi_score(self, predicted_tangles):
        """
        Calculates the nmi score of the predicted tangles
        """
        nmi_score = normalized_mutual_info_score(self.ground_truth, predicted_tangles)
        return nmi_score



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
        Creating clusters for fixed centroids in arbitrary dimensions.
        The points are created from Gaussian Distribution.
        """
        dimensions = len(centroids[0])

        points = [[] for _ in range(dimensions)]  # List to store points for each dimension

        for truth, center in enumerate(centroids):
            # Generate points using Gaussian distribution for each dimension
            for dim in range(dimensions):
                points[dim].extend(np.random.normal(loc=center[dim], scale=self.std_deviation, size=cluster_points))

            for _ in range(cluster_points):
                self.ground_truth.append(truth)

        # Combine points from all dimensions into a single list of tuples
        self.points = [list(x) for x in zip(*points)]
           

    def random_clusters(self, cluster_points, dimensions, overlap=0.3):
        """
        Create a chosen number of clusters from Gaussian Distribution.
        Standard deviation and centroids are chosen randomly.
        """
        # Parameter that controls how much overlap is allowed

        std_low = 0.1
        std_high = 0.5

        tries = 0
        while tries < 1000:
            tries += 1
            start_over = False
            centroids = []
            std_deviations = []
            for i in range(self.numb_clusters):
                std_deviation_cluster = np.random.uniform(std_low, std_high, size=dimensions)
                center_cluster = np.random.uniform(self.box_low_x, self.box_high_x, size=dimensions)
                
                for centroid, std_deviation in zip(centroids, std_deviations):
                    for j in range(dimensions):
                        if np.abs(centroid[j] - center_cluster[j]) < (std_deviation[j] + std_deviation_cluster[j]) * overlap:
                            start_over = True
                            break

                if start_over:
                    break
                
                centroids.append(center_cluster)
                std_deviations.append(std_deviation_cluster)

            if not start_over:
                points = [[] for _ in range(dimensions)]  # List to store points for each dimension

                for truth, (center, std_deviation) in enumerate(zip(centroids, std_deviations)):
                    # Generate points using Gaussian distribution for each dimension
                    for dim in range(dimensions):
                        points[dim].extend(np.random.normal(loc=center[dim], scale=std_deviation[dim], size=cluster_points))

                    for _ in range(cluster_points):
                        self.ground_truth.append(truth)

                # Combine points from all dimensions into a single list of tuples
                self.points = [list(x) for x in zip(*points)]
                break

        if tries == 1000:
            print("Too many tries")

    def plot_points(self):
        clusters = sorted(set(self.ground_truth))
        colors = plt.cm.viridis(np.linspace(0, 1, len(clusters)))
        color_map = {cluster: color for cluster, color in zip(clusters, colors)}

        # Plot the points with color
        for point, truth in zip(self.points, self.ground_truth):
            plt.scatter(point[0], point[1], color=color_map[truth])

        # Add labels and title
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Colorized Clusters')

        # Display the plot
        plt.show()

    
    def davies_bouldin_score(self, ground_truth, labels):
        score = davies_bouldin_score(ground_truth, labels)
        return score

    def nmi_score(self, ground_truth, labels):
        """
        Calculates the nmi score of the predicted tangles
        """
        nmi_score = normalized_mutual_info_score(ground_truth, labels)
        return nmi_score

    def k_means(self, k): 
        """
        Clusters the data with K-means algorithm
        """
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(self.points)
        return kmeans.labels_
    
    def spectral_clustering(self, k):
        """
        Clusters the data with spectral algorithm
        """
        spectral = SpectralClustering(n_clusters=k)
        spectral.fit(self.points)
        return spectral.labels_

            
class BitSet:
    def __init__(self, size):
        self.size = size
        self.bitset = np.zeros(size, dtype=bool)

    def add(self, index):
        self.bitset[index] = True

    def setValue(self, index, value):
        self.bitset[index] = value

    def flip(self, index):
        self.bitset[index] = not self.bitset[index]

    def get(self, index):
        return self.bitset[index]     
                
             
class GenerateDataBinaryQuestionnaire():
    def __init__(self, numb_of_participants, numb_of_questions, numb_of_clusters):
        self.numb_of_participants = numb_of_participants
        self.numb_of_questions = numb_of_questions
        self.numb_of_clusters = numb_of_clusters
        self.result = []
        self.points = []
        self.ground_truth = []
        self.questionaire = []

       

    def generate_random_binary_questionnaire_answers(self,numberOfAnswers, numberOfQuestions):
        result = [None] * numberOfAnswers
        rng = default_rng()
        for i in range(numberOfAnswers):
            result[i] = BitSet(numberOfQuestions)
            for j in range(numberOfQuestions):
                if rng.choice([True, False]):
                    result[i].add(j)
        return result

    def generate_biased_binary_questionnaire_answers(self):
        self.result = [BitSet(self.numb_of_questions) for _ in range(self.numb_of_participants)]
        self.ground_truth = np.zeros(self.numb_of_participants, dtype=int)
        index = 0
        extra = self.numb_of_participants % self.numb_of_clusters

        for i in range(self.numb_of_clusters):
            center = BitSet(self.numb_of_questions)
            for j in range(self.numb_of_questions):
                if random.random() < 0.5:  # Simulating a random choice between True and False
                    center.add(j)

            cluster_size = self.numb_of_participants // self.numb_of_clusters + (1 if i < extra else 0)
            for _ in range(cluster_size):
                for k in range(self.numb_of_questions):
                    value = center.get(k)
                    self.result[index].setValue(k, value)
                    if random.random() >= 0.90:  # Flip the bit with a 10% chance
                        self.result[index].flip(k)
                self.ground_truth[index] = i
                index += 1

        return self.result, self.ground_truth
    @staticmethod
    def generateBiased_binary_questionnaire_answers(numberOfAnswers, numberOfQuestions, distributionPercentage=0.5):
        if distributionPercentage < 0 or distributionPercentage > 1:
            return None

        result = [None] * numberOfAnswers
        groundTruth = [None] * numberOfAnswers
        rng = default_rng()
        nPartition = int(numberOfAnswers * distributionPercentage)

        # Cluster of mainly false answers
        for i in range(nPartition):
            result[i] = BitSet(numberOfQuestions)
            groundTruth[i] = 0
            for j in range(numberOfQuestions):
                if rng.integers(0, 100) >= 90:
                    result[i].add(j)

        # Cluster of mainly true answers
        for i in range(nPartition, numberOfAnswers):
            result[i] = BitSet(numberOfQuestions)
            groundTruth[i] = 1
            for j in range(numberOfQuestions):
                if rng.integers(0, 100) < 90:
                    result[i].add(j)

        return result, groundTruth
    
    def res_to_points(self):
        self.questionaire = np.array([bitset.bitset for bitset in self.result])
        tsne_coordinates = perform_tsne(self.questionaire, 2)
        points_x = tsne_coordinates['Dim1'].values
        points_y = tsne_coordinates['Dim2'].values
        self.points = [(x, y, z) for z, (x, y) in enumerate(zip(points_x, points_y))]

    def nmi_score(self, predicted_tangles):
        """
        Calculates the nmi score of the predicted tangles
        """
        nmi_score = normalized_mutual_info_score(self.ground_truth, predicted_tangles)
        return nmi_score
    
    def DBscan(self, min_s, e): 
        """
        Clusters the data with DBSCAN algorithm
        """
        scan = DBSCAN( eps=e, min_samples=min_s,)

        clusters = scan.fit(self.points)
        print(clusters.labels_)


        return clusters.labels_
    
    def k_means(self, k): 
        """
        Clusters the data with K-means algorithm
        """
        kmeans = KMeans(n_clusters=k)

        kmeans.fit(self.points)
        print(kmeans.labels_)

        return kmeans.labels_