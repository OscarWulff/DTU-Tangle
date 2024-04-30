import math
import sys 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import svd
from sklearn.decomposition import PCA
from Model.DataType import DataType
from Model.Cut import Cut
from sklearn.manifold import TSNE
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import davies_bouldin_score
from sklearn.neighbors import KernelDensity
import random 
from sklearn.cluster import SpectralClustering
from scipy.spatial import cKDTree
import heapq

class DataSetFeatureBased(DataType):

    def __init__(self,agreement_param, cuts : list[Cut] =[], search_tree=None):
        super().__init__(agreement_param, cuts, search_tree)
        self.points = []
        #self.initialize()

    def initialize(self):
        _, X = pca("iris.csv")
        x1 = X[:, 0]
        x2 = X[:, 1]

        for z, (x, y) in enumerate(zip(x1, x2)):
           self.points.append((x, y, z))
        
        self.cut_generator_axis_dimensions()
        self.cost_function()

    def density_cost(self, radius):
    
        # Create KDE of the all the points before the cut
        # Afterwards create KDE of the all the points of each side of the cut

        # Take every point and calculate the diffrence in density from the KDE before and after the cut
        # Sum the diffrences and set it as the cost of the cut
        def calculate_density(points, radius):
            # Convert points to numpy array
            points = np.array(points)

            # Fit Kernel Density Estimation (KDE) model
            kde = KernelDensity(bandwidth=radius, kernel='gaussian')
            kde.fit(points)

            # Estimate density at each point
            density = np.exp(kde.score_samples(points))

            # Return density values
            return density

        density_all = calculate_density(self.points, radius)

        for cut in self.cuts: 
            sum_cost = 0.0
            left_oriented = cut.A
            right_oriented = cut.Ac

            left_points = []
            right_points = []

            for left_or in left_oriented:
                left_points.append(self.points[left_or][:-1])

            for right_or in right_oriented:
                right_points.append(self.points[right_or][:-1])

            density_left = calculate_density(left_points, radius)
            density_right = calculate_density(right_points, radius)

            for i in range(len(density_left)):
                sum_cost += abs(density_all[i] - density_left[i])

            for i in range(len(density_right)):
                sum_cost += abs(density_all[i] - density_right[i])

            cut.cost = sum_cost


    def CURE_cost(self):

        def select_representatives(points, num_representatives):
            # Randomly select representatives from the data
            indices = np.random.choice(len(points), size=num_representatives)
            representatives = [points[idx] for idx in indices] 
            return representatives
        

        def find_clusters(clusters, k):
            class Cluster:
                def __init__(self, point):
                    self.points = [point]  # Initially, each cluster has only one point
                    self.mean = np.array(point)  # Mean of the points in the cluster
            
            
            clusters = [Cluster(point) for point in clusters]
            
            
            while len(clusters) > k: 
                min_distance = float('inf')
                closest_clusters = []
                for c1 in clusters:
                    for c2 in clusters:
                        if c1 != c2:
                            distance = -np.linalg.norm(c1.mean - c2.mean)
                            if distance < min_distance:
                                min_distance = distance
                                closest_clusters = [c1, c2]
            
                clusters.remove(closest_clusters[0])
                clusters.remove(closest_clusters[1])

                merged_cluster = Cluster(None)
                merged_cluster.points = closest_clusters[0].points + closest_clusters[1].points
                merged_cluster.mean = np.mean(merged_cluster.points, axis=0)
                clusters.append(merged_cluster)

            return clusters
            

        for cut in self.cuts: 
            sum_cost = 0.0
            left_oriented = cut.A
            right_oriented = cut.Ac

            left_points = []
            right_points = []

            for left_or in left_oriented:
                left_points.append(self.points[left_or][:-1])

            for right_or in right_oriented:
                right_points.append(self.points[right_or][:-1])

            # Find representatives for each cluster
            rep_left = select_representatives(left_points, min(10, self.agreement_param//2))
            rep_right = select_representatives(right_points, min(10, self.agreement_param//2))

            clusters_left = find_clusters(rep_left, math.ceil(len(left_points)/self.agreement_param))
            clusters_right = find_clusters(rep_right, math.ceil(len(right_points)/self.agreement_param))

            for left_cluster in clusters_left:
                for right_cluster in clusters_right:
                    sum_cost += -np.linalg.norm(left_cluster.mean - right_cluster.mean)
            
            cut.cost = sum_cost


    def CURE_cost_heap(self):
        class Cluster:
            def __init__(self, point):
                self.points = [point]  # Initially, each cluster has only one point
                self.mean = np.array(point)  # Mean of the points in the cluster
                self.rep = [point]  # Representative points of the cluster
                self.closest = None  # Closest cluster
                self.closest_distance = float('inf')  # Distance to the closest cluster

            def __lt__(self, other):
                # Define comparison for clusters based on their closest distance
                return self.closest_distance < other.closest_distance

        def calculate_distance(point1, point2):
            # Euclidean distance between two points
            return np.linalg.norm(np.array(point1) - np.array(point2))

        def calculate_mean(cluster):
            # Calculate mean of points in the cluster
            return np.mean(cluster.points, axis=0)

        def update_closest_cluster(cluster, clusters):
            # Update closest cluster for the given cluster
            for other_cluster in clusters:
                if other_cluster != cluster:
                    distance = calculate_distance(cluster.mean, other_cluster.mean)

                    if distance < cluster.closest_distance or cluster.closest not in clusters:
                        cluster.closest = other_cluster
                        cluster.closest_distance = distance

        def update_heap(cluster, clusters, heap):
            # Update heap after merging clusters
            updated_heap = []
            for item in heap:
                if item[1] != cluster and item[1] != cluster.closest:
                    updated_heap.append(item)
            heapq.heapify(updated_heap)
            for other_cluster in clusters:
                if other_cluster != cluster and other_cluster != cluster.closest:
                    distance = calculate_distance(cluster.mean, other_cluster.mean)
                    heapq.heappush(updated_heap, (distance, other_cluster))
            return updated_heap

        def merge_clusters(cluster1, cluster2):
            # Merge two clusters
            merged_cluster = Cluster(None)
            merged_cluster.points = cluster1.points + cluster2.points
            merged_cluster.mean = calculate_mean(merged_cluster)
            merged_cluster.rep = cluster1.rep + cluster2.rep
            return merged_cluster

        def select_representatives(points, num_representatives):
            # Randomly select representatives from the data
            indices = np.random.choice(len(points), size=num_representatives, replace=False)
            representatives = [points[idx] for idx in indices] 
            return representatives

        def find_clusters(points, k):
            # Initialize clusters
            clusters = []
            for point in points:
                clusters.append(Cluster(point))
            #clusters = [Cluster(point) for point in points]
            heap = []

            # Build k-d tree
            tree = cKDTree(points)

            # Populate heap with closest clusters for each cluster
            for cluster in clusters:
                cluster.mean = calculate_mean(cluster)
                for other_cluster in clusters:
                    if other_cluster != cluster:
                        distance = calculate_distance(cluster.mean, other_cluster.mean)
                        if distance < cluster.closest_distance:
                            cluster.closest = other_cluster
                            cluster.closest_distance = distance
                heapq.heappush(heap, (cluster.closest_distance, cluster))

            # Main loop
            while len(heap) > k:
                # Merge closest clusters
                distance, cluster = heapq.heappop(heap)
                other_cluster = cluster.closest
                merged_cluster = merge_clusters(cluster, other_cluster)

                # Update mean and representative points for the merged cluster
                merged_cluster.mean = calculate_mean(merged_cluster)
                merged_cluster.rep = cluster.rep + other_cluster.rep

                clusters.remove(cluster)
                clusters.remove(other_cluster)
                # Remove clusters from heap and tree
                heap = update_heap(cluster, clusters, heap)
                heap = update_heap(other_cluster, clusters, heap)

                # Insert merged cluster into heap and tree
                heapq.heappush(heap, (merged_cluster.closest_distance, merged_cluster))
                clusters.append(merged_cluster)
                # Update closest clusters for remaining clusters
                for remaining_cluster in clusters:
                    update_closest_cluster(remaining_cluster, clusters)
            return clusters

        for cut in self.cuts: 
            sum_cost = 0.0
            left_oriented = cut.A
            right_oriented = cut.Ac

            left_points = []
            right_points = []

            for left_or in left_oriented:
                left_points.append(self.points[left_or][:-1])

            for right_or in right_oriented:
                right_points.append(self.points[right_or][:-1])

            # Find representatives for each cluster
            rep_left = select_representatives(left_points, self.agreement_param//2)
            rep_right = select_representatives(right_points, self.agreement_param//2)

            clusters_left = find_clusters(rep_left, math.ceil(len(left_points)/self.agreement_param))
            clusters_right = find_clusters(rep_right, math.ceil(len(right_points)/self.agreement_param))

            for left_cluster in clusters_left:
                for right_cluster in clusters_right:
                    sum_cost += -np.linalg.norm(left_cluster.mean - right_cluster.mean)
            
            cut.cost = sum_cost


    def pairwise_cost(self):
        """ 
        This function is used to calculate the cost of cuts for feature based data set
        
        Parameters:
        cuts of the dataset

        Returns:
        cost of each cut
        """
        for cut in self.cuts:
            sum_cost = 0.0
            left_oriented = cut.A
            right_oriented = cut.Ac
            # Calculate the cost
            for left_or in left_oriented:
                for right_or in right_oriented:
                    point1 = np.array(self.points[left_or][:-1])
                    point2 = np.array(self.points[right_or][:-1])
                    sum_cost += -np.linalg.norm(point1 - point2)
            cut.cost = sum_cost

    def mean_cost(self):
        for cut in self.cuts:
            sum_cost = 0.0
            left_oriented = cut.A
            right_oriented = cut.Ac

            A_points = []
            Ac_points = []

            # Calculate the mean 
            for left_or in left_oriented:
                A_points.append(self.points[left_or])
                
            for right_or in right_oriented:
                Ac_points.append(self.points[right_or])
            
            mean_A = np.mean(np.array(A_points)[:, :-1], axis=0)
            mean_Ac = np.mean(np.array(A_points)[:, :-1], axis=0)

            # Calculate the cost
            for left_or in left_oriented:
                sum_cost += -np.linalg.norm(self.points[left_or][:-1] - mean_A)
                
            for right_or in right_oriented:
                sum_cost += -np.linalg.norm(self.points[right_or][:-1] - mean_Ac)

            cut.cost = sum_cost


    def random_cuts(self):
        """ 
        This function is used to generate random cuts for feature based data set
        
        Parameters:
        cuts of the dataset

        Returns:
        random cuts
        """
        self.cuts = []
        n = len(self.points)
        dimensions = len(self.points[0])


        # Add index to keep track of original order
        self.points = [point + [z] for z, point in enumerate(self.points)]

        values = [[] for _ in range(dimensions)]  

        # Extract values for each dimension
        for point in self.points:
            for dim in range(dimensions):
                values[dim].append(point[dim])

        sorted_points = [self.sort_for_list(values[dim], self.points) for dim in range(dimensions)]


        for dim in range(dimensions):
            for _ in range(int(2*(n/self.agreement_param))):
                cut = Cut()
                cut.init()
                i = random.uniform(1, n)
                for k in range(0, n):
                    if k < i:
                        cut.A.add(sorted_points[dim][k][dimensions])
                    else:
                        cut.Ac.add(sorted_points[dim][k][dimensions])
                self.cuts.append(cut)


        # If we could make some test that shows that generally how bad random cuts are.
        # But we could also make a test that shows that random cuts are better than the other cuts.
        # This would be a good test to show how the tangles is limited by its cuts.

    def cut_spectral(self):
        self.cuts = []
        partitions = []
        n_partitions = 1

        n_clusters = len(self.points)//self.agreement_param
        n_neighbors = (len(self.points)//n_clusters)//12


        for i in range(n_partitions):
            # Create a SpectralClustering object

            # set n_jobs = -1 , to obtain parallel computation
            spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', n_neighbors = n_neighbors, n_init=1, n_jobs =-1, random_state=43)
            
            # Fit the spectral clustering model to the data
            labels = spectral_clustering.fit_predict(self.points)
            
            partitions.append(labels)
        
        # Create cuts based on the partitions
        for part in partitions:
            index = 1
            while True: 
                cut = Cut()
                cut.init()
            
                for i in range(index):
                    cut.A.update(np.where(part == i)[0])
                for i in range(index, n_clusters):
                    cut.Ac.update(np.where(part == i)[0])

                self.cuts.append(cut)

                if index == math.ceil(n_clusters/2):
                    break
                
                index += 1

    def cut_generator_axis_solveig(self):
        self.cuts = []
        n = len(self.points)
        dimensions = len(self.points[0])


        # Add index to keep track of original order
        if type(self.points) == np.ndarray:
            self.points = self.points.tolist()
        self.points = [point + [z] for z, point in enumerate(self.points)]

        values = [[] for _ in range(dimensions)]  

        # Extract values for each dimension
        for point in self.points:
            for dim in range(dimensions):
                values[dim].append(point[dim])

        sorted_points = [self.sort_for_list(values[dim], self.points) for dim in range(dimensions)]

        
        # Extract the first point for each dimension
        i = 1
        for dim in range(dimensions):
            cut = Cut()
            cut.init()
            cut.A.add(sorted_points[dim][0][dimensions])
            for k in range(i, n):
                cut.Ac.add(sorted_points[dim][k][dimensions])
            self.cuts.append(cut)

        i += self.agreement_param - 1 
        while n > i + self.agreement_param - 1:
            cuts = [Cut() for _ in range(dimensions)]  # Create cuts for each dimension
            for dim in range(dimensions):
                cuts[dim].init()
                for k in range(0, i):
                    cuts[dim].A.add(sorted_points[dim][k][dimensions])
                    if k == i - 1:
                        cuts[dim].line_placement = (sorted_points[dim][k][0], dim)
                for k in range(i, n):
                    cuts[dim].Ac.add(sorted_points[dim][k][dimensions])
                self.cuts.append(cuts[dim])
            i += self.agreement_param - 1

    def cut_generator_axis_dimensions(self):
        self.cuts = []
        n = len(self.points)

        dimensions = len(self.points[0])
        # Add index to keep track of original order
        if type(self.points) == np.ndarray:
            self.points = self.points.tolist()
        self.points = [point + [z] for z, point in enumerate(self.points)]

        values = [[] for _ in range(dimensions)]  # 

        # Extract values for each dimension
        for point in self.points:
            for dim in range(dimensions):
                values[dim].append(point[dim])

        sorted_points = [self.sort_for_list(values[dim], self.points) for dim in range(dimensions)]

        i = self.agreement_param
        while n >= i + self.agreement_param:
            cuts = [Cut() for _ in range(dimensions)]  # Create cuts for each dimension
            for dim in range(dimensions):
                cuts[dim].init()
                for k in range(0, i):
                    cuts[dim].A.add(sorted_points[dim][k][dimensions])
                    if k == i - 1:
                        cuts[dim].line_placement = (sorted_points[dim][k][0], dim)
                for k in range(i, n):
                    cuts[dim].Ac.add(sorted_points[dim][k][dimensions])
                self.cuts.append(cuts[dim])
            i += self.agreement_param
    
    def cut_generator_axis_dimensions_finer(self, increaser):
        self.cuts = []
        n = len(self.points)

        dimensions = len(self.points[0])
        # Add index to keep track of original order
        if type(self.points) == np.ndarray:
            self.points = self.points.tolist()
        self.points = [point + [z] for z, point in enumerate(self.points)]

        values = [[] for _ in range(dimensions)]  # 

        # Extract values for each dimension
        for point in self.points:
            for dim in range(dimensions):
                values[dim].append(point[dim])

        sorted_points = [self.sort_for_list(values[dim], self.points) for dim in range(dimensions)]

        i = increaser
        while n >= i + increaser:
            cuts = [Cut() for _ in range(dimensions)]  # Create cuts for each dimension
            for dim in range(dimensions):
                cuts[dim].init()
                for k in range(0, i):
                    cuts[dim].A.add(sorted_points[dim][k][dimensions])
                    if k == i - 1:
                        cuts[dim].line_placement = (sorted_points[dim][k][0], dim)
                for k in range(i, n):
                    cuts[dim].Ac.add(sorted_points[dim][k][dimensions])
                self.cuts.append(cuts[dim])
            i += increaser


    def order_function(self):
        """ 
        order the cuts after cost 
        
        Paramaters:
        Cuts

        Returns: 
        An order of the cuts    
        """

        costs = []
        for cut in self.cuts: 
            costs.append(cut.cost)

        zipped_data = zip(costs, self.cuts)
        # Sort the zipped data based on the costs
        sorted_data = sorted(zipped_data, key=lambda x: x[0])
        _, cuts_ordered = zip(*sorted_data)
        return cuts_ordered

    def euclidean_distance(self, x1, x2, y1, y2):
        distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        return distance

    def euclidean_distance_eulers(self, x1, x2, y1, y2):
        distance = math.e(-math.sqrt((x1 - x2)**2 + (y1 - y2)**2))
        return distance

    def sort_for_list(self, axis_values, points):
        combined = list(zip(axis_values, points))
        sorted_combined = sorted(combined, key=lambda x: x[0])
        sorted_combined = [e[1] for e in sorted_combined]
        return sorted_combined
    
    def davies_bouldin_score(self, ground_truth, labels):
        score = davies_bouldin_score(ground_truth, labels)
        return score

    def nmi_score(self, ground_truth, labels):
        """
        Calculates the nmi score of the predicted tangles
        """
        nmi_score = normalized_mutual_info_score(ground_truth, labels)
        return nmi_score

def pca(X):
    """ 
    This function is used to reduce the dimension of feature based data set 
    
    Parameters:
    Takes a csv-file of numbers (integers and floats). Each numbers is related to a feature in the dataset. 
    Each row represent object.
    Example of a file: 
    " 
    1.23, 5, 8.9, 1000, 123, 33.5
    3.6, 56, 4.7, 4350, 343, 55.98
    ...
    ...
    "

    Returns:
    Eigenvectors and Projections of the PCA
    """
    N, _ = X.shape
    # Subtract mean value from data
    Y = X - np.ones((N, 1)) * X.mean(0)

    # PCA by computing SVD of Y
    _, S, Vh = svd(Y, full_matrices=False)
    # scipy.linalg.svd returns "Vh", which is the Hermitian (transpose)
    # of the vector V. So, for us to obtain the correct V, we transpose:
    
    X_projected = np.dot(Y, Vh[:2, :].T)

    return S, X_projected

def tsne(X):
    perplexity = min(20, len(X) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity)
    data = tsne.fit_transform(X)
    return data

def read_file(filename):
    df = pd.read_csv(filename)
    X = df.values
    X = X[:1000, :]
    X = X.astype(float)
    return X

def calculate_explained_varince(S):
    rho = (S * S) / (S * S).sum()
    return rho




