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
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.neighbors import KernelDensity
import random 
from sklearn.cluster import SpectralClustering, KMeans
from scipy.spatial import cKDTree
import heapq
from scipy.stats import gaussian_kde
import time

class DataSetFeatureBased(DataType):

    def __init__(self,agreement_param, cuts : list[Cut] =[], search_tree=None):
        super().__init__(agreement_param, cuts, search_tree)
        self.points = []



    # Cost functions 

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

            A_points = []
            Ac_points = []

            for left_or in cut.A:
                A_points.append(self.points[left_or][:-1])

            for right_or in cut.Ac:
                Ac_points.append(self.points[right_or][:-1])

            # Find representatives for each cluster
            rep_left = select_representatives(A_points, min(5, self.agreement_param//2))
            rep_right = select_representatives(Ac_points, min(5, self.agreement_param//2))

            clusters_left = find_clusters(rep_left, math.ceil(len(A_points)/self.agreement_param))
            clusters_right = find_clusters(rep_right, math.ceil(len(Ac_points)/self.agreement_param))
            
            # Calculate the cost
            left_sum = 0
            right_sum = 0
            for left_or in A_points:
                min_distance = float('inf')
                min_point = None
                for left_cluster in clusters_left:
                    if min_distance > np.linalg.norm(left_or - left_cluster.mean):
                        min_distance = np.linalg.norm(left_or - left_cluster.mean)
                        min_point = left_cluster.mean

                left_sum += np.linalg.norm(left_or - min_point)
                
            for right_or in Ac_points:
                min_distance = float('inf')
                min_point = None
                for right_cluster in clusters_right:
                    if min_distance > np.linalg.norm(right_or - right_cluster.mean):
                        min_distance = np.linalg.norm(right_or - right_cluster.mean)
                        min_point = right_cluster.mean

                right_sum += np.linalg.norm(right_or - min_point)

            sum_cost = left_sum/len(A_points) + right_sum/len(Ac_points)

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
                    point1 = np.array(self.points[int(left_or)][:-1])
                    point2 = np.array(self.points[int(right_or)][:-1])
                    sum_cost += -np.linalg.norm(point1 - point2)
            cut.cost = sum_cost

    def mean_cost(self):
        for cut in self.cuts:

            sum_cost = 0.0
            mean_A = np.mean(np.array(cut.A_points), axis=0)
            mean_Ac = np.mean(np.array(cut.Ac_points), axis=0)

            # Calculate the cost
            for point in cut.A_points:
                sum_cost += -np.linalg.norm(point - mean_Ac)
                
            for point in cut.Ac_points:
                sum_cost += -np.linalg.norm(point - mean_A)

            cut.cost = sum_cost
      
    def mean_cut_cost(self):
        for cut in self.cuts:
            sum_cost = 0.0
            mean_A = np.mean(np.array(cut.A_points), axis=0)
            mean_Ac = np.mean(np.array(cut.Ac_points), axis=0)

            # Calculate the cost
            for point in cut.A_points:
                sum_cost += -np.linalg.norm(point - mean_Ac)
                
            for point in cut.Ac_points:
                sum_cost += -np.linalg.norm(point - mean_A)

            cut.cost = sum_cost


    # Cut generators

    def mean_cut(self):
        self.cuts = []
        n = len(self.points)
        dimensions = len(self.points[0])
        interval = self.agreement_param//2
        
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
        
        i = self.agreement_param + int(self.agreement_param * 0.05)
        while n >= i + self.agreement_param:
            cuts = [Cut() for _ in range(dimensions)]  # Create cuts for each dimension
            for dim in range(dimensions):
                cuts[dim].init()
                
                mean_A = np.mean(np.array(sorted_points[dim])[(i-interval):i, :-1], axis=0)
                mean_Ac = np.mean(np.array(sorted_points[dim])[i:(i+interval), :-1], axis=0)

                for point in sorted_points[dim]:
                    if np.linalg.norm(point[:-1] - mean_A) < np.linalg.norm(point[:-1] - mean_Ac):
                        cuts[dim].A.add(point[-1])
                        cuts[dim].A_points.append(point[:-1])
                    else:
                        cuts[dim].Ac.add(point[-1])
                        cuts[dim].Ac_points.append(point[:-1])
                self.cuts.append(cuts[dim])
            i += self.agreement_param + int(self.agreement_param * 0.05)

    
    def adjusted_cut(self):
        self.cuts = []
        n = len(self.points)
        dimensions = len(self.points[0])
        interval = self.agreement_param//2


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

        i = self.agreement_param
        while n >= i + self.agreement_param:
            cuts = [Cut() for _ in range(dimensions)]
            for dim in range(dimensions):
                cuts[dim].init()
                highest_gap = 0
                gap_index = 0
                prev_point = sorted_points[dim][i-interval][dim]
              
                for k in range(i-interval+1, i+interval):
                    gap = abs(prev_point - sorted_points[dim][k][dim])
                    if gap > highest_gap:
                        highest_gap = gap
                        gap_index = k
                    prev_point = sorted_points[dim][k][dim]

                for point in sorted_points[dim][:gap_index]:
                    cuts[dim].A.add(point[-1])
                    cuts[dim].A_points.append(point[:-1])
                for point in sorted_points[dim][gap_index:]:
                    cuts[dim].Ac.add(point[-1])
                    cuts[dim].Ac_points.append(point[:-1])
                self.cuts.append(cuts[dim])
            i += self.agreement_param                
 
    def cut_kmeans(self):
        self.cuts = []
        partitions = []
        n_partitions = 2

        n_clusters = len(self.points)//self.agreement_param

        for i in range(n_partitions):
            # Create a SpectralClustering object

            # set n_jobs = -1 , to obtain parallel computation
            kmeans = KMeans(n_clusters=n_clusters, init="random", n_init=1, max_iter=2)
                
            labels = kmeans.fit_predict(self.points)
            
            partitions.append(labels)

            print(labels)

        # Create cuts based on the partitions
        for part in partitions:
            index = 1
            while True: 
                cut = Cut()
                cut.init()
            
                for i in range(index):
                    cut.A.update(np.where(part == i)[0])
                    selected_points = [self.points[idx] for idx in np.where(part == i)[0]]
                    # Append the selected points to cut.A_points
                    cut.A_points = selected_points
                for i in range(index, n_clusters):
                    cut.Ac.update(np.where(part == i)[0])
                    selected_points = [self.points[idx] for idx in np.where(part == i)[0]]
                    # Append the selected points to cut.A_points
                    cut.Ac_points = selected_points
                self.cuts.append(cut)

                if index == math.ceil(n_clusters/2):
                    break
                
                index += 1

        self.cuts = []

    def cut_spectral(self):
        self.cuts = []
        partitions = []
        n_partitions = 2

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
                    print(cut.A)
                    selected_points = [self.points[idx] for idx in np.where(part == i)[0]]
                    # Append the selected points to cut.A_points
                    cut.A_points = selected_points
                    print(cut.A_points)
                for i in range(index, n_clusters):
                    cut.Ac.update(np.where(part == i)[0])
                    print(cut.Ac)
                    selected_points = [self.points[idx] for idx in np.where(part == i)[0]]
                    # Append the selected points to cut.A_points
                    cut.Ac_points = selected_points
                    print(cut.Ac_points)
                self.cuts.append(cut)

                if index == math.ceil(n_clusters/2):
                    break
                
                index += 1

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
            cut.A_points.append(sorted_points[dim][0][:-1])
            for k in range(i, n):
                cut.Ac.add(sorted_points[dim][k][dimensions])
                cut.Ac_points.append(sorted_points[dim][0][:-1])
            self.cuts.append(cut)

        i += self.agreement_param - 1 
        while n >= i:
            for dim in range(dimensions):
                cut = Cut()
                cut.init()
                for k in range(0, i):
                    cut.A.add(sorted_points[dim][k][dimensions])
                    cut.A_points.append(sorted_points[dim][k][:-1])
                    if k == i - 1:
                        cut.line_placement = (sorted_points[dim][k][0], dim)
                for k in range(i, n):
                    cut.Ac.add(sorted_points[dim][k][dimensions])
                    cut.Ac_points.append(sorted_points[dim][k][:-1])
                self.cuts.append(cut)
            i += self.agreement_param - 1
        
    def cut_axis(self):
        n = len(self.points)
        d = len(self.points[0])
        
        if type(self.points) == np.ndarray:
            self.points = self.points.tolist()
        self.points = [point + [z] for z, point in enumerate(self.points)]

        axis_values = [[] for _ in range(d)]  # 

        # Extract values for each dimension
        for point in self.points:
            for dim in range(d):
                axis_values[dim].append(point[dim])
     
        sorted_points = [self.sort_for_list(axis_values[dim], self.points) for dim in range(d)]

        i = self.agreement_param
        while n >= i + self.agreement_param:
            cuts = [Cut() for _ in range(d)]
            for dim in range(d):
                cuts[dim].init()
                for point in sorted_points[dim][:i]:
                    cuts[dim].A.add(point[-1])
                    cuts[dim].A_points.append(point[:-1])
                for point in sorted_points[dim][i:]:
                    cuts[dim].Ac.add(point[-1])
                    cuts[dim].Ac_points.append(point[:-1])
                self.cuts.append(cuts[dim])
            i += self.agreement_param

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
                    cuts[dim].A_points.append(sorted_points[dim][k][:-1])
                    if k == i - 1:
                        cuts[dim].line_placement = (sorted_points[dim][k][0], dim)
                for k in range(i, n):
                    cuts[dim].Ac.add(sorted_points[dim][k][dimensions])
                    cuts[dim].Ac_points.append(sorted_points[dim][k][:-1])
                self.cuts.append(cuts[dim])
            i += self.agreement_param
   





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

    def sort_for_list(self, axis_values, points):
        combined = list(zip(axis_values, points))
        sorted_combined = sorted(combined, key=lambda x: x[0])
        sorted_combined = [e[1] for e in sorted_combined]
        return sorted_combined
    

    # Evaluation functions

    def davies_bouldin_score(self, ground_truth, labels):
        score = davies_bouldin_score(ground_truth, labels)
        return score

    def nmi_score(self, ground_truth, labels):
        """
        Calculates the nmi score of the predicted tangles
        """
        nmi_score = normalized_mutual_info_score(ground_truth, labels)
        return nmi_score


# dimensionality reduction functions and input functions

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




