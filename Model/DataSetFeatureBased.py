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



    def min_distance_cost(self):
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

    def pairwise_cost(self):
        for cut in self.cuts:

            sum_cost = 0.0
            left_oriented = cut.A
            right_oriented = cut.Ac

            # Calculate the cost
            for left_or in left_oriented:
                for right_or in right_oriented:
                    np.linalg.norm(self.points[left_or] - self.points[right_or])

                    sum_cost += -(self.euclidean_distance(self.points[left_or][0], self.points[right_or][0], self.points[left_or][1], self.points[right_or][1])/(len(cut.A)*len(cut.Ac))) 
            
            cut.cost = sum_cost

    def cut_generator_axis_solveig(self):
        self.cuts = []
        n = len(self.points)

        dimensions = len(self.points[0])
        # Add index to keep track of original order
        if type(self.points[0]) == tuple:
            self.points = [point + (z, ) for z, point in enumerate(self.points)]
        else: 
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


    def cut_generator_axis_dimensions(self):
        self.cuts = []
        n = len(self.points)

        dimensions = len(self.points[0])
        # Add index to keep track of original order
        if type(self.points[0]) == tuple:
            self.points = [point + (z, ) for z, point in enumerate(self.points)]
        else: 
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
    X = X[:20, :-1]
    X = X.astype(float)
    return X

def calculate_explained_varince(S):
    rho = (S * S) / (S * S).sum()
    return rho




