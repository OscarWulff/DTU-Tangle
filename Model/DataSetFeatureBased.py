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
        
        self.cut_generator_axis()
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
                    sum_cost += -1/np.power(self.euclidean_distance(self.points[left_or][0], self.points[right_or][0], self.points[left_or][1], self.points[right_or][1]), 2)
            
            cut.cost = sum_cost

    def cost_function(self):
        pass

    def pairwise_cost(self):
        for cut in self.cuts:

            sum_cost = 0.0
            left_oriented = cut.A
            right_oriented = cut.Ac

            # Calculate the cost
            for left_or in left_oriented:
                for right_or in right_oriented:
                    sum_cost += -(self.euclidean_distance(self.points[left_or][0], self.points[right_or][0], self.points[left_or][1], self.points[right_or][1]))
            
            cut.cost = sum_cost


    def cut_generator_axis(self):

        self.cuts = []
        n = len(self.points)
        x_values = []                
        y_values = []

        for point in self.points:
            x_values.append(point[0]) 
            y_values.append(point[1])
        
        _, sorted_points_x = self.sort_for_list(x_values, self.points)
        _, sorted_points_y = self.sort_for_list(y_values, self.points)
        i = self.agreement_param
        while( n >= i + self.agreement_param ):
            cut_x = Cut()
            cut_x.A = set()
            cut_x.Ac = set()
            cut_y = Cut()
            cut_y.A = set()
            cut_y.Ac = set()
            for k in range(0, i):
                cut_x.A.add(sorted_points_x[k][2])
                cut_y.A.add(sorted_points_y[k][2])
                if k == i-1:
                    cut_x.line_placement = (sorted_points_x[k][0], "x")
                    cut_y.line_placement = (sorted_points_y[k][1], "y")
            for k in range(i, n):
                cut_x.Ac.add(sorted_points_x[k][2])
                cut_y.Ac.add(sorted_points_y[k][2])
            self.cuts.append(cut_x)
            self.cuts.append(cut_y)
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
        
        _, cuts_ordered = self.sort_for_list(costs, self.cuts)
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
        return zip(*sorted_combined)

def pca(filename):
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
        df = pd.read_csv(filename)
        X = df.values

        X = X[:20, :-1]
        X = X.astype(float)
        N, _ = X.shape
        # Subtract mean value from data
        Y = X - np.ones((N, 1)) * X.mean(0)

        # PCA by computing SVD of Y
        _, S, Vh = svd(Y, full_matrices=False)
        # scipy.linalg.svd returns "Vh", which is the Hermitian (transpose)
        # of the vector V. So, for us to obtain the correct V, we transpose:
        
        X_projected = np.dot(Y, Vh[:2, :].T)

        return S, X_projected

def tsne(filename):
    df = pd.read_csv(filename)
    X = df.values
    X = X[:20, :-1]
    X = X.astype(float)

    perplexity = min(30, len(X) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity)
    data = tsne.fit_transform(X)

    return data

    

def calculate_explained_varince(S):
    rho = (S * S) / (S * S).sum()
    return rho

def order_projections(rho, V):
    indices = np.argsort(rho)
    indices = indices[::-1]



