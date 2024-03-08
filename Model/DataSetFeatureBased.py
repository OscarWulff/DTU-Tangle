import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import svd
from DataType import DataType
from Cut import Cut

class DataSetFeatureBased(DataType):

    def __init__(self,agreement_param, cuts=[], search_tree=None):
        super().__init__(agreement_param, cuts, search_tree)
        self.points = []
        self.initialize()

    def initialize(self):
        _, V = dimension_reduction_feature_based("iris.csv")
        pc1 = V[:, 0]
        pc2 = V[:, 1]

        for z, (x, y) in enumerate(zip(pc1, pc2)):
            self.points.append((x, y, z))

        self.cut_generator_axis(0)
        self.cost_function()

    def cost_function(self):
        """ 
        This function is used to calculate the cost of cuts for feature based data set
        
        Parameters:
        cuts of the dataset

        Returns:
        cost of each cut
        """

        # We transpose it such that we can go through the column instead of row
        for cut in self.cuts:

            sum_cost = 0.0
            left_oriented = cut.A
            right_oriented = cut.Ac

            print(left_oriented)
            print(right_oriented)

            # Calcualte the cost
            for left_or in left_oriented:
                for right_or in right_oriented:
                    sum_cost += -(self.euclidean_distance(self.points[left_or][0], self.points[right_or][0], self.points[left_or][1], self.points[right_or][1]))
            
            cut.cost = sum_cost

    def cut_generator_axis(self, axis):

        n = len(self.points)

        values = []
        sorted_points = self.points

        for point in self.points:
            values.append(point[axis])
        
        _, sorted_points = self.sort_for_list(values, sorted_points)

        print(sorted_points)

        i = self.agreement_param
        while( n >= i + self.agreement_param ):
            cut = Cut()
            for k in range(0, i):
                print(f"A : {sorted_points[k][2]}")
                cut.A.add(sorted_points[k][2])
            for k in range(i, n):
                print(f"Ac : {sorted_points[k][2]}")
                cut.Ac.add(sorted_points[k][2])
            print("____")
            self.cuts.append(cut)
            i += self.agreement_param

    
    def order_function_featurebased(self):
        """ 
        order the cuts after cost 
        
        Paramaters:
        Cuts

        Returns: 
        An order of the cuts    
        """
        
        self.cost_function()

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

def dimension_reduction_feature_based(filename):
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
        
        X = X[:, :-1]
        X = X.astype(float)

        N, _ = X.shape

        # Subtract mean value from data
        Y = X - np.ones((N, 1)) * X.mean(0)

        # PCA by computing SVD of Y
        _, S, Vh = svd(Y, full_matrices=False)
        # scipy.linalg.svd returns "Vh", which is the Hermitian (transpose)
        # of the vector V. So, for us to obtain the correct V, we transpose:
        V = Vh.T

        return S, V

def calculate_explained_varince(S):
    rho = (S * S) / (S * S).sum()
    return rho

def order_projections(rho, V):
    indices = np.argsort(rho)
    indices = indices[::-1]

def main():
    test = DataSetFeatureBased(1)
    for cut in test.cuts:
        print(cut.A)
        print(cut.Ac)
        print(cut.cost)
        

main()


