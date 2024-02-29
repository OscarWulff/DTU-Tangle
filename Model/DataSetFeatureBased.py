import math
import numpy as np
import pandas as pd
from scipy.linalg import svd
from DataSet import DataSet

class P_a():

    pc = []

    def __init__(self, n, a):
        number_cuts = math.ceil(n/a) - 1
        
        # creates matrix where row are objects and columns are cuts
        # 0 means left orientation and 1 means right orientation
        self.matrix = np.zeros((n, number_cuts))
    
    def add_right_orientation(self, object, feature):
        self.matrix[object, feature] = 1





class DataSetFeatureBased(DataSet):

    def __init__(self, a):
        self.a = a

    def dimension_reduction_feature_based(self, filename):
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

    def calculate_explained_varince(self, S):
        rho = (S * S) / (S * S).sum()
        return rho
    
    def order_projections(self, rho, V):
        indices = np.argsort(rho)
        indices = indices[::-1]
        
    def euclidean_distance(self, x1, y1, x2, y2):
        distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        return distance

    def cost_function(self, P:P_a, pc1, pc2):
        """ 
        This function is used to calculate the cost of cuts for feature based data set
        
        Parameters:
        cuts of the dataset

        Returns:
        cost of each cut
        """
        # The index of the array represent the cut and the value represent the cost
        costs = []

        # We transpose it such that we can go through the column instead of row
        for column in P.matrix.T:

            sum_cost = 0
            left_oriented = []
            right_oriented = []

            # Opdel objects for hver orientation
            for i, orientation in enumerate(column):
                if orientation == 0: 
                    left_oriented.append(i)
                else:
                    right_oriented.append(i)

            # Calcualte the cost
            for left_or in left_oriented:
                for right_or in right_oriented:
                    sum_cost += -(self.euclidean_distance(pc1[left_or], pc1[right_or], pc2[left_or], pc2[right_or]))
            
            costs.append(sum_cost)
        
        return costs

    def each_axis_cuts(self, P : P_a, n, pc):
        cut_numb = 0
        slice_numb = self.a
        while(slice_numb < n):
            slice_value = P.pc[slice_numb]

            for i in range(0, n):
                if slice_value <= pc[i]:
                    P.add_right_orientation(i, cut_numb)

            cut_numb += 1
            slice_numb += self.a
        return P


    def cut_generator(self, pc1, pc2):
        """
        This function is used to generate the cuts for feature based data set
        
        Parameters:
        Projections of the PCA
        
        Returns:
        cuts of the dataset
        """
        
        n = len(pc1)
        P_X = P_a(n, self.a)
        P_Y = P_a(n, self.a)

        P_X.pc, P_Y.pc = sorted(pc1), sorted(pc2)
        
        P_X = self.each_axis_cuts(P_X, n, pc1)
        P_Y = self.each_axis_cuts(P_Y, n, pc2)

        return P_X, P_Y



featurebased = DataSetFeatureBased(1)

S, V = featurebased.dimension_reduction_feature_based("iris.csv")
rho = featurebased.calculate_explained_varince(S)
pc1 = V[:, 0]
pc2 = V[:, 1]

PX, PY = featurebased.cut_generator(pc1, pc2)

print(featurebased.cost_function(PX, pc1, pc2))


print(PX.matrix)
print(PY.matrix)

