import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import svd
from DataSet import DataSet



def main():
    a = 1

    _, V = dimension_reduction_feature_based("iris.csv")
    pc1 = V[:, 0]
    pc2 = V[:, 1]

    Points = []
    for x, y in zip(pc1, pc2):
        Points.append((x, y))

    P, points = cut_generator_axis(Points, a, 0)

    costs = cost_function(P, points)
    print(P)
    print(costs)

def sort_points(axis_values, points):
    combined = list(zip(axis_values, points))
    sorted_combined = sorted(combined, key=lambda x: x[0])
    return zip(*sorted_combined)

def cut_generator_axis(objects, a, axis):
    n = len(objects)
    number_cuts = math.ceil(n/a) - 1
    # creates matrix where row are objects and columns are cuts
    # 0 means left orientation and 1 means right orientation
    matrix = np.zeros((n, number_cuts))

    values = []
    sorted_points = []

    for point in objects:
        values.append(point[axis])
  
    _, sorted_points = sort_points(sorted(values), objects)
    k = 1
    cut_numb = 0
    while(n - k >= a):
        for i in range(k, n):
            matrix[i, cut_numb] = 1
        k += a
        cut_numb += 1

    return matrix, sorted_points

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
    
def euclidean_distance(x1, y1, x2, y2):
    distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return distance

def euclidean_distance_eulers(x1, y1, x2, y2):
    distance = math.e(-math.sqrt((x1 - x2)**2 + (y1 - y2)**2))
    return distance

def cost_function(P_matrix, points):
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
    for column in P_matrix.T:

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
                sum_cost += -(euclidean_distance(points[left_or][0], points[right_or][0], points[left_or][1], points[right_or][1]))
        
        costs.append(sum_cost)

    return costs

def order_function(P_matrix, points):
    """ 
    order the cuts after cost 
    
    Paramaters:
    Cuts

    Returns: 
    An order of the cuts    
    """
    costs = cost_function(P_matrix, points)

    costs_order = np.argsort(costs)
    
    return costs_order




class Searchtree():

    def __init__(self, parent_node):
        self.parent_node = parent_node
        self.left_node = None
        self.right_node = None
        self.tangle = []


def create_searchtree( P_matrix ):
    """ 
    create searchtree from the ordered cuts 
    
    Paramaters:
    ordered cuts

    Returns: 
    Search tree
    """

    root = Searchtree(None)


    


main()