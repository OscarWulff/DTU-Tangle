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
    for z, (x, y) in enumerate(zip(pc1, pc2)):
        Points.append((x, y, z))

    cuts = cut_generator_axis(Points, a, 0)
    costs = cost_function(cuts, Points)


def cost_function(cuts, points):
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
    for cut in cuts:

        sum_cost = 0.0
        left_oriented = cut[0]
        right_oriented = cut[1]

        # Calcualte the cost
        for left_or in left_oriented:
            for right_or in right_oriented:
                sum_cost += -(euclidean_distance(points[left_or][0], points[right_or][0], points[left_or][1], points[right_or][1]))
        
        costs.append(sum_cost)
    return costs

def cut_generator_axis(objects, a, axis):

    def sort_for_list(axis_values, points):
        combined = list(zip(axis_values, points))
        sorted_combined = sorted(combined, key=lambda x: x[0])
        return zip(*sorted_combined)

    n = len(objects)
    cuts = []

    values = []
    sorted_points = []

    for point in objects:
        values.append(point[axis])

  
    _, sorted_points = sort_for_list(values, objects)


    i = a
    while( n >= i + a ):
        cut = []
        A = set()
        for k in range(0, i):
            A.add(sorted_points[k][2])
        Ac = set()
        for k in range(i, n):
            Ac.add(sorted_points[k][2])
        cut.append(A)
        cut.append(Ac)
        cuts.append(cut)
        i += a

    return cuts


def order_function_featurebased(cuts, points):
    """ 
    order the cuts after cost 
    
    Paramaters:
    Cuts

    Returns: 
    An order of the cuts    
    """
    def sort_for_list(axis_values, points):
        combined = list(zip(axis_values, points))
        sorted_combined = sorted(combined, key=lambda x: x[0])
        return zip(*sorted_combined)
    
    costs = cost_function(cuts, points)
    _, cuts_ordered = sort_for_list(costs, cuts)
    return cuts_ordered


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
    
def euclidean_distance(x1, x2, y1, y2):
    distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return distance

def euclidean_distance_eulers(x1, x2, y1, y2):
    distance = math.e(-math.sqrt((x1 - x2)**2 + (y1 - y2)**2))
    return distance
