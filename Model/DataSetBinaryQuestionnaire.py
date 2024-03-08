import pandas as pd
import numpy as np
from scipy.linalg import svd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from Model.Cut import Cuts
from DataSet import extract_data



# we need something that can giv ecoordinates from the dimension_reduction_binary

def perform_pca(matrix, num_components=2):
    """
    Perform Principal Component Analysis (PCA) on a binary matrix.

    Parameters:
    - matrix: Binary matrix where rows represent people and columns represent questions.
    - num_components: Number of components to keep (default is 2 for 2D visualization).

    Returns:
    - pca_coordinates: DataFrame containing the PCA coordinates for each person.
    """

    # Initialize PCA with the specified number of components
    pca = PCA(n_components=num_components)

    # Fit the PCA model to the data and transform the data
    pca_result = pca.fit_transform(matrix)
    

    # Create a DataFrame with the PCA coordinates
    pca_coordinates = pd.DataFrame(data=pca_result, columns=[f'PC{i+1}' for i in range(num_components)])

    return pca_coordinates



# def calculate_explained_varince(S):
#         rho = (S * S) / (S * S).sum()
#         return rho



# def dimension_reduction_binary(csv_file_path):

#     # This can be put into DataSet.py
#     """ 
#     This function is used to reduce the dimension of binary questionnaires data set 
    
#     Parameters:
#     Takes a csv-file of zeros and ones. Zero represents "No" and ones represent "yes" to a question. 
#     Each row represent object.
#     Example of a file: 
#     "1,0,0,0,0,1,1,1,1
#      0,1,0,0,1,1,1,0,0
#      ....
#      ...
#     "

#     Returns:
#     Eigenvectors and Projections of the PCA
#     """

#     df = pd.read_csv(csv_file_path)
#     df = df.values
#     # eigen_vectors = np.linalg.eig(df)
    
#     N, _ = df.shape
   
#     # Subtract mean value from data
#     Y = df - np.ones((N, 1)) * df.mean(0)

#     # PCA by computing SVD of Y
#     _, S, Vh = svd(Y, full_matrices=False)
#     # scipy.linalg.svd returns "Vh", which is the Hermitian (transpose)
#     # of the vector V. So, for us to obtain the correct V, we transpose:
#     V = Vh.T


#     return S,V


def cost_function_binary(A, Ac, questionnaires):
    """ 
    This function is used to calculate the cost of cuts for binary questionnaires data set 
    
    Parameters:
    cuts of the dataset

    Returns:
    cost of each cut
    """
    print(A, Ac)
    n = questionnaires.shape[0]

    cost = 0
    norm = 1.0/(len(A)*(n-len(A)))

    # for i in range(len(A)):
    #     for j in range(len(Ac)):
           
    #        cost += sim(questionnaires(A[i]), questionnaires(Ac[j]))
    
    # cost = cost*norm


    for yes in A:
        for no in Ac:
            cost += sim(questionnaires[yes], questionnaires[no])

    cost = cost*norm

    return cost





    

def sim(v,w):

    """
    This function calculates the similarity between two vectors v and w.
    
    Parameters:
    v (numpy.ndarray): First vector
    w (numpy.ndarray): Second vector
    
    Returns:
    similarity (float): Similarity between v and w
    """
    similarity = 0
    for i in range(len(v)):
        if v[i] == w[i]:
            similarity += 1

    return similarity

               
        

def cut_generator_binary(nd_questionnaires):
    
    """ 
    This function is used to generate the cuts for binary questionnaires data set
    
    Parameters:
    Takes a csv-file of zeros and ones. Zero represents "No" and ones represent "yes" to a question. 
    Each row represent object.
    Example of a file: 
    "1,0,0,0,0,1,1,1,1
     0,1,0,0,1,1,1,0,0
     ....
     ...
    "

    Returns:
    cuts of the dataset
    example: cuts_y[i] = {p1,p2,p3}, this means that person p1, p2 and p3 answered yes to question i
    
    """
    
    
    cut_list = []
    cost = 0
    orientation = "None"

    num_of_participants = len(nd_questionnaires[0])
    num_of_quest = len(nd_questionnaires[1])

    cuts_list = []
    cuts_y = [set() for _ in range(num_of_quest)]
    cuts_n = [set() for _ in range(num_of_quest)]
     

    for i in range(num_of_quest):
        for j in range(num_of_participants):
            if nd_questionnaires[j][i] == 1:
                cuts_y[i].add(j)
            else:
                cuts_n[i].add(j)
        
        cost = cost_function_binary(cuts_y[i], cuts_n[i], nd_questionnaires)
        cuts_list.append(Cuts(i, cost, orientation, cuts_y[i], cuts_n[i]))

    return cuts_list


def order_cuts_by_cost(lst):
    """
    This function orders the cuts based on their cost in ascending order.
    
    Parameters:
    cuts_y (list): List of sets representing 'yes' cuts
    cuts_n (list): List of sets representing 'no' cuts
    questionnaires (numpy.ndarray): Array of questionnaires data
    
    Returns:
    ordered_cuts (list): List of cuts ordered by cost
    costs (list): List of costs corresponding to each cut
    """
    
    return sorted(lst, key=lambda x: x.cost)


def intersect_sets(set_list):
    """
    This function returns the intersection of all sets in the provided list.

    Parameters:
    set_list (list): List of sets to be intersected

    Returns:
    intersection_set (set): Intersection of all sets in the list
    """
    intersection_set = set_list[0]
    for s in set_list[1:]:
        intersection_set = intersection_set.intersection(s)
    return intersection_set

# {1,2,3,4}, 0.2, {643,3}
# {1,4,62,3}, 0.3, {0,90,23}
# {11,2,3}, 0.3, {14,80,33}


def remove_cost_and_id(list):
    new_list = []
    for [A, Ac, _, _] in list:
        new_list.append([A, Ac])

    
    














#  ----------------------------------------------test------------------------------------------------

def plot_2d_coordinates(dataframe, x_col='x', y_col='y', title="2D Plot", xlabel="X-axis", ylabel="Y-axis"):
    """
    Plot 2D coordinates from a Pandas DataFrame using matplotlib.

    Parameters:
    - dataframe: Pandas DataFrame containing 2D coordinates with specified column names.
    - x_col: Column name for x-axis coordinates (default is 'x').
    - y_col: Column name for y-axis coordinates (default is 'y').
    - title: Title for the plot (default is "2D Plot").
    - xlabel: Label for the x-axis (default is "X-axis").
    - ylabel: Label for the y-axis (default is "Y-axis").
    """

    plt.figure(figsize=(8, 8))
    plt.scatter(dataframe[x_col], dataframe[y_col])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

# dimension_reduction_binary("/Users/MortenHelsoe/Desktop/DTU/6. Semester/Bachelor Projekt/Tangle-lib-ORM/DTU-Tangle/csv_test/test.csv")

data = extract_data("/Users/MortenHelsoe/Desktop/DTU/6. Semester/Bachelor Projekt/Tangle-lib-ORM/DTU-Tangle/csv_test/test.csv")
res = cut_generator_binary(data)


order_res = order_cuts_by_cost(res)

for i in order_res:
    print(i.A)