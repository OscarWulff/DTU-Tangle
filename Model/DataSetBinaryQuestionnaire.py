import pandas as pd
import numpy as np
from scipy.linalg import svd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# from Model.Cut import Cuts
from Model.DataSet import extract_data
from Model.DataType import DataType
from Model.Cut import Cut

class DataSetBinaryQuestionnaire(DataType):
    def __init__(self, agreement_param, cuts=[], search_tree=None):
        super().__init__(agreement_param, cuts, search_tree)
        

    def order_function(self):
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

        return sorted(self.cuts, key=lambda x: x.cost)

    
    def cut_generator_binary(self,nd_questionnaires):
        
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
        nd_questionnaires = nd_questionnaires[1:]
        cost = 0
        orientation = "None"

        num_of_participants = nd_questionnaires.shape[0]
        num_of_quest = nd_questionnaires.shape[1]

        print(nd_questionnaires)
        
        # print(num_of_participants)
        # print(num_of_quest)

        cuts_list = []
        cuts_y = [set() for _ in range(num_of_quest)]
        cuts_n = [set() for _ in range(num_of_quest)]
        

        for i in range(num_of_quest):
            self.cuts.append(Cut(i, cost, orientation, set(), set()))
            for j in range(num_of_participants):
                
                if nd_questionnaires[j][i] == 1:
                    self.cuts[i].A.add(j) 
                else:
                   
                    self.cuts[i].Ac.add(j)

            
            cost = cost_function_binary(self.cuts[i].A, self.cuts[i].Ac, nd_questionnaires)
            self.cuts[i].cost = cost
            # cuts_list.append(Cuts(i, cost, orientation, cuts_y[i], cuts_n[i]))

        return self





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
    matrix = matrix[1:]
    # print(matrix)

    # Initialize PCA with the specified number of components
    pca = PCA(n_components=num_components)

    # Fit the PCA model to the data and transform the data
    pca_result = pca.fit_transform(matrix)
    

    # Create a DataFrame with the PCA coordinates
    pca_coordinates = pd.DataFrame(data=pca_result, columns=[f'PC{i+1}' for i in range(num_components)])

    return pca_coordinates


import pandas as pd
from sklearn.manifold import TSNE

import pandas as pd
from sklearn.manifold import TSNE

def perform_tsne(matrix, num_components=2, random_state=42, perplexity=None, learning_rate=200, n_iter=3000, early_exaggeration=12):
    """
    Perform t-Distributed Stochastic Neighbor Embedding (t-SNE) on a binary matrix.

    Parameters:
    - matrix: Binary matrix where rows represent individuals and columns represent questions.
    - num_components: Number of components to keep for visualization (default is 2 for 2D visualization).
    - random_state: Seed for random number generator to make the results reproducible (default is 42).
    - perplexity: The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms. Larger datasets usually require a larger perplexity. Consider selecting a value between 5 and 50. Default is calculated as min(30, n_samples - 1) / 3.
    - learning_rate: The learning rate for t-SNE, which can significantly affect the outcome. Common values range from 10 to 1000.
    - n_iter: The number of iterations for optimization. More complex datasets might require more iterations.
    - early_exaggeration: Controls how tight natural clusters in the original space are in the embedded space and how much space will be between them.

    Returns:
    - tsne_coordinates: DataFrame containing the t-SNE coordinates for each individual.
    """
    # Remove potential header or first row if necessary
    matrix = matrix[1:]
    
    # Automatically adjust perplexity if not set
    if perplexity is None:
        perplexity = min(30, len(matrix) - 1) / 3
    
    # Ensure perplexity is less than the number of samples
    perplexity = min(perplexity, len(matrix) - 1)
    
    # Initialize t-SNE with the specified parameters
    tsne = TSNE(n_components=num_components, 
                random_state=random_state, 
                perplexity=perplexity, 
                learning_rate=learning_rate, 
                n_iter=n_iter, 
                early_exaggeration=early_exaggeration)
    
    # Run t-SNE on the data
    tsne_result = tsne.fit_transform(matrix)
    
    # Create a DataFrame with the t-SNE coordinates
    tsne_coordinates = pd.DataFrame(data=tsne_result, columns=[f'Dim{i+1}' for i in range(num_components)])

    return tsne_coordinates




# def calculate_explained_varince(S):
#         rho = (S * S) / (S * S).sum()
#         return rho







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

               
def cost_function_binary(A, Ac, questionnaires):
    """ 
    Calculate the cost of cuts for binary questionnaires dataset.
    
    Parameters:
    - A: Indices of questionnaires in one subset.
    - Ac: Indices of questionnaires in the complementary subset.
    - questionnaires: The dataset of questionnaires.
    
    Returns:
    - The cost of the cut.
    """
    n = questionnaires.shape[0]
    
    # Check to prevent division by zero
    if len(A) == 0 or len(A) == n:
        return float('inf')  # Return a high cost to indicate an invalid cut
    
    norm = 1.0 / (len(A) * (n - len(A)))
    cost = 0
    
    for yes in A:
        for no in Ac:
            # Assuming `sim` is a function that calculates similarity between two questionnaires
            cost += sim(questionnaires[yes], questionnaires[no])
    
    cost *= norm  # Apply normalization to the cost

    return cost







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




def plot_2d_coordinates(dataframe, title="2D Plot", xlabel="X-axis", ylabel="Y-axis"):
    """
    Plot 2D coordinates from a Pandas DataFrame using matplotlib.

    Parameters:
    - dataframe: Pandas DataFrame containing 2D coordinates with specified column names.
    - title: Title for the plot (default is "2D Plot").
    - xlabel: Label for the x-axis (default is "X-axis").
    - ylabel: Label for the y-axis (default is "Y-axis").
    """

    fig = plt.figure(figsize=(8, 8))
    plt.scatter(dataframe['PC1'], dataframe['PC2'])  # Assuming PC1 and PC2 are the column names of your principal components
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    return fig

