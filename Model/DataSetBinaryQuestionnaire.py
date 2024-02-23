import pandas as pd
import numpy as np
from scipy.linalg import svd

def calculate_explained_varince(S):
        rho = (S * S) / (S * S).sum()
        return rho

def get_questionnaires(csv_file_path):
    df = pd.read_csv(csv_file_path).values 
    return df

def dimension_reduction_binary(csv_file_path):

    # This can be put into DataSet.py
    """ 
    This function is used to reduce the dimension of binary questionnaires data set 
    
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
    Eigenvectors and Projections of the PCA
    """

    df = pd.read_csv(csv_file_path)
    df = df.values
    # eigen_vectors = np.linalg.eig(df)
    
    N, _ = df.shape
   
    # Subtract mean value from data
    Y = df - np.ones((N, 1)) * df.mean(0)

    # PCA by computing SVD of Y
    _, S, Vh = svd(Y, full_matrices=False)
    # scipy.linalg.svd returns "Vh", which is the Hermitian (transpose)
    # of the vector V. So, for us to obtain the correct V, we transpose:
    V = Vh.T


    return S,V

   




    pass


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

               
        

def cut_generator_binary(csv_file_path):
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
    nd_questionnaires = pd.read_csv(csv_file_path).values
   

    num_of_participants = len(nd_questionnaires[0])
    num_of_quest = len(nd_questionnaires[1])

    cuts_y = [set() for _ in range(num_of_quest)]
    cuts_n = [set() for _ in range(num_of_quest)]
     

    for i in range(num_of_participants):
        for j in range(num_of_quest):
            if nd_questionnaires[i][j] == 1:
                cuts_y[j].add(i)
            else:
                cuts_n[j].add(i)
        

    return cuts_y, cuts_n


# dimension_reduction_binary("/Users/MortenHelsoe/Desktop/DTU/6. Semester/Bachelor Projekt/Tangle-lib-ORM/DTU-Tangle/csv_test/test.csv")

cy, cn = cut_generator_binary("/Users/MortenHelsoe/Desktop/DTU/6. Semester/Bachelor Projekt/Tangle-lib-ORM/DTU-Tangle/csv_test/test.csv")
q = get_questionnaires("/Users/MortenHelsoe/Desktop/DTU/6. Semester/Bachelor Projekt/Tangle-lib-ORM/DTU-Tangle/csv_test/test.csv")

for i in range(len(cy)):
   cost_of_cut = cost_function_binary(cy[i], cn[i], q)
   print(cost_of_cut)