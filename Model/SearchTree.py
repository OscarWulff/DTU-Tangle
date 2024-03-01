import math

class SearchTree:
    def __init__(self, propability) -> None:
        self.propability = propability
        self.children = []


def is_consistent(chosen_cuts, cut_layer, agrrement_parameter):
    intersection = set(chosen_cuts).intersection(cut_layer)
    return len(intersection) >= agrrement_parameter

    pass


def h(cost):
    return math.exp(-cost)

class Searchtree():

    def __init__(self, parent_node):
        self.parent_node = parent_node
        self.left_node = None
        self.right_node = None
        self.tangle = []




def calculate_propability(v, cut_object):
    """ 
    Calculate the propability of a cut 
    
    Parameters:
    Some point that the cut has to be oriented towards
    Cut of the dataset

    Returns:
    Propability of the cut
    """
    pass




def contracting_search_tree():
    """ 
    contracting the searchtree 
    
    Parameters:
    Search tree

    Returns:
    Condensed Search tree

    """
    pass 

def soft_clustering():
    """ 
    from a searchtree create a soft clustering of the objects 
    
    Parameters: 
    Search tree

    Returns:
    Soft clustering
    """
    pass

def hard_clustering():
    """   
    based on the soft clustering create a hard clustering of the objects 
    
    Parameters: 
    Soft clustering

    Returns:
    Hard clustering 

    """
    pass