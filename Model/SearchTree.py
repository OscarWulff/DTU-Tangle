import math
class Searchtree():

    def __init__(self, parent_node, cut_id):
        self.parent_node : Searchtree = parent_node
        self.left_node : Searchtree = None
        self.right_node : Searchtree = None
        self.tangle = []
        self.cut_id = cut_id
        self.cut_orientation = ""
        self.leaf = True
        self.characterizing_cuts = []
        self.condensed_oritentations = []
    
    def add_left_child(self, left_child ):
        self.left_node = left_child
        self.leaf = False

    def add_right_child(self, right_child ):
        self.right_node = right_child
        self.leaf = False

def h(cost):
    return math.exp(-cost)




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
    
    
def condense_tree(root: Searchtree):
    nodes = []
    leaves = []

    def traverse(current_node):
        if current_node is not None:
            if current_node.leaf:
                leaves.append(current_node)
            else:
                nodes.append(current_node)
                traverse(current_node.left_node)
                traverse(current_node.right_node)

    traverse(root)

     # Ask Karl how you prune a tree
    def prune(leaf, length):
        pass

    for leaf in leaves: 
        prune(leaf, 1)

   


    def condense_leaf(leaf):
        if leaf.parent_node != None:
            if leaf.parent_node.left_node == None or leaf.parent_node.right_node == None:
                leaf.condensed_oritentations.append(f"{leaf.parent_node.cut_id}"+ leaf.parent_node.cut_orientation)
                if leaf.parent_node.parent_node == None: 
                    leaf.parent_node = None
                else: 
                    if leaf.parent_node.parent_node.left_node == leaf.parent_node:
                        leaf.parent_node.parent_node.left_node = leaf
                    else:
                        leaf.parent_node.parent_node.right_node = leaf
                    leaf.parent_node = leaf.parent_node.parent_node 
                    condense_leaf(leaf)
            elif leaf.parent_node.left_node != None and leaf.parent_node.right_node != None:
                condense_leaf(leaf.parent_node)
                

    for leaf in leaves: 
        condense_leaf(leaf)
    
    return root



def contracting_search_tree(node : Searchtree):
    """ 
    contracting the searchtree 
    
    Parameters:
    Search tree

    Returns:
    contracted Search tree

    """
    # Vi mangler at tilføje nodens selv til condensed_oritentations.
    # Herefter vil få den rigtige contradiction.

    if node != None:
        if node.leaf == False:
            contracting_search_tree(node.left_node)
            contracting_search_tree(node.right_node)
            for co in node.left_node.condensed_oritentations:
                if co in node.right_node.condensed_oritentations:
                    node.condensed_oritentations.append(co)
                else:
                    cut_nr = co[0]
                    cut_or = co[1]
                    if cut_or == "L":
                        if cut_nr+"R" in node.right_node.condensed_oritentations:
                            node.characterizing_cuts.append(co)
                    else:
                        if cut_nr+"L" in node.right_node.condensed_oritentations:
                            node.characterizing_cuts.append(co)
                 

def p_l(v):
    pass

def p_r(v):
    return 1 - p_l(v)



def soft_clustering(node, v, accumulated, softClustering):
    """ 
    from a searchtree create a soft clustering of the objects 
    
    Parameters: 
    node, point, accumumlated, dictionary of the tangles

    Returns:
    Soft clustering of the point
    """
    softClustering = {}
    if node.leaf:
        softClustering[node.cut_id] = accumulated
    else:
        soft_clustering(node.left_node, v, accumulated * p_l(v), softClustering)
        soft_clustering(node.right_node, v, accumulated * p_r(v), softClustering)

    return softClustering


def hard_clustering(softClustering):
    """   
    based on the soft clustering create a hard clustering of the point 
    
    Parameters: 
    Soft clustering

    Returns:
    Hard clustering 

    """
    max_prob = 0
    hard_cluster = 0
    for cluster, propability in softClustering.items():
        if propability > max_prob:
            hard_cluster = cluster

    return hard_cluster

def print_tree(node, indent=0, prefix="Root: "):
    if node is not None:
        print("  " * indent + prefix + f"{node.cut_id}")
        if node.left_node is not None or node.right_node is not None:
            print_tree(node.left_node, indent + 1, "L--- ")
            print_tree(node.right_node, indent + 1, "R--- ")







