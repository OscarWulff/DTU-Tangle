import math
import random
from matplotlib import pyplot as plt
import networkx as nx
from Model.Cut import Cut

class Searchtree():

    def __init__(self, parent_node, cut_id):
        self.parent_node : Searchtree = parent_node
        self.left_node : Searchtree = None
        self.right_node : Searchtree = None
        self.tangle = []
        self.cut : Cut = None
        self.cuts : set[Cut] = set()
        self.cut_id = cut_id
        self.cut_orientation = ""
        self.leaf = True
        self.characterizing_cuts = set()
        self.condensed_oritentations = set()
        self.id = 0
    
    def add_left_child(self, left_child ):
        self.left_node = left_child
        self.leaf = False

    def add_right_child(self, right_child ):
        self.right_node = right_child
        self.leaf = False

    
def h(cost):
    return math.exp(-cost)

    
def condense_tree(root: Searchtree):
    nodes = []
    leaves = []
    prune_length = 1

    def prune_tree(node, max_branch_length):
        """ 
        Kan max bruges til at 'max_branch_length' op til 2
        
        """
        current_node = node
        
        while max_branch_length > 0:
            if current_node.parent_node == root:
                if root.left_node == current_node:
                    root.left_node = None
                else: 
                    root.right_node = None
            if max_branch_length == 0: 
                break
            max_branch_length -= 1
            current_node = current_node.parent_node

    def traverse(current_node):
        if current_node is not None:
            if current_node.leaf:
                leaves.append(current_node)
            else:
                nodes.append(current_node)
                traverse(current_node.left_node)
                traverse(current_node.right_node)

    traverse(root)

    #for leaf in leaves:
    #    prune_tree(leaf, prune_length)

    def condense_leaf(leaf):
        nonlocal root
        if leaf.parent_node != None:
            if leaf.parent_node.left_node == None or leaf.parent_node.right_node == None:
                if leaf.parent_node.parent_node == None: 
                    root = leaf
                    leaf.parent_node = None
                else: 
                    if leaf.parent_node.parent_node.left_node == leaf.parent_node:
                        leaf.parent_node.parent_node.left_node = leaf
                    else:
                        leaf.parent_node.parent_node.right_node = leaf
                    leaf.condensed_oritentations.add(f"{leaf.parent_node.cut_id}"+ leaf.parent_node.cut_orientation)
                    leaf.cuts.add(leaf.parent_node.cut)
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

    if node != None:
        if node.leaf == False:
            contracting_search_tree(node.left_node)
            contracting_search_tree(node.right_node)
            if node.left_node != None and node.right_node != None:
                for co in node.left_node.condensed_oritentations:
                    if co in node.right_node.condensed_oritentations:
                        node.condensed_oritentations.add(co)
                    else:     
                        cut_nr = co[0]
                        cut_or = co[1]
                        if cut_or == "L":
                            if cut_nr+"R" in node.right_node.condensed_oritentations:
                                for cut in node.right_node.cuts:
                                    if str(cut.id) == cut_nr:
                                        node.characterizing_cuts.add(cut)
                        else:
                            if cut_nr+"L" in node.right_node.condensed_oritentations:
                                for cut in node.left_node.cuts:
                                    if str(cut.id) == cut_nr:
                                        node.characterizing_cuts.add(cut)
                 


def soft_clustering(node, v, accumulated : float, softClustering = {}):
    """ 
    from a searchtree create a soft clustering of the objects 
    
    Parameters: 
    node, point, accumumlated, dictionary of the tangles

    Returns:
    Soft clustering of the point
    """
    
    def p_l(node, v):
        sum_right = 0
        sum_all = 0
        for cut in node.characterizing_cuts:
            # Lige tager vi bare cuttet, men vi skal tage the cost of the cut
            sum_all += h(cut.cost)

            if v in cut.A: 
                sum_right += h(cut.cost)
        if sum_all == 0: 
            return 0
        return (sum_right/sum_all)

    if node.leaf:
        softClustering[node.id] = accumulated
    else:
        pl = p_l(node, v)
        if node.left_node != None: 
            soft_clustering(node.left_node, v, accumulated * pl, softClustering)
        if node.right_node != None: 
            soft_clustering(node.right_node, v, accumulated * (1-pl), softClustering)

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
    node_id = 0
    for id, propability in softClustering.items():
        if propability > max_prob:
            node_id = id
            max_prob = propability

    return node_id, max_prob

def print_tree(node, indent=0, prefix="Root: "):
    if node is not None:
        print("  " * indent + prefix + f"{node.cut_id}")
        if node.left_node is not None or node.right_node is not None:
            print_tree(node.left_node, indent + 1, "L--- ")
            print_tree(node.right_node, indent + 1, "R--- ")


            
def plot_search_tree(tree, pos=None, parent_name=None, graph=None, x_pos=0, y_pos=0, horizontal_gap=1.0, level_height=1.0):
    if graph is None:
        graph = nx.Graph()
    if pos is None:
        pos = {tree.id: (x_pos, y_pos)}

    if parent_name is not None:
        graph.add_edge(parent_name, tree.id)

    x, y = pos[tree.id]

    if tree.left_node is not None:
        x_new_left = x - horizontal_gap / 2
        y_new_left = y - level_height
        pos[tree.left_node.id] = (x_new_left, y_new_left)
        plot_search_tree(tree.left_node, pos, tree.id, graph, x_new_left, y_new_left, horizontal_gap / 2, level_height)

    if tree.right_node is not None:
        x_new_right = x + horizontal_gap / 2
        y_new_right = y - level_height
        pos[tree.right_node.id] = (x_new_right, y_new_right)
        plot_search_tree(tree.right_node, pos, tree.id, graph, x_new_right, y_new_right, horizontal_gap / 2, level_height)

    return graph, pos

def show_tree(tree):
    graph, pos = plot_search_tree(tree)
    nx.draw(graph, pos=pos, with_labels=True, node_size=700, node_color="lightblue", font_size=8, font_weight="bold", font_color="black", edge_color="gray", linewidths=1, alpha=0.7)
    plt.show()


def generate_random_color():
    """
    Generate a random color represented as a tuple of RGB values.
    """
    r = random.randint(0, 255)  # Red component
    g = random.randint(0, 255)  # Green component
    b = random.randint(0, 255)  # Blue component
    return (r, g, b)

def generate_color_dict(data, tree):
    vals = []

    for i in range(data.shape[0]-1):
        soft = soft_clustering(tree, i, 1, {})
        vals.append(hard_clustering(soft)[0])

    
    set_vals = set(vals)
    color_dict = {}

    for i in set_vals:
        color_dict[i] = generate_random_color()
    
   


    return color_dict, vals, set_vals


