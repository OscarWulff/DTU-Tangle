
import copy
from Model.DataSetBinaryQuestionnaire import DataSetBinaryQuestionnaire
from Model.SearchTree import *

from Model.DataSetFeatureBased import DataSetFeatureBased
from Model.DataSetGraph import DataSetGraph
from Model.DataSetFeatureBased import DataSetFeatureBased
from Model.DataSetGraph import DataSetGraph

from Model.DataType import DataType
from Model.DataType import DataType


# function to check if the cut is consistent with the tangle
def consistent(chosen_cut, node, agreement_parameter):
    # if-statement the tangle is empty, check if the chosen cut is greater than the agreement parameter
    if len(node.tangle) == 0:
        return len(chosen_cut) >= agreement_parameter
    # if-statement the tangle is of size 1, check if the intersection between the chosen cut and the tangle is greater than the agreement parameter
    elif len(node.tangle) == 1: 
        for a in node.tangle:
             l = set.intersection(a[0], chosen_cut)
             
             if len(l) < agreement_parameter:
                 return False
    # else-statement check if the intersection between the chosen cut and every two tangles is greater than the agreement parameter
    else:
        unique_tangles = []
        seen = set()
        for i in range(len(node.tangle)): 
            for j in range(len(node.tangle)): 
                if tuple(node.tangle[i][0]) not in seen:
                    unique_tangles.append(node.tangle[i])
                    seen.add(tuple(node.tangle[i][0]))
                if i != j:
                    l = set.intersection(set.intersection(node.tangle[i][0], node.tangle[j][0]), chosen_cut)
                    if len(l) < agreement_parameter: 
                        return False
        node.tangle = unique_tangles

    return True

# function to create the search tree
def create_searchtree(data : DataType):
    
    # function to create a child node and append a child node to the parent node
    def create_child(node, orientation, cut, id):
        child = Searchtree(node, node.cut_id+1)
        child.cut_orientation = orientation
        child.tangle = copy.deepcopy(node.tangle)
        node.cut = cut
        child.cuts.add(cut)
        child.condensed_oritentations.add(f"{child.cut_id}"+ child.cut_orientation)
        child.id = id
        if orientation == "L":
            child.tangle.append([cut.A])
            node.add_left_child(child)
        else: 
            child.tangle.append([cut.Ac])
            node.add_right_child(child)
        return child

    # create the root node
    root = Searchtree(None, 0)
    leaves = [root]

    # order the cuts
    cuts_ordered = data.order_function()
    #cuts_ordered = [cuts_ordered[1]]

    # for cut in cuts_ordered:
    #     print(cut.A)
    #     print("-----")
    #     print(cut.Ac)
    #     print("________________")


    # goes through the ordered cuts and creates the search tree
    id = 0
    for cutId, cut in enumerate(cuts_ordered, start=1):
        new_leaves = []
        cut.id = cutId
        for leaf in leaves:
            if consistent(cut.A, leaf, data.agreement_param):
                id += 1
                left_child = create_child(leaf, "L", cut, id)
                new_leaves.append(left_child)

            if consistent(cut.Ac, leaf, data.agreement_param):
                id += 1
                right_child = create_child(leaf, "R", cut, id)
                new_leaves.append(right_child)

        if not new_leaves:
            break     
        leaves = new_leaves
    return root





