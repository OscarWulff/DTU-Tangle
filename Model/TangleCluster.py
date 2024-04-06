
from Model.DataSetBinaryQuestionnaire import DataSetBinaryQuestionnaire
from Model.DataSet import extract_data
from Model.SearchTree import *

from Model.DataSetFeatureBased import DataSetFeatureBased
from Model.DataSetGraph import DataSetGraph
from Model.DataSetFeatureBased import DataSetFeatureBased
from Model.DataSetGraph import DataSetGraph

from Model.DataType import DataType
from Model.DataType import DataType

def consistent(chosen_cut, node, agreement_parameter):
    if len(node.tangle) == 0:
        return len(chosen_cut) >= agreement_parameter
    elif len(node.tangle) == 1: 
        for a in node.tangle:
             l = set.intersection(a[0], chosen_cut)
             if len(l) < agreement_parameter:
                 return False
    else:
        unique_tangles = []
        seen = set()
        for i in range(len(node.tangle)): 
            for j in range(len(node.tangle)): 
                if tuple(node.tangle[i][0]) not in seen:
                    unique_tangles.append(node.tangle[i])
                    seen.add(tuple(node.tangle[i][0]))
                if i != j:
                    ### muligvis gøre sådan at de samme tangles ikke bliver tilføjet igen
                    l = set.intersection(set.intersection(node.tangle[i][0], node.tangle[j][0]), chosen_cut)
                    if len(l) < agreement_parameter: 
                        return False
        node.tangle = unique_tangles

    return True

# Skal ændres i forhold til 'data'
def create_searchtree(data : DataType):
    """ 
    create searchtree from the cuts 
    
    Paramaters:
    cuts

    Returns: 
    Search tree
    """

    def create_child(node, orientation, cut, id):
        child = Searchtree(node, node.cut_id+1)
        child.cut_orientation = orientation
        child.tangle += node.tangle
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


    root = Searchtree(None, 0)

    leaves = [root]
    cuts_ordered = data.order_function()
    for index, cut in enumerate(cuts_ordered, start=1):
        print(f"cut.id = {index} - A: {cut.A} - Ac: {cut.Ac} - cost: {cut.cost}")

    for index, cut in enumerate(cuts_ordered, start=1):
        print(f"cut.id = {index} - A: {cut.A} - Ac: {cut.Ac} - cost: {cut.cost}")

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






