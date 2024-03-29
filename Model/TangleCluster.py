
from Model.DataSetBinaryQuestionnaire import DataSetBinaryQuestionnaire
from Model.DataSet import extract_data
from Model.SearchTree import *

from Model.DataSetFeatureBased import DataSetFeatureBased
from Model.DataSetGraph import DataSetGraph
from Model.DataSetFeatureBased import DataSetFeatureBased
from Model.DataSetGraph import DataSetGraph

from Model.DataType import DataType
from Model.DataType import DataType

def consistent(chosen_cut, tangles, agrrement_parameter):
    if len(tangles) == 0:
        return len(chosen_cut) >= agrrement_parameter
    elif len(tangles) == 1: 
        for a in tangles:
             l = set.intersection(a[0], chosen_cut)
             if len(l) < agrrement_parameter:
                 return False
    else:
        for a in tangles: 
            for b in tangles: 
                if a[0] != b[0]: 
                    l = set.intersection(set.intersection(a[0], b[0]), chosen_cut)
                    if len(l) < agrrement_parameter: 
                        return False
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

    id = 0
    for cutId, cut in enumerate(cuts_ordered, start=1):
        new_leaves = []
        cut.id = cutId
        for leaf in leaves:
            if consistent(cut.A, leaf.tangle, data.agreement_param):
                id += 1
                left_child = create_child(leaf, "L", cut, id)
                new_leaves.append(left_child)

            if consistent(cut.Ac, leaf.tangle, data.agreement_param):
                id += 1
                right_child = create_child(leaf, "R", cut, id)
                new_leaves.append(right_child)

        if not new_leaves:
            break     
        leaves = new_leaves
    return root


       



# ---------------------------------------------------------- test ----------------------------------------------------------


# cuts = [[{0}, {1,2}], [{1}, {0, 2}], [{2}, {0,1}]]
# points = [(1, 1), (2, 2), (3, 3)]
# new_tree = create_searchtree(cuts, 1, points)
# Example usage:
# Constructing a simple tree

# root = Searchtree(None, 0)
# left_child = Searchtree(root, 1)
# left_child.cut_orientation ="L"
# left_child.cut = Cut()
# left_child.cuts.add(left_child.cut)


# root.add_left_child(Searchtree(root, "1L"))
# root.add_right_child(Searchtree(root, "1R"))
# root.right_node.add_right_child(Searchtree(root.right_node, "2R"))


# split = root.right_node.right_node

# split.add_right_child(Searchtree(split, "3R"))
# split.right_node.add_right_child(Searchtree(split.right_node, "4R"))
# split.right_node.right_node.add_left_child(Searchtree(split.right_node.right_node, "5L"))
# split.right_node.right_node.left_node.add_right_child(Searchtree(split.right_node.right_node.left_node, "6R"))
# split.right_node.right_node.left_node.right_node.add_right_child(Searchtree(split.right_node.right_node.left_node.right_node, "7R"))

# split.add_left_child(Searchtree(split, "3L"))
# split.left_node.add_left_child(Searchtree(split.left_node, "4L"))
# split.left_node.left_node.add_left_child(Searchtree(split.left_node.left_node, "5L"))


# split2 = split.left_node.left_node.left_node

# split2.add_left_child(Searchtree(split2, "6L"))
# split2.left_node.add_left_child(Searchtree(split2.left_node, "7L"))
# split2.left_node.left_node.add_left_child(Searchtree(split2.left_node.left_node, "8L"))

# split2.add_right_child(Searchtree(split2, "6R"))
# split2.right_node.add_left_child(Searchtree(split2.right_node, "7L"))
# split2.right_node.left_node.add_left_child(Searchtree(split2.right_node.left_node, "8L"))

# new_new_tree = condense_tree(root)
# print_tree(new_new_tree)
# contracting_search_tree(new_new_tree)




# data = extract_data("/Users/MortenHelsoe/Desktop/DTU/6. Semester/Bachelor Projekt/Tangle-lib-ORM/DTU-Tangle/csv_test/test.csv")
# res = DataSetBinaryQuestionnaire(1)


# # order_cuts = res_cuts.order_function()



# root_binary = res.cut_generator_binary(data)

# tree = create_searchtree(root_binary)
# tree_condense = condense_tree(tree)
# contracting_search_tree(tree_condense)

# # tree_contract = contracting_search_tree(tree_condense)


# soft = soft_clustering(tree_condense, 0, 1, {})
# hard = hard_clustering(soft)
# print(hard)



# print_tree(root)
# new_new_tree = condense_tree(root)
# print_tree(new_new_tree)
# contracting_search_tree(new_new_tree)

# ben = [new_new_tree]

# for n in ben: 
#     print("___")
#     print(str(n.cut_id)+n.cut_orientation)
#     print(n.condensed_oritentations)
#     print({cut.id for cut in n.characterizing_cuts})
#     if n.left_node != None:
#         ben.append(n.left_node)
#     if n.right_node != None: 
#         ben.append(n.right_node)
#     print("___")

# res = cut_generator_binary("/Users/MortenHelsoe/Desktop/DTU/6. Semester/Bachelor Projekt/Tangle-lib-ORM/DTU-Tangle/csv_test/test.csv")

# john = create_searchtree(res, 3)

# show_tree(john)






