
from SearchTree import *
#from DataSetBinaryQuestionnaire import *

from DataSetFeatureBased import order_function_featurebased

from DataType import DataType

def is_consistent(chosen_cut, tangles, agrrement_parameter):
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
    root = Searchtree(None, 0)

    leaves = [root]
    cuts_ordered = data.order_function_featurebased()
    #cuts_ordered = order_cuts_by_cost(cuts)
    # print(cuts_ordered)
    id = 0
    for  [A, Ac] in cuts_ordered:
        new_leaves = []
        for leaf in leaves:
            # print(leaf.tangle)
            if is_consistent(A, leaf.tangle, data.a):
                left_child = Searchtree(leaf, leaf.cut_id+1)
                left_child.cut_orientation = "L"
                left_child.tangle.append([A])
                left_child.tangle += leaf.tangle
                id = id+1
                left_child.id = id
                leaf.add_left_child(left_child)
                new_leaves.append(left_child)
            if is_consistent(Ac, leaf.tangle, data.a):
                right_child = Searchtree(leaf, leaf.cut_id+1)
                right_child.cut_orientation = "R"
                right_child.tangle.append([Ac])
                right_child.tangle += leaf.tangle
                id = id+1
                right_child.id = id
                leaf.add_right_child(right_child)
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
root = Searchtree(None, "root")
root.add_left_child(Searchtree(root, "1L"))
root.add_right_child(Searchtree(root, "1R"))
root.right_node.add_right_child(Searchtree(root.right_node, "2R"))


split = root.right_node.right_node

split.add_right_child(Searchtree(split, "3R"))
split.right_node.add_right_child(Searchtree(split.right_node, "4R"))
split.right_node.right_node.add_left_child(Searchtree(split.right_node.right_node, "5L"))
split.right_node.right_node.left_node.add_right_child(Searchtree(split.right_node.right_node.left_node, "6R"))
split.right_node.right_node.left_node.right_node.add_right_child(Searchtree(split.right_node.right_node.left_node.right_node, "7R"))

split.add_left_child(Searchtree(split, "3L"))
split.left_node.add_left_child(Searchtree(split.left_node, "4L"))
split.left_node.left_node.add_left_child(Searchtree(split.left_node.left_node, "5L"))


split2 = split.left_node.left_node.left_node

split2.add_left_child(Searchtree(split2, "6L"))
split2.left_node.add_left_child(Searchtree(split2.left_node, "7L"))
split2.left_node.left_node.add_left_child(Searchtree(split2.left_node.left_node, "8L"))

split2.add_right_child(Searchtree(split2, "6R"))
split2.right_node.add_left_child(Searchtree(split2.right_node, "7L"))
split2.right_node.left_node.add_left_child(Searchtree(split2.right_node.left_node, "8L"))

print_tree(root)
new_new_tree = condense_tree(root)
print_tree(new_new_tree)
contracting_search_tree(new_new_tree)

ben = [new_new_tree]

for n in ben: 
    print(n.cut_id+n.cut_orientation)
    print(n.condensed_oritentations)
    print(n.characterizing_cuts)
    if n.left_node != None:
        ben.append(n.left_node)
    if n.right_node != None: 
        ben.append(n.right_node)

# res = cut_generator_binary("/Users/MortenHelsoe/Desktop/DTU/6. Semester/Bachelor Projekt/Tangle-lib-ORM/DTU-Tangle/csv_test/test.csv")

# john = create_searchtree(res, 3)

# show_tree(john)






