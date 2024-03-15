from Model.SearchTree import *
from Model.TangleCluster import *
from Model.GenerateTestData import *

def main():

    gdfb = GenerateDataFeatureBased(10, 1)

    gdfb.random_clusters(30)

    gdfb.plot_points()
    # root = create_searchtree(DataSetFeatureBased(2))
    # print_tree(root)
    # new_new_tree = condense_tree(root)
    # print_tree(new_new_tree)
    # contracting_search_tree(new_new_tree)
    # soft = soft_clustering(root, 3, 1)
    # print(soft)
    # hard = hard_clustering(soft)
    # print(hard)

main()
