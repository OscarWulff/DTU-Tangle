from Model.SearchTree import *
from Model.TangleCluster import *
from Model.GenerateTestData import *

def main():

    gdfb = GenerateDataFeatureBased(3, 1)

    gdfb.random_clusters(10)

    gdfb.plot_points()


    data = DataSetFeatureBased(10)
    data.points = gdfb.points

    for point in data.points:
        print(point[0],point[1]) 

    data.cut_generator_axis(0)
    data.cost_function()

    root = create_searchtree(data)
    print_tree(root)
    new_root = condense_tree(root)
    contracting_search_tree(new_root)

    soft = soft_clustering(new_root)
    print(soft)

    hard = hard_clustering(soft)
    print(hard)

main()
