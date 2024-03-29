from Model.SearchTree import *
from Model.TangleCluster import *
from Model.GenerateTestData import *

def main():

    gdfb = GenerateDataFeatureBased(5, 0.4)

    gdfb.random_clusters(10)

    gdfb.plot_points()


    data = DataSetFeatureBased(10)
    data.points = gdfb.points

    data.cut_generator_axis()
    data.cost_function()

    root = create_searchtree(data)
    print_tree(root)
    new_root = condense_tree(root)
    contracting_search_tree(new_root)
    print_tree(new_root)

    soft = soft_clustering(new_root)
    print(soft)

    hard = hard_clustering(soft)
    print(hard)

    print("nmi-score = ", gdfb.nmi_score(hard))

    truth1 = gdfb.k_means(5)
    print(truth1)
    print("nmi-score = ", gdfb.nmi_score(truth1))
    truth2 = gdfb.spectral_clustering(5)
    print(truth2)
    print("nmi-score = ", gdfb.nmi_score(truth2))

    gdfb.ground_truth = hard
    gdfb.plot_points()            
                


main()
