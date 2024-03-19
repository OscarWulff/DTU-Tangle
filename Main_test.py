import random
from Model.DataSet import extract_data
from Model.DataSetBinaryQuestionnaire import DataSetBinaryQuestionnaire

from Model.SearchTree import condense_tree, contracting_search_tree, print_tree, soft_clustering, hard_clustering
from Model.TangleCluster import create_searchtree


data1 = extract_data("/Users/MortenHelsoe/Desktop/DTU/6. Semester/Bachelor Projekt/Tangle-lib-ORM/DTU-Tangle/csv_test/test.csv")

dbq = DataSetBinaryQuestionnaire(3)

data_cuts = dbq.cut_generator_binary(data1)


for cut in data_cuts.cuts:
    print(cut.A, " ", cut.Ac, " ", cut.cost)




tree = create_searchtree(data_cuts)

# print_tree(tree)
new_new_tree = condense_tree(tree)
# print_tree(new_new_tree)
contracting_search_tree(new_new_tree)



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

    for i in range(data.shape[0]):
        soft = soft_clustering(tree, i, 1, {})
        vals.append(hard_clustering(soft)[0])

    
    set_vals = set(vals)
    color_dict = {}

    for i in set_vals:
        color_dict[i] = generate_random_color()


    return color_dict, vals


