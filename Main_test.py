import random
from Model.DataSet import extract_data
from Model.DataSetBinaryQuestionnaire import DataSetBinaryQuestionnaire, perform_tsne
from Model.SearchTree import generate_color_dict
from Model.GenerateTestData import *


from Model.SearchTree import condense_tree, contracting_search_tree, print_tree, soft_clustering, hard_clustering
from Model.TangleCluster import create_searchtree
import matplotlib.pyplot as plt


data1 = extract_data("/Users/MortenHelsoe/Desktop/DTU/6. Semester/Bachelor Projekt/Tangle-lib-ORM/DTU-Tangle/csv_test/test2.csv")

dbq = DataSetBinaryQuestionnaire(3)
data_gen = GenerateDataBinaryQuestionnaire()

data,_ = data_gen.generate_biased_binary_questionnaire_answers(200, 1000, 4)
data_array = np.array([bitset.bitset for bitset in data])

points = perform_tsne(data_array, perplexity=30)

plt.figure(figsize=(10, 8))  # Set the figure size for better readability
plt.scatter(points['Dim1'], points['Dim2'], alpha=0.6)  # Plot Dim1 vs Dim2 with some transparency

plt.title('t-SNE Visualization of Questionnaire Data')  # Title for the plot
plt.xlabel('Dim1')  # Label for the x-axis
plt.ylabel('Dim2')  # Label for the y-axis
plt.grid(True)  # Show a grid for easier visualization
plt.show()  # Display the plot



# print(data)


# data_cuts = dbq.cut_generator_binary(data)


# for cut in data_cuts.cuts:
#     print(cut.A, " ", cut.Ac, " ", cut.cost)




# tree = create_searchtree(data_cuts)

# # print_tree(tree)
# new_new_tree = condense_tree(tree)
# # print_tree(new_new_tree)
# contracting_search_tree(new_new_tree)


# soft = soft_clustering(tree)
# hard = hard_clustering(soft)

# color_dict = generate_color_dict(hard)

# print(color_dict)





















# def generate_random_color():
#     """
#     Generate a random color represented as a tuple of RGB values.
#     """
#     r = random.randint(0, 255)  # Red component
#     g = random.randint(0, 255)  # Green component
#     b = random.randint(0, 255)  # Blue component
#     return (r, g, b)


# def generate_color_dict(data, tree):
#     vals = []

#     for i in range(data.shape[0]):
#         soft = soft_clustering(tree, i, 1, {})
#         vals.append(hard_clustering(soft)[0])

    
#     set_vals = set(vals)
#     color_dict = {}

#     for i in set_vals:
#         color_dict[i] = generate_random_color()


#     return color_dict, vals


