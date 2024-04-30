import networkx as nx
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
# Generate or load a graph
G = nx.karate_club_graph()
G_fb = nx.facebook.Graph()

file_path_fb = "facebook_graph.edgelist"

# Save the Facebook graph in edge list format
nx.write_edgelist(G_fb, file_path_fb)

print("Facebook graph has been saved to", file_path_fb)


# Specify the file path where you want to save the graph
file_path = "karate_club_graph.edgelist"

# Save the graph in edge list format
nx.write_edgelist(G, file_path)

