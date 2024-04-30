from Model.GenerateTestData import GenerateRandomGraph
from Model.DataSetGraph import DataSetGraph
from sklearn.cluster import SpectralClustering
from PyQt5.QtWidgets import QFileDialog, QCheckBox, QComboBox, QLineEdit, QPushButton, QMainWindow
from Model.TangleCluster import *
import time


class GraphController:
    def __init__(self, view):
        self.view = view
        self.view.upload_data_button.clicked.connect(self.upload_data)
        self.view.generate_random_button.clicked.connect(self.generate_random)
        self.view.generate_tangles_button.clicked.connect(self.tangles)
        self.view.generate_spectral_button.clicked.connect(self.spectral)
        self.random_graph_generator = None  # Initialize random_graph_generator attribute


    
    def generate_random(self):
        try:
            # Get the values from the input fields
            num_of_nodes = int(self.view.numb_nodes.text())
            num_of_clusters = int(self.view.numb_clusters.text())
            avg_edges_to_same_cluster = float(self.view.average_edges_to_same_cluster.text())
            avg_edges_to_other_clusters = float(self.view.average_edges_to_other_clusters.text())

            # Create an instance of GenerateRandomGraph
            self.random_graph_generator = GenerateRandomGraph(num_of_nodes, num_of_clusters, avg_edges_to_same_cluster,
                                                         avg_edges_to_other_clusters)

            # Generate a random graph using the ground truth
            self.view.generated_graph, self.view.generated_ground_truth = self.random_graph_generator.generate_random_graph()
            print("ground truth", self.view.generated_ground_truth)

            self.view.setup_plots()

        except ValueError:
            print("Invalid input")

    
    def tangles(self):
        try:
            # Check if the generated graph exists
            if self.view.generated_graph is None:
                print("No generated graph available.")
                return

            if self.view.tangles_plot is None: 
                self.view.numb_plots += 1 
            # Perform tangles on the generated graph
            agreement_parameter = int(self.view.agreement_parameter.text())
            start_time = time.time()
            data = DataSetGraph(agreement_param=agreement_parameter)
            data.G = self.view.generated_graph
            data.generate_multiple_cuts(data.G) 
            data.cost_function_Graph()

            
            root = create_searchtree(data)
            root_condense = condense_tree(root)
            contracting_search_tree(root_condense)
            soft = soft_clustering(root_condense)
            hard = hard_clustering(soft)


            self.view.tangles_plot = hard
            end_time = time.time()

            total_time = end_time - start_time
            print("Total time for tangles:", total_time)
            # Visualize tangles
            # if data is uploaded nmi score should not be calculated
            if self.random_graph_generator is not None:
                self.view.nmi_score_tangles = round(self.random_graph_generator.nmi_score(hard), 2)

            self.view.setup_plots()

        except ValueError as e:
            print("Invalid input", e)

    def spectral(self):
        try:
            # Check if the generated graph exists
            if self.view.generated_graph is None:
                print("No generated graph available.")
                return

            start_time = time.time()
            G = self.view.generated_graph
            # Get adjacency matrix as numpy array
            adj_mat = nx.convert_matrix.to_numpy_array(G)

            # Get the number of clusters from the input field
            k = int(self.view.k_spectral.text())  # Assuming you have a QLineEdit for input
            if self.view.spectral_plot is None: 
                self.view.numb_plots += 1

            # Cluster
            sc = SpectralClustering(k)
            sc.fit(adj_mat)

            # Plot the spectral clustering result
            self.view.spectral_plot = sc.labels_

            if self.random_graph_generator is not None:
                self.view.nmi_score_spectral = round(self.random_graph_generator.nmi_score(sc.labels_), 2)
            
            self.view.setup_plots()
            end_time = time.time()

            total_time = end_time - start_time
            print("Total time for spectral clustering:", total_time)

        except Exception as e:
            print("Error in spectral clustering:", e)


    def upload_data(self):
        try:
            # Open a file dialog to select the file
            file_dialog = QFileDialog()
            if file_dialog.exec_():
                selected_files = file_dialog.selectedFiles()
                if selected_files:
                    selected_file = selected_files[0]
                    print("Selected file:", selected_file)
                    G = nx.read_edgelist(selected_file)  # Assuming the data is stored in an edge list format

                    if G is not None:
                        # Convert node labels to integers
                        G = nx.convert_node_labels_to_integers(G)

                        # Clear existing generated graph and ground truth
                        self.view.generated_graph = G
                        self.view.generated_ground_truth = [1] * len(G.nodes)
                        # Set the generated graph in the view

                        # Show the upload data view
                        self.view.upload_data_show()
                        self.view.setup_plots()
                    else:
                        print("Error: Graph is None")

        except Exception as e:
            # Handle errors more gracefully
            print("Error uploading data:", e)
            # Optionally, show an error dialog to the user or log the error for debugging




        