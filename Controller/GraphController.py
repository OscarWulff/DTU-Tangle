from sklearn.metrics import normalized_mutual_info_score
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

            # Calculate NMI score directly using ground truth and predicted tangles
            ground_truth = self.view.generated_ground_truth
            nmi_score = self.nmi_score(ground_truth, hard)  # Assuming hard contains the predicted tangles

            self.view.nmi_score_tangles = round(nmi_score, 2)
            self.view.setup_plots()

        except Exception as e:
            print("Error:", e)

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
            sc = SpectralClustering(k, affinity='precomputed')  # Specify affinity as precomputed
            sc.fit(adj_mat)

            # Plot the spectral clustering result
            self.view.spectral_plot = sc.labels_

            # Calculate NMI score only if ground truth is available
            if self.view.generated_ground_truth:
                ground_truth = self.view.generated_ground_truth
                nmi_score = self.nmi_score(ground_truth, sc.labels_)
                self.view.nmi_score_spectral = round(nmi_score, 2)
            
            self.view.setup_plots()
            end_time = time.time()

            total_time = end_time - start_time
            print("Total time for spectral clustering:", total_time)

        except Exception as e:
            print("Error in spectral clustering:", e)


    def nmi_score(self, ground_truth, predicted_tangles):
        """
        Calculates the NMI score of the predicted tangles
        """
        nmi_score = normalized_mutual_info_score(ground_truth, predicted_tangles)
        return nmi_score


    def upload_data(self):
        try:
            # Open a file dialog to select the file
            file_dialog = QFileDialog()
            if file_dialog.exec_():
                selected_files = file_dialog.selectedFiles()
                if selected_files:
                    selected_file = selected_files[0]
                    print("Selected file:", selected_file)
                    G = nx.read_gml(selected_file)  # Assuming the data is stored in a .gml file
                    
                    if G is not None:
                        # Extract ground truth labels based on conference affiliations
                        ground_truth = []
                        for node_id, data in G.nodes(data=True):
                            if 'value' in data:
                                # Convert 'value' to an integer if it's not already
                                value = int(data['value']) if not isinstance(data['value'], int) else data['value']
                                ground_truth.append(value)
                            elif 'gt' in data:
                                # Ground truth is a string, append it directly
                                ground_truth.append(data['gt'])
                            else:
                                # If neither 'value' nor 'gt' attribute is found, assign 0 as the default label
                                ground_truth.append(0)

                        # Convert node labels to integers
                        G = nx.convert_node_labels_to_integers(G)
                        
                        # Clear existing generated graph and ground truth
                        self.view.generated_graph = G
                        self.view.generated_ground_truth = ground_truth
                        
                        # Set the generated graph in the view
                        self.view.upload_data_show()
                        self.view.setup_plots()
                    else:
                        print("Error: Graph is None")
        except Exception as e:
            print("Error:", e)






        