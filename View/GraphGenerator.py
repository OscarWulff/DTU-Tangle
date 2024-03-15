import matplotlib.pyplot as plt
import networkx as nx
from PyQt5.QtWidgets import QMainWindow, QPushButton, QVBoxLayout, QWidget, QHBoxLayout, QFileDialog, QComboBox, QLabel, QLineEdit, QApplication
from PyQt5 import QtCore
from Model.GenerateTestData import GenerateRandomGraph
from Model.DataSetGraph import DataSetGraph
from Model.TangleCluster import *

class GraphGeneratorWindow(QMainWindow):
    def __init__(self, main_page):
        super().__init__()
        self.main_page = main_page
        self.setWindowTitle("Graph Generation Window")
        self.setGeometry(100, 100, 800, 600)

        # Center the window on the screen
        screen_geometry = QApplication.desktop().availableGeometry()
        window_geometry = self.frameGeometry()
        x = (screen_geometry.width() - window_geometry.width()) // 2
        y = (screen_geometry.height() - window_geometry.height()) // 2
        self.move(x, y)

        

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        # Add back button
        self.back_button = QPushButton("Back", self)
        self.back_button.clicked.connect(self.go_back_to_main_page)
        layout.addWidget(self.back_button, alignment=QtCore.Qt.AlignLeft)

        # Add description and text box for the number of nodes
        node_description_label = QLabel("Amount of Nodes", self)
        node_description_label.setAlignment(QtCore.Qt.AlignCenter)
        node_description_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(node_description_label, alignment=QtCore.Qt.AlignCenter)
        self.node_textbox = QLineEdit(self)
        self.node_textbox.setMaximumWidth(200)  # Set maximum width for the text box
        layout.addWidget(self.node_textbox, alignment=QtCore.Qt.AlignCenter)
        
        # Add description and text box for the number of clusters
        cluster_description_label = QLabel("Amount of Clusters", self)
        cluster_description_label.setAlignment(QtCore.Qt.AlignCenter)
        cluster_description_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(cluster_description_label, alignment=QtCore.Qt.AlignCenter)
        self.cluster_textbox = QLineEdit(self)
        self.cluster_textbox.setMaximumWidth(200)  # Set maximum width for the text box
        layout.addWidget(self.cluster_textbox, alignment=QtCore.Qt.AlignCenter)

        # Add header for dropdowns
        header_label = QLabel("Choose Ground Truth and Algorithm", self)
        header_label.setAlignment(QtCore.Qt.AlignCenter)
        header_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(header_label, alignment=QtCore.Qt.AlignCenter)  # Center the header

        # Add buttons and dropdowns
        dropdown_layout = QVBoxLayout()
        dropdown_layout.setAlignment(QtCore.Qt.AlignCenter)  # Center the dropdowns

        # Add dropdown for ground truth
        self.ground_truth_combo = QComboBox(self)
        self.ground_truth_combo.addItem("Ground Truth: Off")
        self.ground_truth_combo.addItem("Ground Truth: On")
        layout.addWidget(self.ground_truth_combo, alignment=QtCore.Qt.AlignCenter)

        # Add dropdown for clustering algorithm
        self.algorithm_combo = QComboBox(self)
        self.algorithm_combo.addItem("Tangle")
        self.algorithm_combo.addItem("K-Means")
        self.algorithm_combo.addItem("Spectral")
        layout.addWidget(self.algorithm_combo, alignment=QtCore.Qt.AlignCenter)

        # Connect the signal to update the algorithm combo box visibility
        self.ground_truth_combo.currentTextChanged.connect(self.update_algorithm_combo_visibility)


        layout.addLayout(dropdown_layout)

        # Add generate random data button
        self.generate_data_button = QPushButton("Generate Random Data", self)
        self.generate_data_button.clicked.connect(self.generate_random_data)
        layout.addWidget(self.generate_data_button, alignment=QtCore.Qt.AlignCenter)


    def go_back_to_main_page(self):
        self.close()  # Close the current window
        # Show the main page again
        self.main_page.show()

    def update_algorithm_combo_visibility(self, text):
        if text == "Ground Truth: Off":
            self.algorithm_combo.show()
        else:
            self.algorithm_combo.hide()

    def generate_random_data(self):
        # Get the values from the text boxes
        num_nodes = int(self.node_textbox.text())
        num_clusters = int(self.cluster_textbox.text())

        # Generate a random graph
        graph_generator = GenerateRandomGraph(num_of_nodes=num_nodes, num_of_clusters=num_clusters)
        random_graph, ground_truth = graph_generator.generate_random_graph()

        # Create an instance of DataSetGraph
        data_set_graph = DataSetGraph(agreement_param=1)

        # Assign the generated graph to the G attribute
        data_set_graph.G = random_graph

        # Initialize the DataSetGraph instance
        data_set_graph.initialize()  # Initialize the graph and generate cuts

        root = create_searchtree(data_set_graph)
        root_condense = condense_tree(root)
        contracting_search_tree(root_condense)

        # Soft clustering
        soft = soft_clustering(root_condense, 0, 1, {})
        hard = hard_clustering(soft)

        # Visualize the graph with ground truth clusters
        if (self.ground_truth_combo.currentText() == "Ground Truth: Off"):
            # Get the colors for hard clustering
            node_colors = [hard[node] for node in random_graph.nodes()]

            # Plot the graph with hard clustering colors
            plt.figure(figsize=(8, 6))  # Set figure size
            pos = nx.spring_layout(random_graph)
            nx.draw(random_graph, pos, with_labels=True, node_color=node_colors, node_size=500, edge_color='black', linewidths=1, font_size=10)
            plt.title('Graph with Hard Clustering Colors')
            plt.show()

        else:
            graph_generator.visualize_graph(random_graph, ground_truth=ground_truth, visualize_ground_truth=True)

        plt.title('Graph with Cuts')
        plt.show()

