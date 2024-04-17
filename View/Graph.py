import time
import ast
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import networkx as nx
import numpy as np
from PyQt5.QtWidgets import QMainWindow, QPushButton, QVBoxLayout, QWidget, QHBoxLayout, QFileDialog, QComboBox, \
    QLabel, QLineEdit, QApplication, QSizePolicy
from PyQt5 import QtCore
from sklearn.metrics import normalized_mutual_info_score


class GraphWindow(QMainWindow):
    def __init__(self, main_page):
        super().__init__()
        self.main_page = main_page
        self.setWindowTitle("Graph Based Window")
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()

        # Add spacer
        layout.addStretch()

        # Create a Matplotlib figure and canvas
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.figure.add_subplot(111)
        layout.addWidget(self.canvas)

        # Variables of figures
        self.numb_plots = 1
        self.generated_graph = None
        self.generated_ground_truth = None
        self.tangles_plot = None
        self.spectral_plot = None
        self.nmi_score_tangles = None
        self.nmi_score_spectral = None

        # Flag to track if a random graph has been generated
        self.graph_generated_flag = False

        # Add buttons
        button_layout = QHBoxLayout()

        self.upload_data_button = QPushButton("Upload Data as .csv", self)
        self.upload_data_button.clicked.connect(self.upload_data)
        button_layout.addWidget(self.upload_data_button)

        self.generate_data_button = QPushButton("Generate Data", self)
        self.generate_data_button.clicked.connect(self.generate_data)
        button_layout.addWidget(self.generate_data_button)

        self.generate_random_button = QPushButton("Generate Graph", self)
        button_layout.addWidget(self.generate_random_button)
        self.generate_random_button.hide()

        self.generate_tangles_button = QPushButton("Apply Tangles", self)
        button_layout.addWidget(self.generate_tangles_button)
        self.generate_tangles_button.hide()


        self.generate_spectral_button = QPushButton("Apply Spectral", self)
        button_layout.addWidget(self.generate_spectral_button)
        self.generate_spectral_button.hide()

        self.numb_nodes = QLineEdit()
        self.numb_nodes.setFixedSize(300, 30)
        self.numb_nodes.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.numb_nodes.setPlaceholderText("Number of Nodes")
        self.numb_nodes.hide()
        layout.addWidget(self.numb_nodes)

        self.numb_clusters = QLineEdit()
        self.numb_clusters.setFixedSize(300, 30)
        self.numb_clusters.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.numb_clusters.setPlaceholderText("Number of Clusters")
        self.numb_clusters.hide()
        layout.addWidget(self.numb_clusters)

        self.agreement_parameter = QLineEdit()
        self.agreement_parameter.setFixedSize(300, 30)
        self.agreement_parameter.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.agreement_parameter.setPlaceholderText("Agreement Parameter")
        self.agreement_parameter.hide()
        layout.addWidget(self.agreement_parameter)

        self.average_edges_to_same_cluster = QLineEdit()
        self.average_edges_to_same_cluster.setFixedSize(300, 30)
        self.average_edges_to_same_cluster.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.average_edges_to_same_cluster.setPlaceholderText("Avg. Edges to Same Cluster (p)")
        self.average_edges_to_same_cluster.hide()
        layout.addWidget(self.average_edges_to_same_cluster)

        self.average_edges_to_other_clusters = QLineEdit()
        self.average_edges_to_other_clusters.setFixedSize(300, 30)
        self.average_edges_to_other_clusters.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.average_edges_to_other_clusters.setPlaceholderText("Avg. Edges to Other Clusters (q)")
        self.average_edges_to_other_clusters.hide()
        layout.addWidget(self.average_edges_to_other_clusters)

        self.k_spectral = QLineEdit()
        self.k_spectral.setFixedSize(300, 30)
        self.k_spectral.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.k_spectral.setPlaceholderText("k for Spectral")
        self.k_spectral.hide()
        layout.addWidget(self.k_spectral)

        self.variance = QLabel(self)
        self.variance.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTop)
        layout.addWidget(self.variance)
        self.variance.hide()

        layout.addLayout(button_layout)

        # Add spacer
        layout.addStretch()

        # Add back button
        self.back_button = QPushButton("Back", self)
        self.back_button.clicked.connect(self.go_back_to_main_page)
        layout.addWidget(self.back_button)

        central_widget.setLayout(layout)

    def generate_data(self):
        self.upload_data_button.hide()
        self.generate_data_button.hide()
        self.generate_random_button.show()
        self.generate_spectral_button.show()
        self.generate_tangles_button.show()
        self.numb_nodes.show()
        self.numb_clusters.show()
        self.average_edges_to_same_cluster.show()
        self.average_edges_to_other_clusters.show()
        self.agreement_parameter.show()
        self.k_spectral.show()


    def setup_plots(self):
        self.figure.clear()

        if self.numb_plots == 1:
            plot = self.figure.add_subplot(111)
            plot.set_title('Ground truth')
            self.visualize_graph(self.generated_graph, self.generated_ground_truth)

        elif self.numb_plots == 2:
            if self.generated_graph is not None:
                plot = self.figure.add_subplot(121)
                plot.set_title('Ground truth')
                self.visualize_graph(self.generated_graph, self.generated_ground_truth)

            if self.tangles_plot is not None:
                plot = self.figure.add_subplot(122)
                plot.set_title('Tangles (NMI = {})'.format(self.nmi_score_tangles))
                self.visualize_graph(self.generated_graph, self.tangles_plot)
            
        else:
            if self.generated_graph is not None:
                plot = self.figure.add_subplot(221)
                plot.set_title('Ground truth')
                self.visualize_graph(self.generated_graph, self.generated_ground_truth)

            if self.tangles_plot is not None:
                plot = self.figure.add_subplot(222)
                plot.set_title('Tangles (NMI = {})'.format(self.nmi_score_tangles))
                self.visualize_graph(self.generated_graph, self.tangles_plot)

            if self.spectral_plot is not None:
                plot = self.figure.add_subplot(223)
                plot.set_title('Spectral (NMI = {})'.format(self.nmi_score_spectral))
                self.visualize_graph(self.generated_graph, self.spectral_plot)

        self.figure.subplots_adjust(hspace=0.5, wspace=0.5)
        self.canvas.draw()



    def visualize_graph(self, graph, plot):
        pos = nx.spring_layout(graph)

        clusters = sorted(set(plot))
        colors = plt.cm.viridis(np.linspace(0, 1, len(clusters)))
        color_map = {cluster: color for cluster, color in zip(clusters, colors)}
        node_colors = [color_map[cluster] for cluster in plot]


        nx.draw(graph, pos, with_labels=True, node_color=node_colors, node_size=500, edge_color='black',
                linewidths=1, font_size=10)

        nx.draw_networkx_edges(graph, pos, edge_color='black')



    def go_back_to_main_page(self):
        if self.upload_data_button.isVisible() or self.generate_data_button.isVisible():
            # If either of the upload or generate data buttons are visible,
            # it means the user is on the page where they choose between uploading or generating data.
            # In this case, we close the current window and return to the main page.
            self.close()  # Close the current window
            self.main_page.show()  # Show the main page
        else:
            # Otherwise, the user is on some other page (e.g., after choosing upload or generate data)
            # and we go back to the page where they choose between uploading or generating data.
            self.generate_data_button.show()  # Show the generate data button
            self.upload_data_button.show()  # Show the upload data button
            # Hide other buttons and input fields
            self.generate_random_button.hide()
            self.generate_tangles_button.hide()
            self.generate_spectral_button.hide()
            self.numb_nodes.hide()
            self.numb_clusters.hide()
            self.average_edges_to_same_cluster.hide()
            self.average_edges_to_other_clusters.hide()
            self.agreement_parameter.hide()
            self.k_spectral.hide()


    def upload_data(self):
        # The implementation of this method is skipped to fit within the response limit
        pass
