import time
import ast
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import networkx as nx
import numpy as np
from PyQt5.QtWidgets import QMainWindow, QPushButton, QVBoxLayout, QWidget, QHBoxLayout, QFileDialog, QLabel, QLineEdit, QApplication, QSizePolicy, QComboBox, QMessageBox, QCheckBox
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
        self.louvain_plot = None
        self.nmi_score_tangles = None
        self.nmi_score_spectral = None
        self.nmi_score_louvain = None
        self.tangles_time = None
        self.spectral_time = None
        self.louvain_time = None
        self.prob = None

        # Flag to track if a random graph has been generated
        self.graph_generated_flag = False
        self.prob_checked = False

        # Add buttons
        button_layout = QHBoxLayout()

        self.upload_data_button = QPushButton("Upload Data as .gml file", self)
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

        self.soft_clustering = QCheckBox("soft clustering")
        self.soft_clustering.stateChanged.connect(self.soft_clustering_changed)
        self.soft_clustering.hide()
        layout.addWidget(self.soft_clustering)

        self.generate_spectral_button = QPushButton("Apply Spectral", self)
        button_layout.addWidget(self.generate_spectral_button)
        self.generate_spectral_button.hide()

        self.generate_louvain_button = QPushButton("Apply Louvain", self)  # Add Louvain button
        button_layout.addWidget(self.generate_louvain_button)
        self.generate_louvain_button.hide()

        # Add label to prompt user to choose initial partitioning method
        self.partition_label = QLabel("Choose Initial Partitioning Method:", self)
        layout.addWidget(self.partition_label)

        # Create a combo box for choosing initial partitioning method
        self.partition_method_combobox = QComboBox()
        self.partition_method_combobox.addItems(["K-Means", "Kernighan-Lin", "K-Means-Half", "K-Means-Both"])
        self.partition_label.hide()
        self.partition_method_combobox.hide()
        layout.addWidget(self.partition_method_combobox)


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
        self.generate_louvain_button.show()  # Show Louvain button
        self.partition_label.show()
        self.partition_method_combobox.show()
        self.numb_nodes.show()
        self.numb_clusters.show()
        self.average_edges_to_same_cluster.show()
        self.average_edges_to_other_clusters.show()
        self.agreement_parameter.show()
        self.k_spectral.show()
        self.soft_clustering.show()

    def update_partition_method(self):
        self.selected_partition_method = self.partition_method_combobox.currentText()

    def show_input_fields(self):
        self.numb_nodes.show()
        self.numb_clusters.show()
        self.average_edges_to_same_cluster.show()
        self.average_edges_to_other_clusters.show()
        self.agreement_parameter.show()
        self.k_spectral.show()
        self.soft_clustering.show()

    def soft_clustering_changed(self, state):
        if state == 2:  # Checked
            if self.tangles_plot is None:
                QMessageBox.warning(self, "Error", "You must apply Tangles before using Soft Clustering.")
                self.soft_clustering.setChecked(False)
                self.prob_checked = False
                
            else:
                self.prob_checked = True
                self.setup_plots()
        else:  # Unchecked
            self.prob_checked = False
            self.setup_plots()


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
                plot.set_title('Tangles (NMI = {:.3f}), Time = {:.3f} sec'.format(self.nmi_score_tangles, self.tangles_time))
                if self.prob_checked:
                    self.visualize_graph_prob(self.generated_graph, self.tangles_plot)
                else:
                    self.visualize_graph(self.generated_graph, self.tangles_plot)

        elif self.numb_plots == 3:
            if self.generated_graph is not None:
                plot = self.figure.add_subplot(221)
                plot.set_title('Ground truth')
                self.visualize_graph(self.generated_graph, self.generated_ground_truth)

            if self.tangles_plot is not None:
                plot = self.figure.add_subplot(222)
                plot.set_title('Tangles (NMI = {:.3f}), Time = {:.3f} sec'.format(self.nmi_score_tangles, self.tangles_time))
                if self.prob_checked:
                    self.visualize_graph_prob(self.generated_graph, self.tangles_plot)
                else:
                    self.visualize_graph(self.generated_graph, self.tangles_plot)

            if self.spectral_plot is not None:
                plot = self.figure.add_subplot(223)
                plot.set_title('Spectral (NMI = {:.3f}), Time = {:.3f} sec'.format(self.nmi_score_spectral, self.spectral_time))
                self.visualize_graph(self.generated_graph, self.spectral_plot)

        elif self.numb_plots == 4:
            if self.generated_graph is not None:
                plot = self.figure.add_subplot(221)
                plot.set_title('Ground truth')
                self.visualize_graph(self.generated_graph, self.generated_ground_truth)

            if self.tangles_plot is not None:
                plot = self.figure.add_subplot(222)
                plot.set_title('Tangles (NMI = {:.3f}), Time = {:.3f} sec'.format(self.nmi_score_tangles, self.tangles_time))
                if self.prob_checked:
                    self.visualize_graph_prob(self.generated_graph, self.tangles_plot) 
                else:
                    self.visualize_graph(self.generated_graph, self.tangles_plot)

            if self.spectral_plot is not None:
                plot = self.figure.add_subplot(223)
                plot.set_title('Spectral (NMI = {:.3f}), Time = {:.3f} sec'.format(self.nmi_score_spectral, self.spectral_time))
                self.visualize_graph(self.generated_graph, self.spectral_plot)

            if self.louvain_plot is not None:
                plot = self.figure.add_subplot(224)
                plot.set_title('Louvain (NMI = {:.3f}), Time = {:.3f} sec'.format(self.nmi_score_louvain, self.louvain_time))
                self.visualize_graph(self.generated_graph, self.louvain_plot)

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

    def visualize_graph_prob(self, graph, plot):
        pos = nx.spring_layout(graph)

        clusters = sorted(set(plot))
        colors = plt.cm.viridis(np.linspace(0, 1, len(clusters)))
        color_map = {cluster: color for cluster, color in zip(clusters, colors)}

        # Adjusting the node colors based on probabilities
        alpha_values = [self.prob[i] for i in range(len(plot))] if self.prob is not None else [1] * len(plot)

        for node, (cluster, alpha) in enumerate(zip(plot, alpha_values)):
            nx.draw_networkx_nodes(graph, pos, nodelist=[node], node_color=[color_map[cluster]], alpha=alpha, node_size=500)

        nx.draw_networkx_labels(graph, pos, font_size=10)
        nx.draw_networkx_edges(graph, pos, edge_color='black', width=1)



    def go_back_to_main_page(self):
        self.close()  # Close the current window
        # Show the main page again
        self.main_page.show()

    def upload_data_show(self):
        self.upload_data_button.hide()
        self.generate_data_button.hide()
        self.generate_random_button.hide()
        self.generate_spectral_button.show()
        self.generate_tangles_button.show()
        self.generate_louvain_button.show()  # Show Louvain button
        self.numb_nodes.hide()
        self.numb_clusters.hide()
        self.average_edges_to_same_cluster.hide()
        self.average_edges_to_other_clusters.hide()
        self.agreement_parameter.show()
        self.k_spectral.show()
