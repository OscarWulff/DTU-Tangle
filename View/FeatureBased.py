from PyQt5.QtWidgets import QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QWidget, QHBoxLayout, QLabel
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QWidget, QHBoxLayout, QLabel, QLineEdit, QComboBox, QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from Model.GenerateTestData import GenerateDataFeatureBased
import ast
from PyQt5.QtCore import Qt
from Model.DataSetFeatureBased import dimension_reduction_feature_based, calculate_explained_varince, DataSetFeatureBased
from Model.TangleCluster import create_searchtree
from Model.SearchTree import *


class FeatureBasedWindow(QMainWindow):
    def __init__(self, main_page):
        super().__init__()
        self.main_page = main_page
        self.setWindowTitle("Feature Based Window")
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
        self.tangles_plot = None
        self.tangles_points = None
        self.spectral_plot = None
        self.spectral_points = None
        self.kmeans_plot = None
        self.kmeans_points = None
        self.nmi_score_tangles = None
        self.nmi_score_kmeans = None
        self.nmi_score_spectral = None



        # Add buttons
        button_layout = QHBoxLayout()
        
        self.upload_data_button = QPushButton("Upload Data as .csv", self)
        self.upload_data_button.clicked.connect(self.upload_data)
        button_layout.addWidget(self.upload_data_button)
        
        self.generate_data_button = QPushButton("Generate Data", self)
        self.generate_data_button.clicked.connect(self.generate_data)
        button_layout.addWidget(self.generate_data_button)

        self.generate_fixed_button = QPushButton("generate", self)
        self.generate_fixed_button.clicked.connect(self.generate_fixed)
        button_layout.addWidget(self.generate_fixed_button)
        self.generate_fixed_button.hide()

        self.generate_random_button = QPushButton("generate", self)
        self.generate_random_button.clicked.connect(self.generate_random)
        button_layout.addWidget(self.generate_random_button)
        self.generate_random_button.hide()


        self.generate_tangles_button = QPushButton("apply tangles", self)
        self.generate_tangles_button.clicked.connect(self.tangles)
        button_layout.addWidget(self.generate_tangles_button)
        self.generate_tangles_button.hide()

        self.cuts_button = QPushButton("show cuts", self)
        self.cuts_button.clicked.connect(self.show_cuts)
        button_layout.addWidget(self.cuts_button)
        self.cuts_button.hide()
        
        self.generate_spectral_button = QPushButton("apply spectral", self)
        self.generate_spectral_button.clicked.connect(self.spectral)
        button_layout.addWidget(self.generate_spectral_button)
        self.generate_spectral_button.hide()

        
        self.generate_Kmeans_button = QPushButton("apply k-means", self)
        self.generate_Kmeans_button.clicked.connect(self.kmeans)
        button_layout.addWidget(self.generate_Kmeans_button)
        self.generate_Kmeans_button.hide()

        self.cluster_points = QLineEdit()
        self.cluster_points.setFixedSize(150, 30)  # Set a fixed size for the input field
        self.cluster_points.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.cluster_points.setPlaceholderText("Data-points pr cluster")
        self.cluster_points.hide()
        layout.addWidget(self.cluster_points)

        self.numb_clusters = QLineEdit()
        self.numb_clusters.setFixedSize(150, 30)  # Set a fixed size for the input field
        self.numb_clusters.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.numb_clusters.setPlaceholderText("amount of clusters")
        self.numb_clusters.hide()
        layout.addWidget(self.numb_clusters)

        self.std = QLineEdit()
        self.std.setFixedSize(150, 30)  # Set a fixed size for the input field
        self.std.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.std.setPlaceholderText("Standard deviation")
        self.std.hide()
        layout.addWidget(self.std)

        self.overlap = QLineEdit()
        self.overlap.setFixedSize(150, 30)  # Set a fixed size for the input field
        self.overlap.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.overlap.setPlaceholderText("Allowed overlap")
        self.overlap.hide()
        layout.addWidget(self.overlap)

        self.centroids = QLineEdit()
        self.centroids.setFixedSize(300, 30)  # Set a fixed size for the input field
        self.centroids.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.centroids.setPlaceholderText("Assign centers, i.e. [(2, 2), (4, 1)]")
        self.centroids.hide()
        layout.addWidget(self.centroids)

        self.k_spectral = QLineEdit()
        self.k_spectral.setFixedSize(300, 30)  # Set a fixed size for the input field
        self.k_spectral.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.k_spectral.setPlaceholderText("k for spectral")
        self.k_spectral.hide()
        layout.addWidget(self.k_spectral)

        self.k_kmeans = QLineEdit()
        self.k_kmeans.setFixedSize(300, 30)  # Set a fixed size for the input field
        self.k_kmeans.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.k_kmeans.setPlaceholderText("k for kmeans")
        self.k_kmeans.hide()
        layout.addWidget(self.k_kmeans)

        self.random_centers_button = QPushButton("Clusters with random centers", self)
        self.random_centers_button.clicked.connect(self.random_centers)
        button_layout.addWidget(self.random_centers_button)
        self.random_centers_button.hide()
        
        self.fixed_centers_button = QPushButton("Clusters with fixes centers", self)
        self.fixed_centers_button.clicked.connect(self.fixed_centers)
        button_layout.addWidget(self.fixed_centers_button)
        self.fixed_centers_button.hide()

        self.variance = QLabel(self)
        self.variance.setAlignment(Qt.AlignRight | Qt.AlignTop)
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


    def show_cuts(self): 
        splitting_nodes = []

        # Recursive function to traverse the tree
        def traverse(node):
            if node is None:
                return
            if node.left_node is not None and node.right_node is not None:
                splitting_nodes.append(node)
            traverse(node.left_node)
            traverse(node.right_node)

        # Start traversal from the root node
        traverse(self.tangle_root)

        x = []
        y = []
 

        for node in splitting_nodes:
            if node.cut.line_placement[1] == "x":
                x.append(node.cut.line_placement[0]) 
            else: 
                y.append(node.cut.line_placement[0])

        plot = None
        self.figure.delaxes(self.figure.axes[1]) 
        
        if self.numb_plots == 2:
            plot = self.figure.add_subplot(122)
            for x_value in x:
                plot.axvline(x=x_value, color='r', linestyle='--')
            for y_value in y: 
                plot.axhline(y=y_value, color='r', linestyle='--')

            if self.tangles_plot != None: 
                plot.set_title('Tangles')
                self.plot_points(self.tangles_points, self.tangles_plot, plot)
        else: 
            plot = self.figure.add_subplot(222)
            for x_value in x:
                plot.axvline(x=x_value, color='r', linestyle='--')
            for y_value in y: 
                plot.axhline(y=y_value, color='r', linestyle='--')
            if self.tangles_plot != None: 
                plot.set_title('Tangles')
                self.plot_points(self.tangles_points, self.tangles_plot, plot)
        
    
        self.canvas.draw()


    def tangles(self):
    
        # Creating the tangles
        data = DataSetFeatureBased(5)
        self.tangles_points = self.generated_data.points
        data.points = self.generated_data.points
        data.cut_generator_axis()
        data.cost_function()
        root = create_searchtree(data)
        print_tree(root)
        self.tangle_root = condense_tree(root)
        contracting_search_tree(self.tangle_root)
        print_tree(self.tangle_root)
        soft = soft_clustering(self.tangle_root)
        print(soft)
        hard = hard_clustering(soft)


        if self.tangles_plot == None: 
            self.numb_plots += 1
        self.tangles_plot = hard

        self.nmi_score_tangles = round(self.generated_data.nmi_score(hard), 2)
        self.setup_plots()
        

    def spectral(self):
        k = self.k_spectral.text()

        try:       
            k = int(k)
        except ValueError: 
            print("Invalid input")

        generated_data = GenerateDataFeatureBased(0, 0)
        generated_data.points = self.generated_data.points
        self.spectral_points = self.generated_data.points

        if self.spectral_plot is None: 
            self.numb_plots += 1

        self.spectral_plot = generated_data.spectral_clustering(k)
        self.nmi_score_spectral = round(self.generated_data.nmi_score(self.spectral_plot.tolist()), 2)
        self.setup_plots()


    def kmeans(self):
        k = self.k_kmeans.text()

        try:      
            k = int(k)
        except ValueError: 
            print("Invalid input")

        generated_data = GenerateDataFeatureBased(0, 0)
        generated_data.points = self.generated_data.points
        self.kmeans_points = self.generated_data.points
        if self.kmeans_plot is None: 
            self.numb_plots += 1
        self.kmeans_plot = generated_data.k_means(k)
        self.nmi_score_kmeans = round(self.generated_data.nmi_score(self.kmeans_plot.tolist()), 2)
        self.setup_plots()


    def fixed_centers(self):
        self.random_centers_button.hide()
        self.fixed_centers_button.hide()
        self.generate_fixed_button.show()
        self.generate_Kmeans_button.show()
        self.generate_spectral_button.show()
        self.generate_tangles_button.show()

        self.cluster_points.show()
        self.std.show()
        self.centroids.show()
        self.k_spectral.show()
        self.k_kmeans.show()
        self.cuts_button.show()

        
    def generate_fixed(self): 
        cluster_points = self.cluster_points.text()
        std = self.std.text()
        overlap = self.overlap.text()
        centroids = self.centroids.text()

        try: 
            cluster_points = int(cluster_points)
            std = float(std)        
            centroids = ast.literal_eval(centroids)

        except ValueError: 
            print("Invalid input")

        try: 
            overlap = int(overlap)
        except ValueError:
            overlap = None

        self.generated_data = GenerateDataFeatureBased(cluster_points, std)

        if overlap != None: 
            self.generated_data.overlap = overlap
        
        self.generated_data.fixed_clusters(cluster_points, centroids)
        
        self.setup_plots()


    def generate_random(self):
        cluster_points = self.cluster_points.text()
        numb_clusters = self.numb_clusters.text()
        std = self.std.text()
        overlap = self.overlap.text()

        try: 
            cluster_points = int(cluster_points)
            std = float(std) 
            numb_clusters = int(numb_clusters)  
        except ValueError: 
            print("Invalid input")

        try: 
            overlap = int(overlap)
        except ValueError:
            overlap = None

        self.generated_data = GenerateDataFeatureBased(numb_clusters, std)

        if overlap != None: 
            self.generated_data.overlap = overlap

        self.generated_data.random_clusters(cluster_points)
        
        self.setup_plots()


    def random_centers(self):
        self.random_centers_button.hide()
        self.fixed_centers_button.hide()
        self.generate_random_button.show()
        self.generate_Kmeans_button.show()
        self.generate_spectral_button.show()
        self.generate_tangles_button.show()
        self.numb_clusters.show()
        self.cluster_points.show()
        self.std.show()
        self.overlap.show()
        self.k_spectral.show()
        self.k_kmeans.show()
        self.cuts_button.show()

    
    def setup_plots(self):
        self.figure.clear()

        if self.numb_plots == 1:
            plot = self.figure.add_subplot(111)
            plot.set_title('Ground thruth')
            self.plot_points(self.generated_data.points, self.generated_data.ground_truth, plot)
        elif self.numb_plots == 2:
            if self.generated_data != None: 
                plot = self.figure.add_subplot(121)
                plot.set_title('Ground truth')
                self.plot_points(self.generated_data.points, self.generated_data.ground_truth, plot)
            if self.tangles_plot != None: 
                plot = self.figure.add_subplot(122)
                plot.set_title('Tangles')
                self.plot_points(self.tangles_points, self.tangles_plot, plot)
            if self.spectral_plot is not None: 
                plot = self.figure.add_subplot(122)
                plot.set_title('Spectral')
                self.plot_points(self.spectral_points, self.spectral_plot, plot)
            if self.kmeans_plot is not None: 
                plot = self.figure.add_subplot(122)
                plot.set_title('K-means')
                self.plot_points(self.kmeans_points, self.kmeans_plot, plot)
        else:
            if self.generated_data != None: 
                plot = self.figure.add_subplot(221)
                plot.set_title('Ground truth')
                self.plot_points(self.generated_data.points, self.generated_data.ground_truth, plot)
            if self.tangles_plot != None: 
                plot = self.figure.add_subplot(222)
                plot.set_title('Tangles')
                self.plot_points(self.tangles_points, self.tangles_plot, plot)
            if self.spectral_plot is not None: 
                plot = self.figure.add_subplot(223)
                plot.set_title('Spectral')
                self.plot_points(self.spectral_points, self.spectral_plot, plot)
            if self.kmeans_plot is not None: 
                plot = self.figure.add_subplot(224)
                plot.set_title('K-means')
                self.plot_points(self.kmeans_points, self.kmeans_plot, plot)

        self.figure.subplots_adjust(hspace=0.5, wspace=0.5)
        self.canvas.draw()

    def plot_points(self, points, labels, plot):

        clusters = sorted(set(labels))
        colors = plt.cm.viridis(np.linspace(0, 1, len(clusters)))
        color_map = {cluster: color for cluster, color in zip(clusters, colors)}

        if (plot.get_title() == "Tangles"):
             plot.text(0.95, 0.95, f'nmi = {self.nmi_score_tangles}', verticalalignment='top', horizontalalignment='right', transform=plot.transAxes, fontsize=10)
        if (plot.get_title() == "Spectral"):
             plot.text(0.95, 0.95, f'nmi = {self.nmi_score_spectral}', verticalalignment='top', horizontalalignment='right', transform=plot.transAxes, fontsize=10)
        if (plot.get_title() == "K-means"):
             plot.text(0.95, 0.95, f'nmi = {self.nmi_score_kmeans}', verticalalignment='top', horizontalalignment='right', transform=plot.transAxes, fontsize=10)

        # Plot the points with color
        for point, truth in zip(points, labels):
            plot.scatter(point[0], point[1], color=color_map[truth])

        # Add labels and title
        plot.set_xlabel('X')
        plot.set_ylabel('Y')
        

    def go_back_to_main_page(self):
        self.close()  # Close the current window
        # Show the main page again
        self.main_page.show()

    def upload_data(self):
        file_dialog = QFileDialog()
        if file_dialog.exec_():
            selected_file = file_dialog.selectedFiles()[0]  # Get the path of the selected file
            self.eigenvalues, self.data = dimension_reduction_feature_based(selected_file)
        self.upload_data_button.hide()
        self.generate_data_button.hide()
        self.generated_data = GenerateDataFeatureBased(0, 0)

        self.generated_data.points = [inner + [index] for index, inner in enumerate(self.data.tolist())]
        self.generated_data.ground_truth = [1] * len(self.generated_data.points)

        var = calculate_explained_varince(self.eigenvalues)
        self.variance.setText(f"Explained variance = {round((var[0]+var[1]) * 100)}%")
        self.variance.show()
        # Display the plot
        self.canvas.draw()
        self.generate_Kmeans_button.show()
        self.generate_spectral_button.show()
        self.generate_tangles_button.show()
        self.k_spectral.show()
        self.k_kmeans.show()
        self.cuts_button.show()

        self.setup_uploaded_plots()

    def setup_uploaded_plots(self):
        self.figure.clear()

        if self.numb_plots == 1:
            plot = self.figure.add_subplot(111)
            plot.set_title('Ground truth')
            self.plot_uploaded_points(self.generated_data.points, self.generated_data.ground_truth, plot)
        elif self.numb_plots == 2:
            if self.generated_data != None: 
                plot = self.figure.add_subplot(121)
                plot.set_title('Ground truth')
                self.plot_uploaded_points(self.generated_data.points, self.generated_data.ground_truth, plot)
            if self.tangles_plot != None: 
                plot = self.figure.add_subplot(122)
                plot.set_title('Tangles')
                self.plot_uploaded_points(self.tangles_points, self.tangles_plot, plot)
            if self.spectral_plot is not None: 
                plot = self.figure.add_subplot(122)
                plot.set_title('Spectral')
                self.plot_uploaded_points(self.spectral_points, self.spectral_plot, plot)
            if self.kmeans_plot is not None: 
                plot = self.figure.add_subplot(122)
                plot.set_title('K-means')
                self.plot_uploaded_points(self.kmeans_points, self.kmeans_plot, plot)
        else:
            if self.generated_data != None: 
                plot = self.figure.add_subplot(221)
                plot.set_title('Ground truth')
                self.plot_uploaded_points(self.generated_data.points, self.generated_data.ground_truth, plot)
            if self.tangles_plot != None: 
                plot = self.figure.add_subplot(222)
                plot.set_title('Tangles')
                self.plot_uploaded_points(self.tangles_points, self.tangles_plot, plot)
            if self.spectral_plot is not None: 
                plot = self.figure.add_subplot(223)
                plot.set_title('Spectral')
                self.plot_uploaded_points(self.spectral_points, self.spectral_plot, plot)
            if self.kmeans_plot is not None: 
                plot = self.figure.add_subplot(224)
                plot.set_title('K-means')
                self.plot_uploaded_points(self.kmeans_points, self.kmeans_plot, plot)

        self.canvas.draw()

    def plot_uploaded_points(self, points, labels, plot):

        clusters = sorted(set(labels))
        colors = plt.cm.viridis(np.linspace(0, 1, len(clusters)))
        color_map = {cluster: color for cluster, color in zip(clusters, colors)}

        # Plot the points with color
        for point, truth in zip(points, labels):
            plot.scatter(point[0], point[1], color=color_map[truth])

        # Add labels and title
        plot.set_xlabel('X')
        plot.set_ylabel('Y')
        

    def generate_data(self):
        self.upload_data_button.hide()
        self.generate_data_button.hide()
        self.random_centers_button.show()
        self.fixed_centers_button.show()