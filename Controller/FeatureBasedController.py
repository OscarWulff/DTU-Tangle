import ast
import time
import numpy as np
from PyQt5.QtWidgets import QFileDialog, QCheckBox, QComboBox, QLineEdit, QPushButton, QMainWindow
from Model.GenerateTestData import GenerateDataFeatureBased
from Model.DataSetFeatureBased import tsne, read_file, pca, calculate_explained_varince, DataSetFeatureBased
from Model.TangleCluster import create_searchtree
from Model.SearchTree import condense_tree, soft_clustering, hard_clustering, contracting_search_tree
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

class FeatureBasedController:
    def __init__(self, view):
        self.view = view
        self.view.test_button.clicked.connect(self.test)
        self.view.upload_data_button.clicked.connect(self.upload_data_PCA)
        self.view.generate_fixed_button.clicked.connect(self.generate_fixed)
        self.view.generate_random_button.clicked.connect(self.generate_random)
        self.view.generate_tangles_button.clicked.connect(self.tangles)
        self.view.generate_spectral_button.clicked.connect(self.spectral)
        self.view.generate_Kmeans_button.clicked.connect(self.kmeans)
        self.view.cuts_button.clicked.connect(self.show_cuts)
        self.view.plot_tree.stateChanged.connect(self.plot_tree_changed)
        self.view.show_tangle.stateChanged.connect(self.tangle_show_changed)
        self.view.soft_clustering.stateChanged.connect(self.soft_clustering_changed)
        self.view.nmi.stateChanged.connect(self.nmi_changed)
        self.view.davies.stateChanged.connect(self.davies_changed)


    def generate_fixed(self):
        cluster_points = self.view.cluster_points.text()
        std = self.view.std.text()
        overlap = self.view.overlap.text()
        centroids = self.view.centroids.text()

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

        generated_data = GenerateDataFeatureBased(cluster_points, std)

        if overlap != None: 
            generated_data.overlap = overlap
        
        generated_data.fixed_clusters(cluster_points, centroids)
        self.view.original_points = generated_data.points

        if len(centroids[0]) > 2: 
            self.view.plotting_points = tsne(np.array(generated_data.points))
        else: 
            self.view.plotting_points = generated_data.points

        self.view.ground_truth = generated_data.ground_truth
        self.view.setup_plots()

    def generate_random(self):
        cluster_points = self.view.cluster_points.text()
        numb_clusters = self.view.numb_clusters.text()
        std = self.view.std.text()
        overlap = self.view.overlap.text()
        dimension = self.view.dimensions.text()
        try: 
            cluster_points = int(cluster_points)
            std = float(std) 
            numb_clusters = int(numb_clusters)  
            dimension = int(dimension)
        except ValueError: 
            print("Invalid input")

        try: 
            overlap = int(overlap)
        except ValueError:
            overlap = None

        generated_data = GenerateDataFeatureBased(numb_clusters, std)

        if overlap != None: 
            generated_data.overlap = overlap

        generated_data.random_clusters(cluster_points, dimension)
        self.view.original_points = generated_data.points
  
        if dimension > 2: 
            self.view.plotting_points = tsne(np.array(generated_data.points))
        else: 
            self.view.plotting_points = generated_data.points

        self.view.ground_truth = generated_data.ground_truth
        self.view.setup_plots()


    def upload_data(self):
        # Logic to upload data
        file_dialog = QFileDialog()
        if file_dialog.exec_():
            selected_file = file_dialog.selectedFiles()[0]  # Get the path of the selected file
            X = read_file(selected_file)
            data = tsne(X)
        
        self.view.original_points = data.tolist()
        self.view.plotting_points = data.tolist()

        self.view.ground_truth = [1] * len(data)

        self.view.upload_datas()
        self.view.setup_plots()

    def upload_data_PCA(self):
        file_dialog = QFileDialog()
        if file_dialog.exec_():
            selected_file = file_dialog.selectedFiles()[0]  # Get the path of the selected file
            X = read_file(selected_file)
            eigenvalues, data = pca(X)
        
        self.view.original_points = data.tolist()
        self.view.plotting_points = data.tolist()

        self.view.ground_truth = [1] * len(data)

        var = calculate_explained_varince(eigenvalues)
        self.view.variance.setText(f"Explained variance = {round((var[0]+var[1]) * 100)}%")
        self.view.variance.show()

        self.view.upload_datas()
        self.view.setup_plots()


    def tangles(self):
        a = self.view.agreement_parameter.text()
        cut_generator = self.view.cut_generator.currentText()
        cost_function = self.view.cost_function.currentText()

        try:      
            a = int(a)
        except ValueError: 
            print("Invalid input")
    
        # Creating the tangles
        generated_data = DataSetFeatureBased(a)

        generated_data.points = self.view.original_points
        self.view.tangles_points = self.view.plotting_points

        cut_generator_mapping = {
            "axis cuts": generated_data.cut_generator_axis_solveig
        }

        cost_function_mapping = {
            "pairwise cost": generated_data.pairwise_cost,
            "mean cost": generated_data.mean_cost 
        }

        cut = cut_generator_mapping[cut_generator]
        cost = cost_function_mapping[cost_function]
        start_time = time.time()
        cut()
        cost()
        root = create_searchtree(generated_data)
        self.tangle_root = condense_tree(root)
        contracting_search_tree(self.tangle_root)
        end_time = time.time()
        print(f"Time to create tangles: {end_time - start_time} seconds")
        soft = soft_clustering(self.tangle_root)
        hard = hard_clustering(soft)

        if self.view.tangles_plot == None: 
            self.view.numb_plots += 1   

        self.view.tangles_plot = hard
        self.prob = []

        for i in range(len(soft)):
            prob = 0
            for j in range(len(soft[0])):
                if soft[i][j] > prob:
                    prob = soft[i][j]
            self.prob.append(prob)
        self.view.nmi_score_tangles = round(generated_data.nmi_score(self.view.ground_truth, self.view.tangles_plot), 2)
        
        self.view.davies_score_tangles = round(generated_data.davies_bouldin_score(self.view.original_points, self.view.tangles_plot), 2)

        self.view.setup_plots()

    def spectral(self):
        k = self.view.k_spectral.text()

        try:       
            k = int(k)
        except ValueError: 
            print("Invalid input")

        generated_data = GenerateDataFeatureBased(0, 0)
        generated_data.points = self.view.original_points
        self.view.spectral_points = self.view.plotting_points

        if self.view.spectral_plot is None: 
            self.view.numb_plots += 1

        start_time = time.time()
        self.view.spectral_plot = generated_data.spectral_clustering(k)
        end_time = time.time()
        print(f"Time to create spectral: {end_time - start_time} seconds")
       
        self.view.nmi_score_spectral = round(generated_data.nmi_score(self.view.ground_truth, self.view.spectral_plot.tolist()), 2)
        self.view.davies_score_spectral = round(generated_data.davies_bouldin_score(self.view.original_points, self.view.spectral_plot.tolist()), 2)
        self.view.setup_plots()

    def kmeans(self):
        k = self.view.k_kmeans.text()

        try:      
            k = int(k)
        except ValueError: 
            print("Invalid input")

        generated_data = GenerateDataFeatureBased(0, 0)
        generated_data.points = self.view.original_points
        self.view.kmeans_points = self.view.plotting_points


        if self.view.kmeans_plot is None: 
            self.view.numb_plots += 1

        start_time = time.time()
        self.view.kmeans_plot = generated_data.k_means(k)
        end_time = time.time()
        print(f"Time to create kmeans: {end_time - start_time} seconds")

        self.view.nmi_score_kmeans = round(generated_data.nmi_score(self.view.ground_truth, self.view.kmeans_plot.tolist()), 2)
        self.view.davies_score_kmeans = round(generated_data.davies_bouldin_score(self.view.original_points, self.view.kmeans_plot.tolist()), 2)
        self.view.setup_plots()

    def show_cuts(self):
        # Logic to show cuts
        pass

    def plot_tree_changed(self, state):
        def plot_tree(node, depth=0, pos=(0, 0), x_offset=100, y_offset=100):
            if node is None:
                return

            # Draw current node
            tangle = set()
            for i, t in enumerate(node.tangle):
                if i == 0: 
                    tangle = t[0]
                else: 
                    tangle = tangle.intersection(t[0])

            plt.text(pos[0], pos[1], str(tangle), fontsize=6, ha='center', va='center', wrap=True, bbox=dict(facecolor='white', edgecolor='black', boxstyle='circle'))

            # Draw left subtree
            if node.left_node:
                left_pos = (pos[0] - x_offset, pos[1] - y_offset)
                plt.plot([pos[0], left_pos[0]], [pos[1], left_pos[1]], color='black')
                plot_tree(node.left_node, depth+1, left_pos)

            # Draw right subtree
            if node.right_node:
                right_pos = (pos[0] + x_offset, pos[1] - y_offset)
                plt.plot([pos[0], right_pos[0]], [pos[1], right_pos[1]], color='black')
                plot_tree(node.right_node, depth+1, right_pos)

        if state == 2: 
            plt.ion()
            plt.figure(figsize=(8, 8))
            plot_tree(self.tangle_root)
            plt.axis('off')
            plt.ioff()
            self.plot_tree.setChecked(False)
        else: 
            pass

    def tangle_show_changed(self, state):
        # Logic for changing tangle display state
        pass

    def soft_clustering_changed(self, state):
        if state == 2: 
            self.view.prob_checked = True
        else:
            self.view.prob_checked = False
        self.view.setup_plots()

    def nmi_changed(self, state):
        if state == 2: 
            self.view.nmi_checked = True
        else: 
            self.view.nmi_checked = False
        self.view.setup_plots()

    def davies_changed(self, state):
        if state == 2: 
            self.view.davies_checked = True
        else: 
            self.view.davies_checked = False
        self.view.setup_plots()
    
    def test(self):
        self.upload_data_button.hide()
        self.test_button.hide()
        self.generate_data_button.hide()
        self.canvas.deleteLater()



        # Noise
        # Data distribution - shapes, sizes, densities
        # overlap
        # number of data points
        # number of clusters
        # Test for Run-time and accuracy
