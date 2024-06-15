import ast
import time
import numpy as np
from PyQt5.QtWidgets import QFileDialog, QCheckBox, QComboBox, QLineEdit, QPushButton, QMainWindow, QMessageBox
from Model.GenerateTestData import GenerateDataFeatureBased, export_fig_to_jpg, export_to_csv_feature, export_to_csv_groundtruth
from Model.DataSetFeatureBased import tsne, read_file, pca, calculate_explained_varince, DataSetFeatureBased
from Model.TangleCluster import create_searchtree
from Model.SearchTree import condense_tree, soft_clustering, hard_clustering, contracting_search_tree
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

class FeatureBasedController:
    def __init__(self, view):
        self.view = view
        self.view.upload_data_button.clicked.connect(self.upload_data_PCA)
        self.view.generate_fixed_button.clicked.connect(self.generate_fixed)
        self.view.generate_random_button.clicked.connect(self.generate_random)
        self.view.generate_tangles_button.clicked.connect(self.tangles)
        self.view.generate_spectral_button.clicked.connect(self.spectral)
        self.view.generate_Kmeans_button.clicked.connect(self.kmeans)
        self.view.cuts_button.clicked.connect(self.show_cuts)
        self.view.show_tangle.stateChanged.connect(self.tangle_show_changed)
        self.view.soft_clustering.stateChanged.connect(self.soft_clustering_changed)
        self.view.nmi.stateChanged.connect(self.nmi_changed)
        self.view.davies.stateChanged.connect(self.davies_changed)
        self.view.export_button.clicked.connect(self.export_data)
        self.view.export_plot_button.clicked.connect(self.export_plot)
        self.view.export_groundtruth_button.clicked.connect(self.export_groundtruth)


    def generate_fixed(self):

        # function to generate fixed clusters
        try: 
            # Get the parameters
            cluster_points = int(self.view.cluster_points.text())
            std = float(self.view.std.text())
            centroids = ast.literal_eval(self.view.centroids.text())

            # create a generated data object
            generated_data = GenerateDataFeatureBased(cluster_points, std)
            
            # generate the fixed clusters
            generated_data.fixed_clusters(cluster_points, centroids)
            self.view.original_points = generated_data.points

            # using dimensionality reduction to plot the data if the data is more than 2D
            if len(centroids[0]) > 2: 
                self.view.plotting_points = tsne(np.array(generated_data.points))
            else: 
                self.view.plotting_points = generated_data.points

            self.view.ground_truth = generated_data.ground_truth

            # setup the plots
            self.view.setup_plots()
        except Exception as e:
            QMessageBox.warning(self.view, "Upload Data", f"Error: {e}")

    def generate_random(self):

        # function to generate random clusters
        try: 
            # Get the parameters
            cluster_points = int(self.view.cluster_points.text())
            numb_clusters = int(self.view.numb_clusters.text())
            std = float(self.view.std.text())
            overlap = float(self.view.overlap.text())
            dimension = int(self.view.dimensions.text())
        
            # create a generated data object
            generated_data = GenerateDataFeatureBased(numb_clusters, std)

            if overlap != None: 
                generated_data.overlap = overlap

            # generate the random clusters
            generated_data.random_clusters(cluster_points, dimension)
            self.view.original_points = generated_data.points

            # using dimensionality reduction to plot the data if the data is more than 2D
            if dimension > 2: 
                self.view.plotting_points = tsne(np.array(generated_data.points))
            else: 
                self.view.plotting_points = generated_data.points

            self.view.ground_truth = generated_data.ground_truth

            # setup the plots
            self.view.setup_plots()

        except Exception as e:
            QMessageBox.warning(self.view, "Upload Data", f"Error: {e}")

    def upload_data(self):
        # Logic to upload data with T-SNE

        try: 
            # obtaining the data from the file
            file_dialog = QFileDialog()
            if file_dialog.exec_():
                selected_file = file_dialog.selectedFiles()[0]  # Get the path of the selected file
                X = read_file(selected_file)
                data = tsne(X)
            
            self.view.original_points = data.tolist()
            self.view.plotting_points = data.tolist()

            # setting the ground truth to one cluster
            self.view.ground_truth = [1] * len(data)

            self.view.upload_datas()

            # setup the plots
            self.view.setup_plots()
        except Exception as e:
            QMessageBox.warning(self.view, "Upload Data", f"Error: {e}")

    def upload_data_PCA(self):
        # Logic to upload data with PCA
        try: 

            # obtaining the data from the file
            file_dialog = QFileDialog()
            if file_dialog.exec_():
                selected_file = file_dialog.selectedFiles()[0]  # Get the path of the selected file
                X = read_file(selected_file)
                eigenvalues, data = pca(X)
            
            self.view.original_points = data.tolist()
            self.view.plotting_points = data.tolist()

            # setting the ground truth to one cluster
            self.view.ground_truth = [1] * len(data)

            # calculating the explained variance
            var = calculate_explained_varince(eigenvalues)
            self.view.variance.setText(f"Explained variance = {round((var[0]+var[1]) * 100)}%")
            self.view.variance.show()

            self.view.upload_datas()

            # setup the plots
            self.view.setup_plots()
        except Exception as e:
            QMessageBox.warning(self.view, "Upload Data", f"Error: {e}")


    def tangles(self):

        # function to run tangles

        try:
            # Get the parameters
            a = int(self.view.agreement_parameter.text())
            cut_generator = self.view.cut_generator.currentText()
            cost_function = self.view.cost_function.currentText()
        
            # Creating the tangles
            generated_data = DataSetFeatureBased(a)
            generated_data.points = self.view.original_points
            self.view.tangles_points = self.view.plotting_points


            cut_generator_mapping = {
                "axis cuts": generated_data.cut_generator_axis_dimensions,
                "mean cuts": generated_data.mean_cut,
                "adjusted cuts": generated_data.adjusted_cut,
                "spectral cuts": generated_data.cut_spectral,
            }

            cost_function_mapping = {
                "pairwise cost": generated_data.pairwise_cost,
                "mean cost": generated_data.mean_cost, 
                "cure cost": generated_data.CURE_cost,
            }

            # finding the cut and cost functions
            cut = cut_generator_mapping[cut_generator]
            cost = cost_function_mapping[cost_function]

            start_time = time.time()

            # executing the tangles
            cut()
            cost()
            root = create_searchtree(generated_data)
            self.tangle_root = condense_tree(root)
            contracting_search_tree(self.tangle_root)
            self.view.tangle_root = self.tangle_root
            soft = soft_clustering(self.tangle_root)
            hard = hard_clustering(soft)
        
            end_time = time.time()

            self.view.time_tangles = round(end_time - start_time, 2)

            if self.view.tangles_plot == None: 
                self.view.numb_plots += 1   

            self.view.tangles_plot = hard
            self.view.prob = []

            # calculating the probability used for plotting the soft clustering
            for i in range(len(soft)):
                prob = 0
                for j in range(len(soft[0])):
                    if soft[i][j] > prob:
                        prob = soft[i][j]
                self.view.prob.append(prob)
            
            # calculating the scores
            self.view.nmi_score_tangles = round(generated_data.nmi_score(self.view.ground_truth, self.view.tangles_plot), 2)
            self.view.davies_score_tangles = round(generated_data.davies_bouldin_score(self.view.original_points, self.view.tangles_plot), 2)

            # setup the plots
            self.view.setup_plots()
        except Exception as e:
            QMessageBox.warning(self.view, "Upload Data", f"Error: {e}")
            
    def spectral(self):

        # function to run spectral clustering

        try: 
            k = int(self.view.k_spectral.text())

            # Create the data
            generated_data = GenerateDataFeatureBased(0, 0)
            generated_data.points = self.view.original_points
            self.view.spectral_points = self.view.plotting_points

            if self.view.spectral_plot is None: 
                self.view.numb_plots += 1

            # run spectral clustering
            start_time = time.time()
            self.view.spectral_plot = generated_data.spectral_clustering(k)
            end_time = time.time()
            self.view.time_spectral = round(end_time - start_time, 2)

            # calculate the scores
            self.view.nmi_score_spectral = round(generated_data.nmi_score(self.view.ground_truth, self.view.spectral_plot.tolist()), 2)
            self.view.davies_score_spectral = round(generated_data.davies_bouldin_score(self.view.original_points, self.view.spectral_plot.tolist()), 2)

            # setup the plots
            self.view.setup_plots()
        except Exception as e:
            QMessageBox.warning(self.view, "Upload Data", f"Error: {e}")

    def kmeans(self):

        # function to run kmeans

        try: 
            k = int(self.view.k_kmeans.text())

            # Create the data
            generated_data = GenerateDataFeatureBased(0, 0)
            generated_data.points = self.view.original_points
            self.view.kmeans_points = self.view.plotting_points


            if self.view.kmeans_plot is None: 
                self.view.numb_plots += 1

            # run kmeans
            start_time = time.time()
            self.view.kmeans_plot = generated_data.k_means(k)
            end_time = time.time()

            self.view.time_kmeans = round(end_time - start_time, 2)

            # calculate the scores
            self.view.nmi_score_kmeans = round(generated_data.nmi_score(self.view.ground_truth, self.view.kmeans_plot.tolist()), 2)
            self.view.davies_score_kmeans = round(generated_data.davies_bouldin_score(self.view.original_points, self.view.kmeans_plot.tolist()), 2)

            # setup the plots
            self.view.setup_plots()

        except Exception as e:
            QMessageBox.warning(self.view, "Upload Data", f"Error: {e}")

    def show_cuts(self, state):
        # Logic to show cuts
        if state == 2:
            self.view.show_cuts = True
        else:
            self.view.show_cuts = False
        self.view.setup_plots()

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

    
    def export_data(self):
        # Logic to export data
        try: 
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            fileName, _ = QFileDialog.getSaveFileName(self.view, "Save CSV", "", "CSV Files (*.csv);;All Files (*)", options=options)
            if fileName:
                print("Saving data to:", fileName)
                export_to_csv_feature(self.view.original_points, fileName + ".csv")
        except Exception as e:
            QMessageBox.warning(self.view, "Export Data", f"Error: {e}")

    def export_plot(self):
        # Logic to export plot
        try: 
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            # Change the dialog to save JPG files
            fileName, _ = QFileDialog.getSaveFileName(self.view, "Save Plot", "", "JPEG Files (*.jpg);;All Files (*)", options=options)
            if fileName:
                print("Saving plot to:", fileName)
                export_fig_to_jpg(self.view.figure ,fileName + ".jpg")
        except Exception as e:
            QMessageBox.warning(self.view, "Export Plot", f"Error: {e}")

    def export_groundtruth(self):
        # Logic to export data
        try: 
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            fileName, _ = QFileDialog.getSaveFileName(self.view, "Save ground truth", "", "CSV Files (*.csv);;All Files (*)", options=options)
            if fileName:
                print("Saving data to:", fileName)
                export_to_csv_groundtruth(self.view.ground_truth, fileName + ".csv")
        except Exception as e:
            QMessageBox.warning(self.view, "Export Data", f"Error: {e}")