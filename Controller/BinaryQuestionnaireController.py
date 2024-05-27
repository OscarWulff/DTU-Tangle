
import ast
import time
import numpy as np
from PyQt5.QtWidgets import QFileDialog, QCheckBox, QComboBox, QLineEdit, QPushButton, QMainWindow
from Model.DataSetBinaryQuestionnaire import DataSetBinaryQuestionnaire, perform_tsne
from Model.DataSetFeatureBased import read_file, tsne
from Model.GenerateTestData import GenerateDataBinaryQuestionnaire, GenerateDataFeatureBased, export_to_csv
from Model.TangleCluster import create_searchtree
from Model.SearchTree import condense_tree, soft_clustering, hard_clustering, contracting_search_tree
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt


class BinaryQuestionnaireController:
    def __init__(self, view):
        self.view = view
        self.view.upload_data_button.clicked.connect(self.upload_data)
        self.view.generate_random_button.clicked.connect(self.generate_random)
        self.view.generate_tangles_button.clicked.connect(self.tangles)
        self.view.generate_DBSCAN_button.clicked.connect(self.dbscan)
        self.view.generate_Kmeans_button.clicked.connect(self.kmeans)
        self.view.export_button.clicked.connect(self.export_data)



    def generate_random(self):
        number_of_clusters = self.view.numb_clusters.text()
        number_of_questions = self.view.numb_questions.text()
        number_of_participants = self.view.numb_participants.text()
        dim_choice = self.view.dim_red.currentText()


        try: 
            number_of_clusters = int(number_of_clusters)
            number_of_questions = int(number_of_questions)
            number_of_participants = int(number_of_participants)
            # agreement_parameter = int(agreement_parameter)

            if (number_of_clusters <= 0 or number_of_questions <= 0 or number_of_participants <= 0 ):
                print("Error: All inputs must be positive integers.")
                return

        except ValueError: 
            print("Invalid input")
            return

    
        self.view.generated_data = GenerateDataBinaryQuestionnaire(number_of_participants, number_of_questions, number_of_clusters)

        self.view.generated_data.generate_biased_binary_questionnaire_answers()
        self.view.generated_data.res_to_points(dim_choice)

        

        
        self.view.generated_data.result
        
        self.view.setup_plots()


    def tangles(self):
        start_tangle = time.time()
        a = self.view.agreement_parameter.text()
       

        try:      
            a = int(a)

            if (a <= 0):
                print("Error: All inputs must be positive integers.")
                return
        except ValueError: 
            print("Invalid input")
            return
        print(self.view.generated_data.questionaire)
        # Creating the tangles
        data = DataSetBinaryQuestionnaire(a).cut_generator_binary(self.view.generated_data.questionaire, self.view.sim_fun.currentText())
        
       
        root = create_searchtree(data)
        
        self.view.tangle_root = condense_tree(root)
        contracting_search_tree(self.view.tangle_root)
        # time.sleep(2)
        soft = soft_clustering(self.view.tangle_root)

        prob_array = np.array(soft)

        # Find the index of the maximum value in each row
        max_indices = np.argmax(prob_array, axis=1)

        # Extract the maximum values using the indices
        max_values = prob_array[np.arange(prob_array.shape[0]), max_indices]

        self.view.prob = max_values
        print(soft)
        
        hard = hard_clustering(soft)

        self.view.time_tangles = round(time.time() - start_tangle, 2)


        if self.view.tangles_plot == None: 
            self.view.numb_plots += 1    
        
        self.prob = []

      

        self.view.tangles_plot = hard
        self.view.tangles_points = self.view.generated_data.points
        try:
            self.view.nmi_score_tangles = round(self.view.generated_data.nmi_score(hard), 2)
        except:
            self.view.nmi_score_tangles = 0
        self.view.setup_plots()


    def dbscan(self):
        dbscan_time = time.time()
        min_s = self.view.min_samples.text()
        eps = self.view.epsilon.text()
        print("min s :", min_s)
        print("eps :", eps)

        try:       
            min_s = int(min_s)
            eps = float(eps)

            if (min_s <= 0 or eps <= 0):
                print("Error: All inputs must be positive integers.")
                return
        except ValueError: 
            print("Invalid input")
            return

        generated_data = GenerateDataBinaryQuestionnaire(0, 0, 0)
        generated_data.points = self.view.generated_data.points
        self.view.dbscan_points = self.view.generated_data.points

        

        if self.view.dbscan_plot is None: 
            self.view.numb_plots += 1

        self.view.dbscan_plot = generated_data.DBscan(min_s, eps)
        self.view.time_dbscan = round(time.time() - dbscan_time, 2)
        self.view.nmi_score_dbscan = round(self.view.generated_data.nmi_score(self.view.dbscan_plot.tolist()), 2)
        self.view.setup_plots()

    def kmeans(self):
        kmeans_time = time.time()
        k = self.view.k_kmeans.text()

        try:      
            k = int(k)
            if (k <= 0):
                print("Error: All inputs must be positive integers.")
                return
            
        except ValueError: 
            print("Invalid input")
            return

        generated_data = GenerateDataBinaryQuestionnaire(0, 0, 0)
        generated_data.points = self.view.generated_data.points
        self.view.kmeans_points = self.view.generated_data.points
        if self.view.kmeans_plot is None: 
            self.view.numb_plots += 1
        self.view.kmeans_plot = generated_data.k_means(k)
        self.view.time_kmeans = round(time.time() - kmeans_time, 2)
        self.view.nmi_score_kmeans = round(self.view.generated_data.nmi_score(self.view.kmeans_plot.tolist()), 2)
        self.view.setup_plots()


    def upload_data(self):
        file_dialog = QFileDialog()
        if file_dialog.exec_():
            selected_file = file_dialog.selectedFiles()[0]
            
            X = read_file(selected_file)  # Get the path of the selected file
            print(X)
            self.data = tsne(X)
           
        self.view.upload_data_button.hide()
        self.view.generate_data_button.hide()
        self.view.generated_data = GenerateDataBinaryQuestionnaire(0, 0,0)
        self.view.generated_data.questionaire = X

        self.view.generated_data.points = [inner + [index] for index, inner in enumerate(self.data.tolist())]
        self.view.generated_data.ground_truth = [1] * len(self.view.generated_data.points)

        self.view.upload_data_show()
        

        self.view.setup_plots()
    
    def export_data(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self.view, "Save CSV", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if fileName:
            print("Saving data to:", fileName)
            export_to_csv(self.view.generated_data.questionaire, fileName)
    