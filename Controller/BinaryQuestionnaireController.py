import ast
import time
import numpy as np
from PyQt5.QtWidgets import QFileDialog, QCheckBox, QComboBox, QLineEdit, QPushButton, QMainWindow
from Model.DataSetBinaryQuestionnaire import DataSetBinaryQuestionnaire, perform_tsne
from Model.DataSetFeatureBased import tsne
from Model.GenerateTestData import GenerateDataBinaryQuestionnaire, GenerateDataFeatureBased
from Model.TangleCluster import create_searchtree
from Model.SearchTree import condense_tree, soft_clustering, hard_clustering, contracting_search_tree
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt


class BinaryQuestionnaireController:
    def __init__(self, view):
        self.view = view
        self.view.generate_random_button.clicked.connect(self.generate_random)
        self.view.generate_tangles_button.clicked.connect(self.tangles)
        self.view.generate_DBSCAN_button.clicked.connect(self.dbscan)
        self.view.generate_Kmeans_button.clicked.connect(self.kmeans)



    def generate_random(self):
        number_of_clusters = self.view.numb_clusters.text()
        number_of_questions = self.view.numb_questions.text()
        number_of_participants = self.view.numb_participants.text()
        agreement_parameter = self.view.agreement_parameter.text()


        try: 
            number_of_clusters = int(number_of_clusters)
            number_of_questions = int(number_of_questions)
            number_of_participants = int(number_of_participants)
            agreement_parameter = int(agreement_parameter)

        except ValueError: 
            print("Invalid input")

    
        self.view.generated_data = GenerateDataBinaryQuestionnaire(number_of_participants, number_of_questions, number_of_clusters)

        self.view.generated_data.generate_biased_binary_questionnaire_answers()
        self.view.generated_data.res_to_points()

        

        
        self.view.generated_data.result
        
        self.view.setup_plots()


    def tangles(self):
        a = self.view.agreement_parameter.text()
       

        try:      
            a = int(a)
        except ValueError: 
            print("Invalid input")
    
        # Creating the tangles
        data = DataSetBinaryQuestionnaire(a).cut_generator_binary(self.view.generated_data.questionaire)
        
       
        root = create_searchtree(data)
        
        self.view.tangle_root = condense_tree(root)
        contracting_search_tree(self.view.tangle_root)
        # time.sleep(2)
        soft = soft_clustering(self.view.tangle_root)
        # print(soft)
        
        hard = hard_clustering(soft)


        if self.view.tangles_plot == None: 
            self.view.numb_plots += 1    
        
        self.prob = []

      

        self.view.tangles_plot = hard
        self.view.tangles_points = self.view.generated_data.points

        self.view.nmi_score_tangles = round(self.view.generated_data.nmi_score(hard), 2)
        self.view.setup_plots()


    def dbscan(self):
        min_s = self.view.agreement_parameter.text()
        eps = self.view.epsilon.text()
        print("min s :", min_s)
        print("eps :", eps)

        try:       
            min_s = int(min_s)
            eps = float(eps)
        except ValueError: 
            print("Invalid input")

        generated_data = GenerateDataBinaryQuestionnaire(0, 0, 0)
        generated_data.points = self.view.generated_data.points
        self.view.dbscan_points = self.view.generated_data.points

        if self.view.dbscan_plot is None: 
            self.view.numb_plots += 1

        self.view.dbscan_plot = generated_data.DBscan(min_s, eps)
        self.view.nmi_score_dbscan = round(self.view.generated_data.nmi_score(self.view.dbscan_plot.tolist()), 2)
        self.view.setup_plots()

    def kmeans(self):
        k = self.view.k_kmeans.text()

        try:      
            k = int(k)
        except ValueError: 
            print("Invalid input")

        generated_data = GenerateDataBinaryQuestionnaire(0, 0, 0)
        generated_data.points = self.view.generated_data.points
        self.view.kmeans_points = self.view.generated_data.points
        if self.view.kmeans_plot is None: 
            self.view.numb_plots += 1
        self.view.kmeans_plot = generated_data.k_means(k)
        self.view.nmi_score_kmeans = round(self.view.generated_data.nmi_score(self.view.kmeans_plot.tolist()), 2)
        self.view.setup_plots()


    def upload_data(self):
        file_dialog = QFileDialog()
        if file_dialog.exec_():
            selected_file = file_dialog.selectedFiles()[0]  # Get the path of the selected file
            self.data = tsne(selected_file)
        self.view.upload_data_button.hide()
        self.view.generate_data_button.hide()
        self.view.generated_data = GenerateDataBinaryQuestionnaire(0, 0,0)

        self.view.generated_data.points = [inner + [index] for index, inner in enumerate(self.data.tolist())]
        self.view.generated_data.ground_truth = [1] * len(self.generated_data.points)

        self.view.upload_data_show()
        

        self.setup_plots()

    
