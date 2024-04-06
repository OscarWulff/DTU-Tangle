import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QWidget, QHBoxLayout, QLabel, QLineEdit, QComboBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QWidget, QHBoxLayout, QLabel, QLineEdit, QComboBox, QSizePolicy
from PyQt5.QtCore import Qt
import numpy as np
import matplotlib.pyplot as plt
import time

from Model.GenerateTestData import GenerateDataBinaryQuestionnaire
from Model.DataSetBinaryQuestionnaire import DataSetBinaryQuestionnaire, perform_tsne
from Model.SearchTree import generate_color_dict, print_tree
from Model.TangleCluster import create_searchtree, condense_tree, contracting_search_tree, soft_clustering, hard_clustering

class BinaryQuestionnaireWindow(QMainWindow):
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



        self.color_dict = None
        self.tangle = None
        self.soft = None
        self.hard = None
        self.agree = None
        

        # self.initUI()

        button_layout = QHBoxLayout()
        
        self.upload_data_button = QPushButton("Upload Data as .csv", self)
        self.upload_data_button.clicked.connect(self.upload_data)
        button_layout.addWidget(self.upload_data_button)
        
        self.generate_data_button = QPushButton("Generate Data", self)
        self.generate_data_button.clicked.connect(self.random_centers)
        button_layout.addWidget(self.generate_data_button)

        self.generate_fixed_button = QPushButton("generate", self)
        self.generate_fixed_button.clicked.connect(self.test)
        button_layout.addWidget(self.generate_fixed_button)
        self.generate_fixed_button.hide()

        self.generate_random_button = QPushButton("generate", self)
        self.generate_random_button.clicked.connect(self.generate_random_data)
        button_layout.addWidget(self.generate_random_button)
        self.generate_random_button.hide()


        self.generate_tangles_button = QPushButton("apply tangles", self)
        self.generate_tangles_button.clicked.connect(self.display_data_window)
        button_layout.addWidget(self.generate_tangles_button)
        self.generate_tangles_button.hide()

        self.cuts_button = QPushButton("show cuts", self)
        self.cuts_button.clicked.connect(self.test)
        button_layout.addWidget(self.cuts_button)
        self.cuts_button.hide()
        
        self.generate_spectral_button = QPushButton("apply spectral", self)
        self.generate_spectral_button.clicked.connect(self.test)
        button_layout.addWidget(self.generate_spectral_button)
        self.generate_spectral_button.hide()

        
        self.generate_Kmeans_button = QPushButton("apply k-means", self)
        self.generate_Kmeans_button.clicked.connect(self.test)
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
        self.numb_clusters.setPlaceholderText("Amount of clusters")
        self.numb_clusters.hide()
        layout.addWidget(self.numb_clusters)

        self.numb_questions = QLineEdit()
        self.numb_questions.setFixedSize(150, 30)  # Set a fixed size for the input field
        self.numb_questions.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.numb_questions.setPlaceholderText("Number of questions")
        self.numb_questions.hide()
        layout.addWidget(self.numb_questions)

        self.agree_param = QLineEdit()
        self.agree_param.setFixedSize(150, 30)  # Set a fixed size for the input field
        self.agree_param.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.agree_param.setPlaceholderText("Agreement parameter")
        self.agree_param.hide()
        layout.addWidget(self.agree_param)

        self.samples = QLineEdit()
        self.samples.setFixedSize(150, 30)  # Set a fixed size for the input field
        self.samples.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.samples.setPlaceholderText("Number of samples")
        self.samples.hide()
        layout.addWidget(self.samples)

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
    def random_centers(self):
            self.generate_data_button.hide()
            self.upload_data_button.hide()
            self.generate_random_button.show()
            # self.generate_Kmeans_button.show()
            # self.generate_spectral_button.show()
            self.numb_questions.show()
            self.generate_tangles_button.show()
            self.numb_clusters.show()
            # self.cluster_points.show()
            self.agree_param.show()
            self.samples.show()
            # self.k_spectral.show()
            # self.k_kmeans.show()
            # self.cuts_button.show()

    def test(self):
        pass    

    
    
    def go_back_to_main_page(self):
        self.close()  # Close the current window
        # Show the main page again
        self.main_page.show()


    def initUI(self):

        # self.hard = None
        # self.soft = None
        # # self.color_dict = None
        # # self.tangle = None
        # self.data_generator = None
        # self.tree = None
        # self.tree_holder = None
        
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Spacer
        layout.addStretch()

        # Matplotlib figure and canvas for plotting
        self.figure = plt.figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Setup UI components similar to FeatureBasedWindow but for binary questionnaire data
        self.setupButtons(layout)

        # Spacer
        layout.addStretch()

        # Back button
        self.back_button = QPushButton("Back", self)
        self.back_button.clicked.connect(self.close)
        layout.addWidget(self.back_button)

    def goToInitialPage(self):
        self.initUI() 
        self.display_data_window()

    # def setup_generate_data(self):
    #     self.figure.clear()
    #     # Clear existing widgets from the layout
        
    #     central_widget = QWidget(self)
    #     self.setCentralWidget(central_widget)
    #     layout = QVBoxLayout(central_widget)

    #     # Add the generate data UI components
    #     self.num_questions_label = QLabel("Number of Questions:", self)
    #     layout.addWidget(self.num_questions_label)
    #     self.num_questions_input = QLineEdit(self)
    #     layout.addWidget(self.num_questions_input)

    #     # Number of samples per cluster
    #     self.num_samples_per_cluster_label = QLabel("Number of Samples:", self)
    #     layout.addWidget(self.num_samples_per_cluster_label)
    #     self.num_samples_per_cluster_input = QLineEdit(self)
    #     layout.addWidget(self.num_samples_per_cluster_input)

    #     # Number of clusters
    #     self.num_clusters_label = QLabel("Number of Clusters:", self)
    #     layout.addWidget(self.num_clusters_label)
    #     self.num_clusters_input = QLineEdit(self)
    #     layout.addWidget(self.num_clusters_input)

    #     self.generate_button = QPushButton("Generate", self)
    #     self.generate_button.clicked.connect(self.generate_random_data)
    #     self.generate_button.clicked.connect(self.goToInitialPage)

    #     self.agree_label = QLabel("Agreement parameter:", self)
    #     layout.addWidget(self.agree_label)
    #     self.agree_input = QLineEdit(self)
    #     layout.addWidget(self.agree_input)

    #     layout.addWidget(self.generate_button)
        

        # You may want to add a button for confirming data generation here,
        # and then connect it to the actual data generation method

        # # Spacer
        # layout.addStretch()

        # # Back button, similar to the initial UI setup
        # self.back_button = QPushButton("Back", self)
        # self.back_button.clicked.connect(self.close)
        # layout.addWidget(self.back_button)

        
    def setupButtons(self, layout):
        # Upload data button
        self.upload_data_button = QPushButton("Upload Data as .csv", self)
        self.upload_data_button.clicked.connect(self.upload_data)
        layout.addWidget(self.upload_data_button)

        # Generate data button for binary questionnaire
        self.generate_data_button = QPushButton("Generate Binary Data", self)
        self.generate_data_button.clicked.connect(self.setup_generate_data)
        layout.addWidget(self.generate_data_button)

        # Button to initiate analysis - Example with tangles or any other analysis applicable to your binary data
        self.analyze_data_button = QPushButton("Analyze Data", self)
        self.analyze_data_button.clicked.connect(self.go_back)
        layout.addWidget(self.analyze_data_button)
        self.analyze_data_button.hide()  # Initially hidden, shown after data is loaded/generated

   

    def upload_data(self):
        file_dialog = QFileDialog()
        if file_dialog.exec_():
            selected_file = file_dialog.selectedFiles()[0]  # Get the path of the selected file
            with open(selected_file, 'r') as file:
                # Load the data from the file as integers
                data = np.loadtxt(file, delimiter=',', dtype=int)
                self.data = data
                
                print("Data uploaded:", self.data)  # Debugging

    def generate_random_data(self):
        self.figure.clear()
        self.data = None

        # Retrieve user inputs
        num_questions = int(self.numb_questions.text())
        num_samples = int(self.samples.text()) # Total samples
        num_clusters = int(self.numb_clusters.text())
        self.agree = float(self.agree_param.text())
       
        # Initialize and generate data
        self.data_generator = GenerateDataBinaryQuestionnaire()
       
        data, _ = self.data_generator.generate_biased_binary_questionnaire_answers(num_samples, num_questions,  num_clusters)  # Assuming your class has a generate method to populate the data attribute
        
        data_array = np.array([bitset.bitset for bitset in data])
       
        self.data = data_array
      
        

        # self.display_data_window()
        # Display the random data
        # self.plot_data(self.data)

    def display_data_window(self):
        self.figure.clear()
        if self.data is not None:
            self.tree_holder = None
            self.tree = None
            holder = DataSetBinaryQuestionnaire(self.agree).cut_generator_binary(self.data)
            holder.cuts = holder.order_function()
            # for cut in holder.cuts:
            #     print(cut.A, cut.Ac, cut.cost)
          
            self.tree_holder = create_searchtree(holder)
            self.tree = condense_tree(self.tree_holder)
            contracting_search_tree(self.tree)
            self.soft = soft_clustering(self.tree)
           
            self.hard = hard_clustering(self.soft)
            s = set(self.hard)
            # print(s)
            print_tree(self.tree)
            # print(self.soft)
            
            self.color_dict, self.tangle, set_vals = generate_color_dict(self.hard)
           
            
            new_data = perform_tsne(self.data, perplexity=5)
            self.plot_data(new_data, self.tangle, self.color_dict)
        else:
            print("No data available")  # Debugging

    def plot_data(self, data, tangle, color_dict):
        # Clear the previous plot
        self.figure.clear()
      

        # Plot the new data as a scatter plot with colors based on cuts
        ax = self.figure.add_subplot(111)
       
        for i in range(len(tangle)):
            if tangle[i] in color_dict:
                color_rgb = color_dict[tangle[i]]
                color_str = "#{:02x}{:02x}{:02x}".format(*color_rgb)
                # ax.scatter(data.iloc[i]['PC1'], data.iloc[i]['PC2'], color=color_str)
                ax.scatter(data.iloc[i]['Dim1'], data.iloc[i]['Dim2'], color=color_str)
            else:
                print(f"Warning: Cluster label {tangle[i]} not found in color dictionary.")

        # Update the labels and title
        ax.set_title("2D Plot")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")

        # Update the canvas
        self.canvas.draw()

    def generate_data(self):
        self.upload_data_button.hide()
        self.generate_data_button.hide()
        self.random_centers_button.show()
        self.fixed_centers_button.show()

    def go_back(self):
        self.close()

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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BinaryQuestionnaireWindow()
    window.show()
    sys.exit(app.exec_())
