import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QWidget, QHBoxLayout, QLabel, QLineEdit, QComboBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import matplotlib.pyplot as plt
import time

from Model.GenerateTestData import GenerateDataBinaryQuestionnaire
from Model.DataSetBinaryQuestionnaire import DataSetBinaryQuestionnaire, perform_tsne
from Model.SearchTree import generate_color_dict
from Model.TangleCluster import create_searchtree, condense_tree, contracting_search_tree, soft_clustering, hard_clustering

class BinaryQuestionnaireWindow(QMainWindow):
    def __init__(self, main_page):
        super().__init__()
        self.setWindowTitle("Combined Window")
        self.setGeometry(100, 100, 800, 600)
        self.data = None

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # Add spacer
        layout.addStretch()

        # Add buttons
        button_layout = QHBoxLayout()

        self.upload_data_button = QPushButton("Upload Data as .csv file", self)
        self.upload_data_button.clicked.connect(self.upload_data)
        button_layout.addWidget(self.upload_data_button)

        self.generate_data_button = QPushButton("Generate Random Data", self)
        self.generate_data_button.clicked.connect(self.generate_random_data)
        button_layout.addWidget(self.generate_data_button)

        self.next_button = QPushButton("Next", self)
        self.next_button.clicked.connect(self.display_data_window)
        button_layout.addWidget(self.next_button)

        layout.addLayout(button_layout)

        # Add spacer
        layout.addStretch()

        # Create a Matplotlib figure and canvas
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Add back button
        self.back_button = QPushButton("Back", self)
        self.back_button.clicked.connect(self.go_back)
        layout.addWidget(self.back_button)

        # Additional attributes for GenerateDataBinaryQuestionnaire
        self.num_questions_label = QLabel("Number of Questions:", self)
        layout.addWidget(self.num_questions_label)
        self.num_questions_input = QLineEdit(self)
        layout.addWidget(self.num_questions_input)

        self.num_samples_per_cluster_label = QLabel("Number of Samples per Cluster:", self)
        layout.addWidget(self.num_samples_per_cluster_label)
        self.num_samples_per_cluster_input = QLineEdit(self)
        layout.addWidget(self.num_samples_per_cluster_input)

        self.num_clusters_label = QLabel("Number of Clusters:", self)
        layout.addWidget(self.num_clusters_label)
        self.num_clusters_input = QLineEdit(self)
        layout.addWidget(self.num_clusters_input)

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

        # Retrieve user inputs
        num_questions = int(self.num_questions_input.text())
        num_samples_per_cluster = int(self.num_samples_per_cluster_input.text())
        num_clusters = int(self.num_clusters_input.text())

        # Initialize and generate data
        data_generator = GenerateDataBinaryQuestionnaire(num_questions, num_clusters, num_samples_per_cluster)
        data = data_generator.generate_binary_data_multiple_clusters(num_questions, num_samples_per_cluster, num_clusters)  # Assuming your class has a generate method to populate the data attribute
        self.data = data
        print(self.data)
        

        # self.display_data_window()
        # Display the random data
        # self.plot_data(self.data)

    def display_data_window(self):
        self.figure.clear()
        if self.data is not None:
            tree_holder = create_searchtree(DataSetBinaryQuestionnaire(5).cut_generator_binary(self.data))
            tree = condense_tree(tree_holder)
            contracting_search_tree(tree)
            time.sleep(5)
            soft = soft_clustering(tree)
            hard = hard_clustering(soft)
            color_dict, tangle, set_vals = generate_color_dict(hard)

            new_data = perform_tsne(self.data)
            self.plot_data(new_data, tangle, color_dict)
        else:
            print("No data available")  # Debugging

    def plot_data(self, data, tangle, color_dict):
        # Clear the previous plot
        self.figure.clear()

        # Plot the new data as a scatter plot with colors based on cuts
        ax = self.figure.add_subplot(111)
        for i in range(len(data)):
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

    def go_back(self):
        self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BinaryQuestionnaireWindow()
    window.show()
    sys.exit(app.exec_())
