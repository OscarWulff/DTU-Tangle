import matplotlib.pyplot as plt
import networkx as nx
from PyQt5.QtWidgets import QMainWindow, QPushButton, QVBoxLayout, QWidget, QHBoxLayout, QFileDialog
from Model.DataSetGraph import DataSetGraph
from Model.GenerateTestData import GenerateRandomGraph

class GraphWindow(QMainWindow):
    def __init__(self, main_page):
        super().__init__()
        self.main_page = main_page
        self.setWindowTitle("Graph Window")
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()

        # Add spacer
        layout.addStretch()

        # Add buttons
        button_layout = QHBoxLayout()
        
        self.upload_data_button = QPushButton("Upload Data as nx Graph", self)
        self.upload_data_button.clicked.connect(self.upload_data)
        button_layout.addWidget(self.upload_data_button)
        
        self.generate_data_button = QPushButton("Generate Random Data", self)
        self.generate_data_button.clicked.connect(self.generate_random_data)
        button_layout.addWidget(self.generate_data_button)

        layout.addLayout(button_layout)

        # Add spacer
        layout.addStretch()

        # Add back button
        self.back_button = QPushButton("Back", self)
        self.back_button.clicked.connect(self.go_back_to_main_page)
        layout.addWidget(self.back_button)

        central_widget.setLayout(layout)

    def go_back_to_main_page(self):
        self.close()  # Close the current window
        # Show the main page again
        self.main_page.show()

    def upload_data(self):
        file_dialog = QFileDialog()
        file_dialog.exec_()

    def generate_random_data(self):
        # Generate a random graph
        graph_generator = GenerateRandomGraph(num_of_nodes=20, num_of_clusters=3)
        random_graph, ground_truth = graph_generator.generate_random_graph()

        # Create an instance of DataSetGraph
        data_set_graph = DataSetGraph(agreement_param=1)

        # Assign the generated graph to the G attribute
        data_set_graph.G = random_graph

        # Initialize the DataSetGraph instance
        data_set_graph.initialize()  # Initialize the graph and generate cuts

        # Sort cuts based on cost
        sorted_cuts = data_set_graph.order_function()

        # Visualize the graph with ground truth clusters
        graph_generator.visualize_graph(random_graph, ground_truth=ground_truth, visualize_ground_truth=True)

        plt.title('Graph with Cuts')
        plt.show()

