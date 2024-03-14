import matplotlib.pyplot as plt
import networkx as nx
from PyQt5.QtWidgets import QMainWindow, QPushButton, QVBoxLayout, QWidget, QHBoxLayout, QFileDialog
from Model.DataSetGraph import DataSetGraph

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
        # Create an instance of DataSetGraph
        data_set_graph = DataSetGraph(agreement_param=1)
        
        # Generate a random graph
        G = nx.Graph()  # Create an instance of the graph
        G.add_weighted_edges_from([(1, 2, 1), (2, 3, 1), (3, 4, 1), 
                                    (4, 5, 1), (5, 6, 1), (6, 7, 1), (7, 8, 1),
                                    (8, 9, 1), (9, 10, 1), (10, 11, 1), (11, 12, 1),
                                    (12, 13, 1), (13, 14, 1), (14, 15, 1), (15, 16, 1),
                                    (16, 17, 1), (17, 18, 1), (18, 19, 1), (19, 20, 1),
                                    (20, 21, 1), (21, 22, 1), (22, 23, 1), (23, 24, 1)])

        
        # Assign the generated graph to the G attribute
        data_set_graph.G = G
        
        # Initialize the DataSetGraph instance
        data_set_graph.initialize()  # Initialize the graph and generate cuts

        # Sort cuts based on cost
        sorted_cuts = data_set_graph.order_function()


        # Plot the cuts on the graph
        for cut in sorted_cuts:
            print("test")
            pos = nx.spring_layout(G)  # Define the layout of the graph
            nx.draw_networkx_nodes(G, pos, nodelist=cut.A, node_color='r')
            nx.draw_networkx_nodes(G, pos, nodelist=cut.Ac, node_color='b')
            nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
            nx.draw_networkx_labels(G, pos)
        
        plt.title('Graph with Cuts')
        plt.show()