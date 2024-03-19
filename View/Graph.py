import matplotlib.pyplot as plt
import networkx as nx
from PyQt5.QtWidgets import QMainWindow, QPushButton, QVBoxLayout, QWidget, QHBoxLayout, QFileDialog, QApplication
from View.GraphGenerator import GraphGeneratorWindow

class GraphWindow(QMainWindow):
    def __init__(self, main_page):
        super().__init__()
        self.main_page = main_page
        self.setWindowTitle("Graph Window")
        self.setGeometry(100, 100, 800, 600)

         # Center the window on the screen
        screen_geometry = QApplication.desktop().availableGeometry()
        window_geometry = self.frameGeometry()
        x = (screen_geometry.width() - window_geometry.width()) // 2
        y = (screen_geometry.height() - window_geometry.height()) // 2
        self.move(x, y)

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
        self.generate_data_button.clicked.connect(self.OpenGenerateRandomDataGraph)
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

    def OpenGenerateRandomDataGraph(self):
        # Close the MainPage
        self.close()

        # Open the GraphWindow and pass a reference to MainPage
        self.graph_generator_window = GraphGeneratorWindow(self)
        self.graph_generator_window.show()




