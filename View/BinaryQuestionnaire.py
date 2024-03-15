from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QWidget, QHBoxLayout, QLabel
import sys
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import matplotlib.pyplot as plt

from Model.DataSetBinaryQuestionnaire import perform_pca

class DataDisplayWindow(QMainWindow):
    def __init__(self, main_page, data):
        super().__init__()
        self.main_page = main_page
        self.data = data
        self.setWindowTitle("Data Display Window")
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # Create a Matplotlib figure and canvas
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Plot the data
        new_data = perform_pca(data)
        print("New data:", new_data)  # Debugging
        self.plot_data(new_data)

        # Add back button
        back_button = QPushButton("Back", self)
        back_button.clicked.connect(self.go_back_to_previous_window)
        layout.addWidget(back_button)

    def plot_data(self, data):
        # Clear the previous plot
        self.figure.clear()

        # Plot the new data as a scatter plot
        ax = self.figure.add_subplot(111)
        ax.scatter(data['PC1'], data['PC2'])

        # Update the labels and title
        ax.set_title("2D Plot")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")

        # Update the canvas
        self.canvas.draw()


    def go_back_to_previous_window(self):
        self.close()  # Close the current window
        # Show the previous window again
        self.main_page.show()

class BinaryQuestionnaireWindow(QMainWindow):
    def __init__(self, main_page):
        super().__init__()
        self.main_page = main_page
        self.setWindowTitle("Binary Questionnaire Window")
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
        self.next_button.clicked.connect(self.display_data_window)  # Connect "Next" button to display_data_window method
        button_layout.addWidget(self.next_button)

        layout.addLayout(button_layout)

        # Add spacer
        layout.addStretch()

        # Add back button
        self.back_button = QPushButton("Back", self)
        self.back_button.clicked.connect(self.go_back_to_main_page)
        layout.addWidget(self.back_button)

    def go_back_to_main_page(self):
        self.close()  # Close the current window
        # Show the main page again
        self.main_page.show()

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
        pass  # Placeholder for generating random data

    def display_data_window(self):
        self.close()  # Close the current window
        print("Displaying data window")  # Debugging
        if self.data is not None:
            if not hasattr(self, 'data_display_window'):
                self.data_display_window = DataDisplayWindow(self.main_page, self.data)
            self.data_display_window.show()
        else:
            print("No data available")  # Debugging


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_page = QMainWindow()
    binary_questionnaire_window = BinaryQuestionnaireWindow(main_page)
    main_page.show()
    sys.exit(app.exec_())
