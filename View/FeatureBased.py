from PyQt5.QtWidgets import QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QWidget, QHBoxLayout, QLabel

class FeatureBasedWindow(QMainWindow):
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

        # Add buttons
        button_layout = QHBoxLayout()
        
        self.upload_data_button = QPushButton("Upload Data as .csv", self)
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
        if file_dialog.exec_():
            selected_file = file_dialog.selectedFiles()[0]  # Get the path of the selected file
            with open(selected_file, 'r') as file:
                data = file.read()  # Read the contents of the file
                # Process the data as needed
                print(data)

    def generate_random_data(self):
        pass  # Placeholder for generating random data