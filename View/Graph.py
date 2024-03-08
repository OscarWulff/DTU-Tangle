from PyQt5.QtWidgets import QMainWindow, QPushButton

class GraphWindow(QMainWindow):
    def __init__(self, main_page):
        super().__init__()
        self.main_page = main_page
        self.setWindowTitle("Graph Window")
        self.setGeometry(100, 100, 800, 600)
        self.back_button = QPushButton("Back", self)
        self.back_button.setGeometry(20, 20, 100, 30)
        self.back_button.clicked.connect(self.go_back_to_main_page)

    def go_back_to_main_page(self):
        self.close()  # Close the current window
        # Show the main page again
        self.main_page.show()
