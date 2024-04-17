import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget
from Controller.BinaryQuestionnaireController import BinaryQuestionnaireController
from Controller.FeatureBasedController import FeatureBasedController
from View.BinaryQuestionnaire import BinaryQuestionnaireWindow
from View.FeatureBased import FeatureBasedWindow

from View.FeatureBasedView import FeatureBasedView
from View.Graph import GraphWindow


class MainPage(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Main Page")
        self.setGeometry(100, 100, 800, 600)
        self.BinController = None
        self.FeatController = None

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        layout.setContentsMargins(50, 50, 50, 50)

        self.binary_questionnaire_button = QPushButton("Binary Questionnaire")
        self.feature_based_button = QPushButton("Feature Based")
        self.graph_button = QPushButton("Graph")

        button_height = 60
        self.binary_questionnaire_button.setFixedHeight(button_height)
        self.feature_based_button.setFixedHeight(button_height)
        self.graph_button.setFixedHeight(button_height)

        layout.addWidget(self.binary_questionnaire_button)
        layout.addWidget(self.feature_based_button)
        layout.addWidget(self.graph_button)

        central_widget.setLayout(layout)

        self.binary_questionnaire_button.clicked.connect(self.open_binary_questionnaire_window)
        self.feature_based_button.clicked.connect(self.open_feature_based_window)
        self.graph_button.clicked.connect(self.open_graph_window)

    def open_binary_questionnaire_window(self):
        # Close the MainPage
        self.close()

        # Open the BinaryQuestionnaireWindow
        self.binary_questionnaire_window = BinaryQuestionnaireWindow(self)
        self.BinController = BinaryQuestionnaireController(self.binary_questionnaire_window)
        self.binary_questionnaire_window.show()

    def open_feature_based_window(self):
        # Close the MainPage
        self.close()

        # Open the FeatureBasedWindow
        self.feature_based_window = FeatureBasedView(self)
        self.FeatController = FeatureBasedController(self.feature_based_window)
        self.feature_based_window.show()

    def open_graph_window(self):
    # Close the MainPage
        self.close()

        # Open the GraphWindow and pass a reference to MainPage
        self.graph_window = GraphWindow(self)
        self.graph_window.show()



def create_and_show_main_page():
    app = QApplication(sys.argv)
    main_page = MainPage()
    main_page.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    create_and_show_main_page()
