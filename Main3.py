import sys
from PyQt5.QtWidgets import QApplication
from Controller.BinaryQuestionnaireController import BinaryQuestionnaireController
from View.BinaryQuestionnaire import BinaryQuestionnaireWindow

def main():
    # Creating the application instance
    app = QApplication(sys.argv)

    # Creating the view instance
    view = BinaryQuestionnaireWindow(None)

    # Creating the controller instance and connecting it to the view
    controller = BinaryQuestionnaireController(view)

    # Showing the view  
    view.show()

    # Executing the application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()