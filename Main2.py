import sys
from PyQt5.QtWidgets import QApplication
from Controller.FeatureBasedController import FeatureBasedController
from View.FeatureBasedView import FeatureBasedView

def main():
    # Creating the application instance
    app = QApplication(sys.argv)

    # Creating the view instance
    view = FeatureBasedView(None)

    # Creating the controller instance and connecting it to the view
    controller = FeatureBasedController(view)

    # Showing the view  
    view.show()

    # Executing the application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()