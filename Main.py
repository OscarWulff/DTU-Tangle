import sys
from PyQt5.QtWidgets import QApplication
from View.MainPage import MainPage
import warnings

def main(): 
    warnings.filterwarnings("ignore", message="Graph is not fully connected")

    
    # Create a QApplication instance
    app = QApplication([])

    # Create an instance of the MainPage class
    main_page = MainPage()

    # Show the main page GUI
    main_page.show()
 
    # Start the application event loop
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
