import sys
from PyQt5.QtWidgets import QApplication

# Add the parent directory of the Controller folder to the Python path
sys.path.append("..")

from View.MainPage import MainPage
import sys

def main():
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
