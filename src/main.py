import sys

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication
import graphicUserInterface as gui


def main():
    # Create pyqt application
    app = QApplication(sys.argv)
    # Applying style sheet
    with open("../stylesheets/black.css") as file:
        stylesheet = file.read()
    app.setStyleSheet(stylesheet)
    # Set application icon
    app.setWindowIcon(QIcon("../images/icon.png"))
    # Crete first window for initializing perceptron parameters
    parametrize_window = gui.ParametrizeWindow(app)
    parametrize_window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
