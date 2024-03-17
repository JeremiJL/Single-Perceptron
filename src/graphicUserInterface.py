import sys
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox
from stylesheets import sheet_dark
from classifier import Classifier


class ParametrizeWindow(QWidget):
    def __init__(self, app):
        super().__init__()
        # Assign
        self.app = app
        # Basic window parameters
        self.setWindowTitle("Perceptron Classifier")
        self.setGeometry(100, 100, 400, 200)
        # Adjust location on screen
        center_loc = app.primaryScreen().geometry().center() - self.rect().center()
        self.move(center_loc)
        # Init widgets
        self.train_data_path_input = QLineEdit(self)
        self.test_data_path_input = QLineEdit(self)
        self.learning_rate_input = QSpinBox(self)
        self.epochs_input = QSpinBox(self)
        self.train_button = QPushButton("Train", self)
        # Create widgets
        self.create_widgets()
        # Prepare reference for second window for classification interface
        self.classify_window = False
        # Prepare reference for classifier
        self.classifier = False

    def create_widgets(self):
        # Specify layouts
        main_layout = QVBoxLayout()
        top_row_layout = QHBoxLayout()
        left_column_layout = QVBoxLayout()
        right_column_layout = QVBoxLayout()

        # Create widgets : labels, text fields, spinners
        left_column_layout.addWidget(QLabel("Train Data Path : "))
        self.train_data_path_input.setText("../data/iris/training.txt")
        left_column_layout.addWidget(self.train_data_path_input)

        left_column_layout.addWidget(QLabel("Test Data Path : "))
        self.test_data_path_input.setText("../data/iris/test.txt")
        left_column_layout.addWidget(self.test_data_path_input)

        right_column_layout.addWidget(QLabel("Learning rate : "))
        self.learning_rate_input.setMaximum(99)
        self.learning_rate_input.setMinimum(1)
        right_column_layout.addWidget(self.learning_rate_input)

        right_column_layout.addWidget(QLabel("Number of epochs :"))
        self.epochs_input.setMinimum(1)
        right_column_layout.addWidget(self.epochs_input)

        top_row_layout.addLayout(left_column_layout)
        top_row_layout.addLayout(right_column_layout)

        main_layout.addLayout(top_row_layout)

        # train button
        self.train_button.setFixedWidth(100)
        self.train_button.clicked.connect(self.train)
        main_layout.addWidget(self.train_button)

        self.setLayout(main_layout)
        self.setStyleSheet(sheet_dark)

    def train(self):
        # Extract input data
        train_data = self.train_data_path_input.text()
        test_data = self.test_data_path_input.text()
        learning_rate = self.learning_rate_input.value()
        epochs = self.epochs_input.value()

        # Initialize classifier and run classification
        self.classifier = Classifier(train_data, test_data, learning_rate, epochs)

        # Open classification window
        self.classify_window = ClassifyWindow(self.app, self.classifier)
        self.classify_window.show()
        self.close()


class ClassifyWindow(QWidget):
    def __init__(self, app, classifier):
        super().__init__()
        # Assign
        self.app = app
        self.classifier = classifier
        # Basic window parameters
        self.setWindowTitle("Perceptron Classifier")
        self.setGeometry(100, 100, 400, 200)
        # Adjust location on screen
        center_loc = app.primaryScreen().geometry().center() - self.rect().center()
        self.move(center_loc)
        # Create widgets
        self.create_widgets()

    def create_widgets(self):
        # Specify layout
        layout = QVBoxLayout()

        info_label = QLabel("Enter data to classify. Separate attribute values with comma.")
        layout.addWidget(info_label)

        classify_data_input = QLineEdit()
        layout.addWidget(classify_data_input)

        classify_button = QPushButton("Classify")
        classify_button.setFixedWidth(100)
        layout.addWidget(classify_button)

        self.setLayout(layout)
        self.setStyleSheet(sheet_dark)


def main():
    app = QApplication(sys.argv)
    # Crete first window for initializing perceptron parameters
    parametrize_window = ParametrizeWindow(app)
    parametrize_window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
