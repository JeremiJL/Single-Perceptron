import sys

from PyQt6.QtCore import QRunnable, QThreadPool, QThread
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox
from copy import deepcopy
from classifier import Classifier, return_key_by_value


class ParametrizeWindow(QWidget):
    def __init__(self, app):
        super().__init__()
        # Assign parent component
        self.app = app
        # Assign thread pool
        self.threadpool = QThreadPool()
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

        # Create label and text field asking for train data file path
        left_column_layout.addWidget(QLabel("Train Data Path : "))
        self.train_data_path_input.setText("../data/iris/training.txt")
        left_column_layout.addWidget(self.train_data_path_input)

        # Create label and text field asking for test data file path
        left_column_layout.addWidget(QLabel("Test Data Path : "))
        self.test_data_path_input.setText("../data/iris/test.txt")
        left_column_layout.addWidget(self.test_data_path_input)

        # Create label and spinner asking for value of learning rate
        right_column_layout.addWidget(QLabel("Learning rate : "))
        self.learning_rate_input.setMaximum(99)
        self.learning_rate_input.setMinimum(1)
        right_column_layout.addWidget(self.learning_rate_input)

        # Create label and text field asking for number of epochs
        right_column_layout.addWidget(QLabel("Number of epochs :"))
        self.epochs_input.setMinimum(1)
        right_column_layout.addWidget(self.epochs_input)

        # Create train button
        self.train_button.setFixedWidth(100)
        self.train_button.clicked.connect(self.train)

        # Bound layouts together
        self.setLayout(main_layout)
        top_row_layout.addLayout(left_column_layout)
        top_row_layout.addLayout(right_column_layout)
        main_layout.addLayout(top_row_layout)
        # Add train button late in order to place him in the bottom
        main_layout.addWidget(self.train_button)

    def train(self):
        # Extract input data
        train_data = self.train_data_path_input.text()
        test_data = self.test_data_path_input.text()
        learning_rate = self.learning_rate_input.value()
        epochs = self.epochs_input.value()

        # Initialize classifier and run classification
        self.classifier = Classifier(train_data, test_data, learning_rate/100, epochs)
        worker = Worker(self.classifier.begin)
        self.threadpool.start(worker)

        # Open classification window
        self.classify_window = ClassifyWindow(self.app, self.threadpool, self.classifier)
        self.classify_window.show()
        self.close()


class ClassifyWindow(QWidget):
    def __init__(self, app, threadpool, classifier):
        super().__init__()
        # Assign parent component
        self.app = app
        # Assign thread pool
        self.threadpool = threadpool
        # Store classifier reference for communication
        self.classifier = classifier
        # Basic window parameters
        self.setWindowTitle("Perceptron Classifier")
        # Adjust location on screen
        center_loc = self.app.primaryScreen().geometry().center() - self.rect().center()
        self.move(center_loc)
        # Store text fields for attributes values input
        self.attributes_line_edits = []
        self.classification_result = QLabel("Classification Result: ...")
        self.accuracy_label = QLabel("Perceptron accuracy on test set after each epoch :\n")
        # Create widgets
        self.create_widgets()

    def create_widgets(self):
        # Specify layout
        main_layout = QHBoxLayout()
        left_column_layout = QVBoxLayout()
        right_column_layout = QVBoxLayout()

        # Create label prompting user to provide data for classification
        info_label = QLabel("Provide observation data for classification :")
        left_column_layout.addWidget(info_label)

        # Provide text fields allowing user to provide attributes values of data
        for le in range(self.classifier.num_of_dimensions):
            attribute_input = QLineEdit(self)
            attribute_input.setFixedWidth(100)
            left_column_layout.addWidget(QLabel("Value of attribute " + str(le) + " : "))
            left_column_layout.addWidget(attribute_input)
            self.attributes_line_edits.append(attribute_input)

        # Create label informing about classification outcome
        self.classification_result.setVisible(False)
        left_column_layout.addWidget(self.classification_result)

        # Crate classify button that triggers the classification method of classifier
        classify_button = QPushButton("Classify")
        classify_button.clicked.connect(self.ask_to_classify)
        left_column_layout.addWidget(classify_button)

        # Inform about accuracy results from epochs
        right_column_layout.addWidget(self.accuracy_label)

        # Create thread that will update accuracy label text, based on ongoing computations
        worker = Worker(self.update_accuracy_results)
        self.threadpool.start(worker)

        # Crate show plot button that creates new window with plot
        classify_button = QPushButton("Show plot")
        right_column_layout.addWidget(classify_button)

        # Bound layouts together
        self.setLayout(main_layout)
        main_layout.addLayout(left_column_layout)
        main_layout.addLayout(right_column_layout)

    def update_accuracy_results(self):
        # Hold until computations are complete inside classfier
        while not self.classifier.finished:
            pass
        # Afterward update label text to inform about accuracy results
        new_text = self.accuracy_label.text()
        results = [""]
        count = 1
        for e in self.classifier.accuracy_list:
            results.append(str(float.__round__(e, 2)))
            if count % 5 == 0:
                results.append("\n")
            count += 1

        self.accuracy_label.setText(str(new_text) + str("  ".join(results)))


    def ask_to_classify(self):
        # # Extract vector data from text fields
        vector = [float(val.text()) for val in self.attributes_line_edits]
        # Use classifier to classify values given by user
        int_val = self.classifier.classify(vector)
        result = return_key_by_value(self.classifier.classes_map, int_val)
        # # Display classification result
        self.classification_result.setVisible(True)
        self.classification_result.setText("Result of classification : " + str(result))


class Worker(QRunnable):

    def __init__(self, function, *args, **kwargs):
        super().__init__()
        self.args = args
        self.function = function

    def run(self):
        self.function()


# Create pyqt application
app = QApplication(sys.argv)
# Crete first window for initializing perceptron parameters
parametrize_window = ParametrizeWindow(app)
parametrize_window.show()
sys.exit(app.exec())



