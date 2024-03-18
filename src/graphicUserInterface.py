import sys
import matplotlib.pyplot as plt
from PyQt6.QtCore import QRunnable, QThreadPool, QThread
from PyQt6.QtWidgets import QApplication, QDial, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox, \
    QGridLayout
from copy import deepcopy

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

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
        self.setGeometry(100, 100, 620, 200)
        # Adjust location on screen
        center_loc = app.primaryScreen().geometry().center() - self.rect().center()
        self.move(center_loc)
        # Init widgets
        self.learning_rate_info = QLabel("Learning rate : 0.01")
        self.epochs_info = QLabel("Number of epochs : 1")
        self.train_data_path_input = QLineEdit(self)
        self.test_data_path_input = QLineEdit(self)
        self.learning_rate_input = QDial(self)
        self.epochs_input = QDial(self)
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
        self.train_data_path_input.setText("../data/example/train.txt")
        left_column_layout.addWidget(self.train_data_path_input)

        # Create label and text field asking for test data file path
        left_column_layout.addWidget(QLabel("Test Data Path : "))
        self.test_data_path_input.setText("../data/example/test.txt")
        left_column_layout.addWidget(self.test_data_path_input)

        # Create label and dial asking for value of learning rate
        self.learning_rate_input.setMaximum(99)
        self.learning_rate_input.setMinimum(1)
        self.learning_rate_info.setFixedWidth(150)
        self.learning_rate_input.valueChanged.connect(self.update_learning_rate_label)
        right_column_layout.addWidget(self.learning_rate_info)
        right_column_layout.addWidget(self.learning_rate_input)

        # Create label and dial asking for number of epochs
        self.epochs_input.setMinimum(1)
        self.epochs_input.setMaximum(200)
        self.epochs_info.setFixedWidth(150)
        self.epochs_input.valueChanged.connect(self.update_epochs_label)
        right_column_layout.addWidget(self.epochs_info)
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

    def update_learning_rate_label(self):
        self.learning_rate_info.setText("Learning rate : " + str(self.learning_rate_input.value() / 100))

    def update_epochs_label(self):
        self.epochs_info.setText("Number of epochs : " + str(self.epochs_input.value()))

    def train(self):
        # Extract input data
        train_data = self.train_data_path_input.text()
        test_data = self.test_data_path_input.text()
        learning_rate = self.learning_rate_input.value()
        epochs = self.epochs_input.value()

        # Initialize classifier and run classification
        self.classifier = Classifier(train_data, test_data, learning_rate / 100, epochs)
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
        # Store reference to plot window
        self.plot_window = False
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
        plot_button = QPushButton("Show plot")
        # Disable this button if number of dimensions makes it
        # impossible to draw perceptron on a plane
        if self.classifier.num_of_dimensions != 2:
            plot_button.setDisabled(True)
        plot_button.clicked.connect(self.show_plot_in_new_window)
        right_column_layout.addWidget(plot_button)

        # Bound layouts together
        self.setLayout(main_layout)
        main_layout.addLayout(left_column_layout)
        main_layout.addLayout(right_column_layout)

    def show_plot_in_new_window(self):
        self.plot_window = PlotWindow(self.app, self.classifier)
        self.plot_window.show()

    def update_accuracy_results(self):
        # Hold until computations are complete inside classifier
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


class PlotWindow(QWidget):
    def __init__(self, app, classifier):
        super().__init__()
        # Assign parent component
        self.app = app
        # Assign classifier
        self.classifier = classifier
        # Basic window parameters
        self.setWindowTitle("Perceptron Classifier")
        self.setGeometry(100, 100, 600, 600)
        # Store reference for figure - canvas
        self.my_fig = False
        # Create widgets
        self.create_widgets()

    def create_widgets(self):
        # Abbreviation for classifier reference
        c = self.classifier
        # Specify layouts
        main_layout = QVBoxLayout()
        # Create figure
        self.my_fig = Canvas(self, 10, 5)

        # Creating the data set
        x_values = [x for z, x, y, in [o.values for o in c.observations]]
        y_values = [y for z, x, y, in [o.values for o in c.observations]]

        # Adjusting appearance
        colors = ['red' if c.classify([x, y]) == 0 else 'green' for x, y in zip(x_values, y_values)]

        # Plotting observations
        self.my_fig.my_plot.scatter(x_values, y_values, c=colors)

        # Setting names
        self.my_fig.my_plot.set_title("Perceptron")

        main_layout.addWidget(self.my_fig)

        # Create button to save plot as image
        button_save = QPushButton("Save as png")
        button_save.clicked.connect(self.save_plot_as_image)
        main_layout.addWidget(button_save)

        # Bound window with layout
        self.setLayout(main_layout)

    def save_plot_as_image(self):
        self.my_fig.fig.savefig("../plots/plot.png")


class Canvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=10, height=5, dpi=100):
        self.fig = plt.Figure(figsize=(width, height), dpi=dpi)
        self.my_plot = self.fig.add_subplot()
        super(Canvas, self).__init__(self.fig)


class Worker(QRunnable):

    def __init__(self, function, *args, **kwargs):
        super().__init__()
        self.args = args
        self.function = function

    def run(self):
        self.function()


# Create pyqt application
app = QApplication(sys.argv)
# Applying style sheet
style_sheet = """
/* Global styles */
QWidget {
    background-color: #f0f0f0; /* Light gray background */
    color: #333; /* Dark gray text */
    font-family: Arial, sans-serif; /* Default font */
}

/* Push button */
QPushButton {
    background-color: #4CAF50; /* Green button */
    color: white; /* White text */
    border: none; /* No border */
    border-radius: 4px; /* Rounded corners */
    padding: 8px 16px; /* Padding */
    font-size: 14px; /* Font size */
}

QPushButton:hover {
    background-color: #45a049; /* Darker green on hover */
}

/* Line edit */
QLineEdit {
    background-color: white; /* White background */
    border: 1px solid #ccc; /* Gray border */
    border-radius: 4px; /* Rounded corners */
    padding: 4px; /* Padding */
    font-size: 16px; /* Font size */
}

/* Spin box */
QSpinBox {
    background-color: white; /* White background */
    border: 1px solid #ccc; /* Gray border */
    border-radius: 4px; /* Rounded corners */
    padding: 4px; /* Padding */
    font-size: 14px; /* Font size */
}

/* Label */
QLabel {
    color: #666; /* Gray text */
    font-size: 14px; /* Font size */
}
"""
app.setStyleSheet(style_sheet)
# Crete first window for initializing perceptron parameters
parametrize_window = ParametrizeWindow(app)
parametrize_window.show()
sys.exit(app.exec())
