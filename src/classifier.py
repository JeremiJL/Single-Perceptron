class Classifier:
    def __init__(self, train_file_path, test_file_path, learning_rate, epochs):
        # Assign
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.learning_rate = learning_rate
        self.epochs = epochs
        # Determine classes-labels
        self.classes_map = self.find_classes_labels()


    def find_classes_labels(self):
        classes_map = {1: "A", 0: "B"}
        labels = set()
        # Open file
        with open(self.train_file_path, "r") as file:
            while (len(labels) < 2):
                file.readline()
