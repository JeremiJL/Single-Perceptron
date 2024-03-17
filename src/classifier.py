from observation import Observation
from random import shuffle, random


def return_key_by_value(dictionary, searched_value):
    for key, value in dictionary.items():
        if value == searched_value:
            return key


def dot_product(vector, weights):
    result = 0
    for x, y in zip(vector, weights):
        result += x * y
    return result


def heavy_side(y):
    if y > 0:
        return 1
    else:
        return 0


def delta_update_weights(direction, learning_rate, vector, weights):
    for v, i in zip(vector, range(len(vector))):
        weights[i] += direction * learning_rate * v


class Classifier:
    def __init__(self, train_file_path, test_file_path, learning_rate, epochs):
        # Assign
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.learning_rate = learning_rate
        self.epochs = epochs
        # Determine classes-labels map them with numeric values for easier computation
        self.classes_map = self.find_classes_labels()
        # Extract all the observations from train file
        self.observations = self.extract_observations()
        # Set up for learning
        self.num_of_dimensions = self.observations[0].num_of_attributes - 1
        self.weights = self.initialize_starter_weights()
        # Train
        self.train()

    def train(self):
        # Iterate 'epoch' number of times
        for epoch in range(0, self.epochs):
            shuffle(self.observations)
            # Before each iteration shuffle the order of observations
            for observation in self.observations:
                vector = observation.values
                y = self.classify(vector)
                # y - computed result of classification, d - desired result of classification
                d = self.classes_map[observation.label]
                # Check if the result corresponds with the observation class
                # If it doest match apply delta rule
                if y != d:
                    direction = d - y
                    delta_update_weights(direction, self.learning_rate, vector, self.weights)

    def classify(self, vector):
        # add constant value at index 0 if needed
        if len(vector) != self.num_of_dimensions + 1:
            vector.insert(0,1)
        # Calculate the dot product between vector and weights
        # (treshold included in formula as a constant element of vector and weights)
        y = dot_product(vector, self.weights)
        # Perform the heavy side function
        return heavy_side(y)

    def find_classes_labels(self):
        classes_map = dict()
        labels = set()
        # Open file
        with open(self.train_file_path, "r") as file:
            while len(labels) < 2:
                line = file.readline()
                labels.add(line.split(",")[-1].strip("\n"))

        # Bound labels with numeric values
        classes_map[labels.pop()] = 0
        classes_map[labels.pop()] = 1

        return classes_map

    def extract_observations(self):
        # Store observations in a list
        observations_list = []

        # Open file with train data
        with (open(self.train_file_path, "r") as file):
            for line in file:
                line = line.split(",")
                # Create observation
                label = line[-1].strip("\n")
                # Attributes - vector consists of constant 1 at first index and attributes
                # values extracted from file on consequent indices
                attributes = [float(val) for val in line[0:-1]]
                attributes.insert(0, 1)
                observation = Observation(label, attributes)
                # Add new observation
                observations_list.append(observation)

        # Collect all observations from train data file into list
        return observations_list

    def initialize_starter_weights(self):
        # treshold - also known as bias
        bias = 2
        weights = [(random() + 1) / 10 for _ in range(self.num_of_dimensions)]
        weights.insert(0, bias)
        return weights

    def classify_user_input_to_label(self, csv_input):
        # Convert user string in csv format into list
        csv_input = csv_input.split(",")
        # Convert it into float list
        vector = [float(e) for e in csv_input]
        # Result is a classification mapped with corresponding label
        result = return_key_by_value(self.classes_map, self.classify(vector))
        return result

    def test(self):
        print(self.classes_map)
        print(self.weights)

        for o in self.observations:
            print(str(o))
