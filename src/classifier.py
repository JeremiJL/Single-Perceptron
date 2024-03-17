from observation import Observation


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
        self.num_of_dimensions = self.observations[0].num_of_attributes - 1;
        self.weights = self.initialize_starter_weights()
        # Test
        self.test()

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

        # Collect all observations from train data file into tuple
        return tuple(observations_list)

    def initialize_starter_weights(self):
        treshold = 0.05
        weights = [0.1 for _ in range(self.num_of_dimensions)]
        weights.insert(0, treshold)
        return weights

    def test(self):
        print(self.classes_map)
        print(self.weights)

        for o in self.observations:
            print(str(o))
