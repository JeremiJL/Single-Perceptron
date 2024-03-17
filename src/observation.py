class Observation:

    def __init__(self, label, attributes_values):
        self.label = label
        self.values = attributes_values
        self.num_of_attributes = len(attributes_values)

    def __str__(self):
        return "Observation :\tLabel : " + self.label + ". Attributes-Vector : " + str(self.values)
