from classifier import Classifier, return_key_by_value


def ask_for_vector(dimensions):
    atr_vector = list()
    for i in range(dimensions -1):
        atr_vector.append(float(input(str(i) + ". attribute value : ")))
    return atr_vector


# Ask for required perceptron paramteres
train_file_path = input("Provide path of training file : ")
test_file_path = input("Provide path of test file : ")
learning_rate = float(input("Enter learning rate : "))
epochs = int(input("Enter number of epochs : "))

# Tmp
train_file_path = "../data/iris/training.txt"
test_file_path = "../data/iris/test.txt"

# Create classifier
classifier = Classifier(train_file_path, test_file_path, learning_rate, epochs)

running = True
while running:
    option = input("a) classify new observation\nb) show accuracy"
                   "\nc) show plot\nd) exit\nOption : ")
    match option:
        case "a":
            print("Provide observation vector :")
            vector = ask_for_vector(classifier.num_of_dimensions)
            numeric_result = classifier.classify(vector)
            label_result = return_key_by_value(classifier.classes_map, numeric_result)
            print("Result of classification :", label_result)
        case "b":
            print("Accuracy :", classifier.get_accuracy_over_time())
        case "c":
            print("not implemented")
        case "d":
            running = False
