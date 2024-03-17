import classifier
import matplotlib.pyplot as plt


def testClassifier():
    c = classifier.Classifier(train_file_path, test_file_path, learning_rate, epochs)
    print("should be 0", c.classify_user_input_to_label("4,5"))
    print("should be 0", c.classify_user_input_to_label("1,3"))
    print("should be 1", c.classify_user_input_to_label("60,5"))
    print("should be 1", c.classify_user_input_to_label("30,0"))
    print("should be 1", c.classify_user_input_to_label("20,10"))
    print("should be 0", c.classify_user_input_to_label(" 3,4"))

    print(c.weights)


def draw_bar_plot():
    c = classifier.Classifier(train_file_path, test_file_path, learning_rate, epochs)
    # Creating the data set
    x_values = [x for z, x, y, in [o.values for o in c.observations]]
    y_values = [y for z, x, y, in [o.values for o in c.observations]]

    # Creating figure
    plt.figure(figsize=(10, 5))

    # Plotting observations
    plt.xticks(range(0,200,10))
    plt.yticks(range(0,200,10))
    colors = ['red' if c.classify([x, y]) == 0 else 'green' for x, y in zip(x_values, y_values)]
    plt.scatter(x_values, y_values, c=colors)
    # Plotting decision line
    yL = c.weights[2]
    xL = c.weights[1]
    cL = c.weights[0]
    # xL, yL, cL forms an equation
    # xL
    yVals = ()
    xVals = ()
    plt.plot(xVals, yVals, color='blue', linewidth=2)
    # Set names
    plt.xlabel("Values of x")
    plt.ylabel("Values of y")
    plt.title("Perceptron")

    plt.savefig('../plots/plot.png')
    plt.show()


train_file_path = "../data/example/train.txt"
test_file_path = "../data/example/test.txt"
learning_rate = 0.2
epochs = 10

draw_bar_plot()
testClassifier()

# c = classifier.Classifier(train_file_path, test_file_path, learning_rate, epochs)
# for observation in c.observations:
#     print(observation.values)
