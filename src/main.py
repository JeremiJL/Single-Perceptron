import classifier
import matplotlib.pyplot as plt


def testClassifier():
    c = classifier.Classifier(train_file_path, test_file_path, learning_rate, epochs)
    c.begin()

    print("Weights :", c.weights)
    print("Num of epochs :", c.epochs)
    print("accuracy len :",len(c.accuracy_list))


train_file_path = "../data/example/train.txt"
test_file_path = "../data/example/test.txt"
learning_rate = 0.2
epochs = 5

testClassifier()

# def draw_bar_plot():
#     c = classifier.Classifier(train_file_path, test_file_path, learning_rate, epochs)
#     # Creating the data set
#     x_values = [x for z, x, y, in [o.values for o in c.observations]]
#     y_values = [y for z, x, y, in [o.values for o in c.observations]]
#
#     # Creating figure
#     plt.figure(figsize=(10, 5))
#
#     # Plotting observations
#     plt.xticks(range(0,200,10))
#     plt.yticks(range(0,200,10))
#     colors = ['red' if c.classify([x, y]) == 0 else 'green' for x, y in zip(x_values, y_values)]
#     plt.scatter(x_values, y_values, c=colors)
#     # Plotting decision line
#     aY = c.weights[2]
#     bX = c.weights[1]
#     const = c.weights[0]
#     # xL, yL, cL forms an equation
#     # aY * Y + bX * X + const = 0
#     # for X = 0  : aY * Y = -const
#     # so Y = -const/aY for X = 0
#     # for X = 10 : aY * Y + bX * 10 + const = 0
#     # so Y = -(const + bX*10))/aY
#     # print(str(0) + " and " + str(10*bX))
#     # print(str(-const/aY) + " and " + str(-(const + bX*10)/aY))
#     print("Weights " + str(c.weights))
#
#     yVals = [-const/aY,-(const + bX*10)/aY]
#     xVals = [0,10*bX]
#     # yVals = 0
#     # xVals = 0
#     plt.plot(xVals, yVals, color='blue', linewidth=2)
#     # Set names
#     plt.xlabel("Values of x")
#     plt.ylabel("Values of y")
#     plt.title("Perceptron")
#
#     plt.savefig('../plots/plot.png')
#     plt.show()
#
#
# draw_bar_plot()
