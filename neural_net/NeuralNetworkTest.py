from NeuralNetwork import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt

delta = 20 * (2*np.random.random((30,2))-1).round(1)
X1_blue = np.concatenate((80 + delta, [[5,50],[70,20],[65,10],[75,15],[10,40],[20,10],[10,20],[10,10],[10,60],[10,70],[40,90],[90,50],[20,90],[90,10],[75,2]])).T
X1_red = np.concatenate((40 + delta, [[30,15],[15,50],[10,90],[13,80],[20,80],[40,5],[50,15],[30,70],[40,80],[50,70],[80,40],[90,40],[90,20],[80,30],[90,30],[75,45]])).T

X1_blue = X1_blue / 100
X1_red = X1_red / 100

Y1_0 = np.array([[i[0],j[0]] for i,j in zip(np.zeros((len(X1_blue.T),1)), np.ones((len(X1_blue.T),1)))])
Y1_1 = np.array([[i[0],j[0]] for i,j in zip(np.ones((len(X1_red.T),1)), np.zeros((len(X1_red.T),1)))])

X = np.concatenate((X1_blue.T, X1_red.T))
Y = np.concatenate((Y1_0, Y1_1))


"""
split_factor = 0.25
testSize = int(len(X)*split_factor)
X_train = X[:-testSize]
X_test = X[-testSize:]
Y_train = Y[:-testSize]
Y_test = Y[-testSize:]
"""

x = np.arange(0, 1, .01)
y = np.arange(0, 1, .01)
xx, yy = np.meshgrid(x,y)

grid = []
for i in xx[0]:
    for j in yy:
        grid.append([i,j[0]])

plt.rcParams["figure.figsize"] = [10,8]
#plt.scatter(X1_blue[0], X1_blue[1], c="blue", s=75)
#plt.scatter(X1_red[0], X1_red[1], c="red", s=75)
#plt.pause(1)

nn = NeuralNetwork(6)
nn.train(X, Y, t=100)

while(True):
    nn.trainOnline(X, Y)
    pred = nn.predict(grid)
    color = []
    for p in pred:
        if p[0] > p[1]:
            color.append("red")
        else:
            color.append("blue")

    plt.clf()
    plt.scatter([grid[i][0] for i in range(len(grid)) if color[i]=="red"], [grid[i][1] for i in range(len(grid)) if color[i]=="red"], c="red", alpha=.1)
    plt.scatter([grid[i][0] for i in range(len(grid)) if color[i]=="blue"], [grid[i][1] for i in range(len(grid)) if color[i]=="blue"], c="blue", alpha=.1)
    plt.scatter(X1_blue[0], X1_blue[1], c="blue", s=75)
    plt.scatter(X1_red[0], X1_red[1], c="red", s=75)
    plt.pause(0.05)
