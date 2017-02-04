from NeuralNetwork import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt

# input test gridset; just the first 32 numbers in binary
X2 = np.array([[0,0,0,0,0],
                    [0,0,0,0,1],
                    [0,0,0,1,0],
                    [0,0,0,1,1],
                    [0,0,1,0,0],
                    [0,0,1,0,1],
                    [0,0,1,1,0],
                    [0,0,1,1,1],
                    [0,1,0,0,0],
                    [0,1,0,0,1],
                    [0,1,0,1,0],
                    [0,1,0,1,1],
                    [0,1,1,0,0],
                    [0,1,1,0,1],
                    [0,1,1,1,0],
                    [0,1,1,1,1],
                    [1,0,0,0,0],
                    [1,0,0,0,1],
                    [1,0,0,1,0],
                    [1,0,0,1,1],
                    [1,0,1,0,0],
                    [1,0,1,0,1],
                    [1,0,1,1,0],
                    [1,0,1,1,1],
                    [1,1,0,0,0],
                    [1,1,0,0,1],
                    [1,1,0,1,0],
                    [1,1,0,1,1],
                    [1,1,1,0,0],
                    [1,1,1,0,1],
                    [1,1,1,1,0],
                    [1,1,1,1,1]])

# output test gridset; counting the ones of the input in binary
Y2 = np.array([[0,0,0],
                    [0,0,1],
                    [0,0,1],
                    [0,1,0],
                    [0,0,1],
                    [0,1,0],
                    [0,1,0],
                    [0,1,1],
                    [0,0,1],
                    [0,1,0],
                    [0,1,0],
                    [0,1,1],
                    [0,1,0],
                    [0,1,1],
                    [0,1,1],
                    [1,0,0],
                    [0,0,1],
                    [0,1,0],
                    [0,1,0],
                    [0,1,1],
                    [0,1,0],
                    [0,1,1],
                    [0,1,1],
                    [1,0,0],
                    [0,1,0],
                    [0,1,1],
                    [0,1,1],
                    [1,0,0],
                    [0,1,1],
                    [1,0,0],
                    [1,0,0],
                    [1,0,1]])


delta = 0.2 * (2*np.random.random((30,2))-1).round(2)
X1_0 = np.concatenate((0.8 + delta, [[.05,.5],[.65,.25],[.1,.4],[.2,.2],[.2,.1],[.1,.2],[.1,.1],[.1,.6],[.1,.7],[.4,.9],[.95,.4],[.2,.9],[.9,.1],[.75,.2]])).T
X1_1 = np.concatenate((0.4 + delta, [[.15,.5],[.1,.9],[.2,.8],[.4,.05],[.5,.15],[.3,.7],[.4,.8],[.8,.4],[.9,.4],[.9,.2],[.75,.45]])).T

X = np.concatenate((X1_0.T, X1_1.T))
Y = np.concatenate((np.zeros((len(X1_0.T),1)), np.ones((len(X1_1.T),1))))

"""
split_factor = 0.25
testSize = int(len(X)*split_factor)
X_train = X[:-testSize]
X_test = X[-testSize:]
Y_train = Y[:-testSize]
Y_test = Y[-testSize:]
"""

x = np.arange(0, 1, 0.02)
y = np.arange(0, 1, 0.02)
xx, yy = np.meshgrid(x,y)

grid = []
for i in xx[0]:
    for j in yy:
        grid.append([i,j[0]])

plt.rcParams["figure.figsize"] = [10,8]
plt.scatter(X1_0[0], X1_0[1], c="blue", s=75)
plt.scatter(X1_1[0], X1_1[1], c="red", s=75)
plt.pause(3)

nn = NeuralNetwork(10)
nn.train(X, Y, t=1000)

for _ in range(100):
    nn.trainOnline(X, Y, t=200)
    pred = nn.predict(grid)
    color = []
    for p in pred:
        if p.round() == 1:
            color.append("red")
        else:
            color.append("blue")

    plt.clf()
    plt.scatter([grid[i][0] for i in range(len(grid)) if color[i]=="red"], [grid[i][1] for i in range(len(grid)) if color[i]=="red"], c="red", alpha=.4)
    plt.scatter([grid[i][0] for i in range(len(grid)) if color[i]=="blue"], [grid[i][1] for i in range(len(grid)) if color[i]=="blue"], c="blue", alpha=.4)
    plt.scatter(X1_0[0], X1_0[1], c="blue", s=75)
    plt.scatter(X1_1[0], X1_1[1], c="red", s=75)
    plt.pause(0.05)
