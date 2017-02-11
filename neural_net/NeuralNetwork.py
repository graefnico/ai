"""

Copyright (c) 2017 Nico Gräf (nicograef.de, github.com/graefnico)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

from numpy import *

class NeuralNetwork:
    def __init__(self, hidden_layer, learningRate=0.1):
        #random.seed(1)
        self.hiddenLayer = hidden_layer
        self.learningRate = learningRate
        self.weights = []

    def train(self, input, output, t=1000, printError=False):
        if len(self.weights) == 0:
            self.init_weights(input, output)

        for i in range(t):
            layers, layers_with_bias = self.forward_propagation(input) # forward propagation
            deltas = self.backward_propagation(output, layers) # backward propagation
            self.updateWeights(deltas, layers_with_bias) # updating the weights

            if printError and (i % int(t / 20) == 0): # print every 5%
                print(str(int(i / (t / 100))) + "%", end=" ")
                print("Error:", mean(abs(output - layers[-1])).round(4))

    def dream(self, input, output, t=10, printPercentage=False):
        for i in range(t):
            layers, layers_with_bias = self.forward_propagation(input) # forward propagation
            deltas = self.backward_propagation(output, layers) # backward propagation
            input += self.updateInput(deltas, layers_with_bias, input)
            if printPercentage and (i % int(t / 20) == 0): # print every 5%
                print(str(int(i / (t / 100))) + "%")
        return input

    def setLearningRate(self, newLearningRate):
        self.learningRate = newLearningRate;

    def forward_propagation(self, input):
        layers = [input]
        layers_with_bias = [array([append(i, ([1.0])) for i in layers[0]])]

        for i, weights in enumerate(self.weights):
            Sum = dot(layers_with_bias[i], self.weights[i])
            layers.append(self.activation_function(Sum))
            layers_with_bias.append(array([append(i, ([1.0])) for i in layers[i+1]]))

        return layers, layers_with_bias

    def backward_propagation(self, output, layers):
        deltas = []
        for i, weights in enumerate(self.weights):
            if i == 0:
                error = output - layers[-1]
            else:
                error = dot(deltas[i-1], self.weights[-i][:-1].T) # weights without bias weight
            deltas.append(error  * self.activation_function(layers[-1-i], True))

        return deltas

    def updateWeights(self, deltas, layers):
        for i, delta in enumerate(reversed(deltas)):
            self.weights[i] += dot(layers[i].T, delta) * self.learningRate

    def updateInput(self, deltas, layers, input):
        input_error = dot(deltas[-1], self.weights[0][:-1].T)
        input_delta = input_error * self.activation_function(input, True)
        return input_delta

    def init_weights(self, input, output):
        inputSize = len(input[0])
        outputSize= len(output[0])
        for i, units in enumerate(self.hiddenLayer):
            if i == 0:
                self.weights.append(self.randomWeights(inputSize + 1, units)) # + 1 for bias
            else:
                self.weights.append(self.randomWeights(self.hiddenLayer[i-1] + 1, units)) # + 1 for bias
        self.weights.append(self.randomWeights(self.hiddenLayer[-1] + 1, outputSize)) # + 1 for bias

    def randomWeights(self, size1, size2):
        return 2 * random.random((size1, size2)) - 1

    def predict(self, input):
        layers, layers_with_bias = self.forward_propagation(input) # forward propagation
        return layers[-1]

    def activation_function(self, x, deriv=False):
        return self.sigmoid(x, deriv)

    def sigmoid(self, x, deriv=False):
        if deriv == True:
            return x * (1 - x)
        return 1 / (1 + exp(-x))

    def score(self, input, output):
        predicted_output = self.predict(input).round()
        score = 0
        for i, prediction in enumerate(predicted_output):
            if array_equal(prediction, output[i]):
                score += 1
        return score / len(input)
