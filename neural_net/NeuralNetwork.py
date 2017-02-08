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
    def __init__(self, units_per_hidden_layer, learningRate=0.1):
        #random.seed(1)
        self.nHiddenUnits = units_per_hidden_layer
        self.learningRate = learningRate
        self.weights = []

    def train(self, input, output, t=10000, printError=False):
        if len(self.weights) == 0:
            self.weights = self.init_weights(input, output)
            
        for i in range(t):
            layers, layers_with_bias = self.forward_propagation(input) # forward propagation
            deltas = self.backward_propagation(output, layers) # backward propagation
            self.updateWeights(deltas, layers_with_bias) # updating the weights
            
            if printError and (i % int(t / 20) == 0): # print every 5%
                print("Error:", mean(abs(output - layers[2])).round(3))
                
    def dream(self, input, output, t=10):
        for i in range(t):
            layers, layers_with_bias = self.forward_propagation(input) # forward propagation
            deltas = self.backward_propagation(output, layers) # backward propagation
            input += self.updateInput(deltas, layers_with_bias, input)
        return dream
            

    def forward_propagation(self, input):

        input_with_bias = array([append(i, ([1.0])) for i in input])
        sum1 = dot(input_with_bias, self.weights[0])
        l1 = self.activation_function(sum1)

        l1_with_bias = array([append(i, ([1.0])) for i in l1])
        sum2 = dot(l1_with_bias, self.weights[1])
        l2 = self.activation_function(sum2)

        return [input, l1, l2], [input_with_bias, l1_with_bias, l2]

    def backward_propagation(self, output, layers):

        l2_error = output - layers[2]
        l2_delta = l2_error * self.activation_function(layers[2], True)

        l1_error = dot(l2_delta, self.weights[1][:-1].T) # weights without bias weight
        l1_delta = l1_error * self.activation_function(layers[1], True)

        return [l1_delta, l2_delta]

    def updateWeights(self, deltas, layers):
        for i, delta in enumerate(deltas):
            self.weights[i] += dot(layers[i].T, delta) * self.learningRate
            
    def updateInput(self, deltas, layers, input):
        input_error = dot(deltas[0], self.weights[0][:-1].T)
        input_delta = input_error * self.activation_function(input, True)
        return input_delta

    def init_weights(self, input, output):
        inputSize = len(input[0])
        outputSize= len(output[0])
        units = self.nHiddenUnits
        weights = []
        if self.nHiddenUnits > 0:
            weights.append(self.randomWeights(inputSize + 1, units)) # + 1 for bias
            weights.append(self.randomWeights(units + 1, outputSize)) # + 1 for bias
        else:
            weights.append(self.randomWeights(inputSize + 1, outputSize)) # + 1 for bias
        return weights

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
