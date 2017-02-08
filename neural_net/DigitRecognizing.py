from NeuralNetwork import NeuralNetwork
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

# load the dataset (images + labels) (scikit-learn.org/stable/auto_examples/datasets/plot_digits_last_image)
digits = datasets.load_digits()

# each image of a handwritten digit is 8x8 pixels,
# so the input vector has a dimension of 64 (64 features)
num_features = 64

# the maximum value a pixel can have in this dataset is 16
max_value = 16

# turn the pixel values of each image into a feature vector -> input vector of the neural net
X = np.array([list(i.reshape((-1, num_features))[0]) for i in digits.images])

# normalize the input data to values between 0 and 1
X = X / 16

# encode the labels (decimals 0-9) as binary -> output vector of the neural net
# the output of the neural network is always between 0 and 1 (because of the sigmoid function)
Y = []
for t in digits.target:
    if t == 0:
        Y.append([1,0,0,0,0,0,0,0,0,0])
    if t == 1:
        Y.append([0,1,0,0,0,0,0,0,0,0])
    if t == 2:
        Y.append([0,0,1,0,0,0,0,0,0,0])
    if t == 3:
        Y.append([0,0,0,1,0,0,0,0,0,0])
    if t == 4:
        Y.append([0,0,0,0,1,0,0,0,0,0])
    if t == 5:
        Y.append([0,0,0,0,0,1,0,0,0,0])
    if t == 6:
        Y.append([0,0,0,0,0,0,1,0,0,0])
    if t == 7:
        Y.append([0,0,0,0,0,0,0,1,0,0])
    if t == 8:
        Y.append([0,0,0,0,0,0,0,0,1,0])
    if t == 9:
        Y.append([0,0,0,0,0,0,0,0,0,1])
        
# splitting the data into training and testing data.
# (80% for training, 20% for testing)
split_factor = 0.02
testSize = int(len(X) * split_factor)
X_train = X[:-testSize]
X_test = X[-testSize:]
Y_train = Y[:-testSize]
Y_test = Y[-testSize:]
        
# create the neural network with 15 units in the hidden layer
# and a learning rate of 0.015 (approx. 1/64 = 1/num_features)
nn = NeuralNetwork(15, 0.003)

# train the neural network with the test data and 750 iterations
nn.train(X_train, Y_train, t=1000000, printError=True)
"""
# compute and print score for test data
score = nn.score(X_test, Y_test)
print("Score: ", score)

# let the neural network predict the test data
predicted_binary = nn.predict(X_test).round()

# output of the neural network is binary encoded,
# so we have to decode it into the original values
predicted = []
for pred in predicted_binary:
    for i,bit in enumerate(pred):
        if bit == 1:
            predicted.append(i)
            break
    
# show the first 24 test images and the prediction of the neural network
images_and_predictions = list(zip(digits.images[-testSize:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:6]):
    plt.subplot(2, 3, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)
plt.show()
"""

dream_template = np.ones((1,64)) / 2
dream = nn.dream(dream_template, np.array([[1,0,0,0,0,0,0,0,0,0]]), t=100000000)
dream = (dream[0] * 16).round()
print(dream)

plt.subplot(1, 1, 1)
plt.axis('off')
plt.imshow(dream.reshape(8,8) * 16, cmap=plt.cm.gray_r, interpolation='nearest')
plt.title('Dream of a zero')
plt.savefig('dream1.png')
plt.show()
