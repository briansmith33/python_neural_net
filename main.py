import numpy as np
import math
from collections import deque
import os
import colorama
import cv2
import random
from colorama import Fore, Back, Style
from sequence import Sequence
from layer import Dense, Flatten
from tensor import Tensor

colorama.init(autoreset=True)
np.random.seed(0)

X = [[0, 0, 1],
     [0, 1, 1],
     [1, 0, 1],
     [1, 1, 1]]

y = [[0], [1], [1], [0]]

array3d = [[[255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255]],
           [[255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255]],
           [[255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255]],
           [[255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255]],
           [[255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255]]]




def fill_matrix(shape, value=None, current=None, index=0):
    if current is None:
        if value is None:
            value = 0.
        tensor = []
        for i in range(shape[index]):
            tensor.append([])
            fill_matrix(shape, value, tensor[i], index + 1)
    else:
        if index == len(shape) - 1:
            for i in range(shape[index]):
                current.append(value)
            return current
        for i in range(shape[index]):
            current.append([])
            fill_matrix(shape, value, current[i], index+1)
        return current
    return Tensor(tensor, shape, str(type(value))[8:-2])


def flatten(tensor, current=None, vector=None):
    if vector is None:
        vector = []
        for i in range(len(tensor)):
            flatten(tensor, tensor[i], vector)
    else:
        if not isinstance(current[0], list):
            for val in current:
                vector.append(val)
            return vector
        for i in range(len(current)):
            flatten(tensor, current[i], vector)
        return vector
    return vector

'''
IMG_SIZE = 50
training_data = []
for folder in os.listdir('iris training'):
    for image in os.listdir(f'iris training/{folder}'):
        try:
            pixel_values = cv2.imread(f'iris training/{folder}/{image}', cv2.IMREAD_GRAYSCALE)
            resized = cv2.resize(pixel_values, (IMG_SIZE, IMG_SIZE))
            training_data.append([resized, int(folder)])
        except Exception as e:
            pass

random.shuffle(training_data)

X = []
y = []
for features, label in training_data:
    X.append(features)
    y.append(label)



input_shape = X.shape[1:]

layer = Conv3D(input_shape)
layer.forward(X)
print(layer.weights)
print(layer.biases)
print(layer.output)



def main():
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model = Sequence(CCE, sgd)
    model.add(Dense(4, 64, RELU))
    model.add(Dense(64, 8, RELU))
    model.add(Dense(8, 1, STEP))
    model.fit(X, y, 1, 5)


if __name__ == "__main__":
    main()
'''

'''
loss = -np.log(
    np.sum(
        y * np.exp(
            np.dot(
                np.maximum(
                    0,
                    np.dot(
                        X,
                        w1.T
                    ) + b1
                ),
                w2.T,
            ) + b2
        ),
        w3.T
    ) + b3
) / np.sum(
    np.exp(
        np.dot(
            np.maximum(
                0,
                np.dot(
                    np.maximum(
                        0,
                        np.dot(
                            X,
                            w1.T
                        ) + b1
                    ),
                    w2.T
                ) + b2
            ),
            w3.T
        ) + b3
    ),
    axis=1,
    keepdims=True
)
'''

np.random.seed(0)
def create_spiral_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points)+np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    X, y = create_spiral_data(100, 3)
    layer1 = Dense(2, 5, activation='relu')
    layer1.forward(X)
    layer2 = Dense(5, 5, activation='relu')
    layer2.forward(layer1.output)
    layer3 = Dense(5, 3, activation='softmax')
    layer3.forward(layer2.output)
    print(layer1.output)
    print()
    print(layer2.output)
    print()
    print(layer3.output)
    '''
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="brg")
    plt.show()
    '''

    '''
    model = Sequence([
        Dense(2, 5),
        Dense(5, 3)
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    '''
    #model.fit(X, y, epochs=5, batch_size=64)

    # mat = fill_matrix((2, 4, 3))
    # print(flatten(mat))

