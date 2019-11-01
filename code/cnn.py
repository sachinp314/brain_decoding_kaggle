#Imports
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import load_model

def train_cnn(subject_no):
    data = loadmat('../data/train_all/train_subject{}.mat'.format(subject_no))
    X = data['X']
    y = data['y']
    
    # remove first 0.5 seconds
    X = X[:, :, 125:]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    scalers = {}
    for i in range(X_train.shape[1]):
        scalers[i] = StandardScaler()
        X_train[:, i, :] = scalers[i].fit_transform(X_train[:, i, :])
        X_test[:, i, :] = scalers[i].transform(X_test[:, i, :])
        
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    y_train = y_train.reshape(y_train.shape[0])
    y_test = y_test.reshape(y_test.shape[0])
    
    # Initialising the CNN
    classifier = Sequential()

    # Step 1 - Convolution
    classifier.add(Conv2D(32, (10, 10), input_shape = (306, 250, 1), activation = 'relu', padding = 'same'))

    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size = (2, 2), strides = 2))
    
    classifier.add(Conv2D(32, (3, 3), input_shape = (306, 375, 1), activation = 'relu', padding = 'same'))

    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size = (2, 2), strides = 2))

    # Adding a second convolutional layer
    classifier.add(Conv2D(32, (3, 3), activation = 'relu', padding = 'same'))
    classifier.add(MaxPooling2D(pool_size = (2, 2), strides = 2))
    
    classifier.add(Conv2D(32, (3, 3), activation = 'relu', padding = 'same'))
    classifier.add(MaxPooling2D(pool_size = (2, 2), strides = 2))
    
    classifier.add(Conv2D(16, (3, 3), activation = 'relu', padding = 'same'))
    classifier.add(MaxPooling2D(pool_size = (2, 2), strides = 2))

    # Step 3 - Flattening
    classifier.add(Flatten())

    # Step 4 - Full connection
    classifier.add(Dense(units = 32, activation = 'relu'))
    classifier.add(Dense(units = 1, activation = 'softmax'))

    # Compiling the CNN
    classifier.compile(optimizer= 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    print(classifier.summary())
    classifier.fit(x=X_train, y=y_train, steps_per_epoch = 475, epochs = 15, validation_data = (X_test, y_test), validation_steps = 119)
    classifier.save('CNN.h5')
    
    
train_cnn('01')

