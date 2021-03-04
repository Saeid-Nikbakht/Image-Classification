# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 17:26:20 2021

@author: Saeid
"""

"""In this python script, CNN, FC and LSTM models are utilized to classify the CIFAR-10
   dataset."""

# Importing required modules
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Importing keras features
from tensorflow.keras.layers import Dense, Input, LSTM, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.models import Model

# Importing mnist class from Utils
from Utils.image_classifier_cifar import CIFAR10

# Importing the Fashion Mnist dataset and providing an instance from FashionMnist class by chosing the 
# kind of classification model : "FC", "CNN", and "LSTM"
from tensorflow.keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
cifar10 = CIFAR10(X_train, y_train, X_test, y_test)

# The shape of input data
# N = number of observations/ W = width size of image
# H = height size of image/ C = number of channels in the image
N,W,H,C = X_train.shape 
label_map, num_classes = cifar10.data_desciption()

# Manipulating the input data
X_train, X_test = X_train/255.0, X_test/255.0

# Building the CNN model
# In this function, a CNN model is created.
def create_CNN_model():
    
    i = Input(shape = X_train[0].shape)
    x = Conv2D(16, (3,3), strides = 1, activation = relu, padding = 'same')(i)
    x = MaxPool2D(pool_size = (2,2), strides = 2, padding = 'valid')(x)
    x = Conv2D(32, (3,3), strides = 1, activation = relu, padding = 'same')(x)
    x = MaxPool2D(pool_size = (2,2), strides = 2, padding = 'valid')(x)
    x = Conv2D(64, (3,3), strides = 2, activation = relu, padding = 'same')(x)
    #x = MaxPool2D(pool_size = (2,2), strides = 2, padding = 'valid')(x)
    x = Flatten()(x)
    x = Dropout(rate = 0.5)(x)
    x = Dense(units = 1024, activation = relu)(x)
    x = Dropout(rate = 0.2)(x)
    x = Dense(units = 128, activation = relu)(x)
    x = Dense(units = num_classes, activation = softmax)(x)
    model = Model(i,x)
    
    return model

# In this function, the model is compiled using Adam optimizer, spare_categorical_crossentropy loss function.
# Two modes of processing the data through CNN is presented namely "parallel" and "normal".
def compile_model(learning_rate, loss, metrics):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = create_CNN_model()
        model.compile(optimizer = Adam(learning_rate = learning_rate),
                      loss = loss,
                      metrics = metrics)
    return model

# In this function a learning rate scheduler callback is built.
def schedule(lr, epoch):
    
    if epoch < 5:
        return 0.002
    return 0.001

# compiling the model using sparse categorical crossentropy, and learning rate of 0.001
model = compile_model(learning_rate = 0.001, 
                      loss = "sparse_categorical_crossentropy", 
                      metrics = ["accuracy"])

# Training the model using the batch-size of 128 and 10 epochs of training
# the learning rate scheduler function is also input into the training process as a callback
scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)
model_dict = model.fit(X_train, y_train,
              validation_data = (X_test, y_test),
              epochs = 10,
              callbacks = [scheduler],
              batch_size = 128)

# Plotting the loss and accuracy convergence via loss_accuracy function in the Mnist class
cifar10.loss_accuracy(model_dict, 10)

# Predicting the test data
y_pred = np.argmax(model.predict(X_test),axis = 1)

# Plotting the Confusion matrix using the confusion_matrixx function in the Mnist class
cifar10.confusion_matrixx(y_pred, fontsize = 14,thresh = 50,cmap = plt.cm.Blues)

# Plotting Wrong predictions using wrong_pred class in the Mnist class
cifar10.wrong_pred(y_pred)
