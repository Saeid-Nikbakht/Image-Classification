# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 17:26:20 2021

@author: Saeid
"""

"""In this python script, CNN, FC and LSTM models are utilized to classify the mnist dataset."""

# Importing required modules
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Importing mnist class from Utils
from Utils.image_classifier import Mnist

# Importing keras features
from tensorflow.keras.layers import Dense, Input, LSTM, Conv2D, Flatten, MaxPool2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.models import Model
   
# Importing the Mnist dataset and providing an instance from Mnist class by chosing
# the kind of classification model : "FC", "CNN", and "LSTM"
from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
mnist = Mnist("FC", X_train, y_train, X_test, y_test)

# Type of Classification model and shape of input data
kind, label_map, num_classes = mnist.data_desciption()
N,T,D = X_train.shape

# Manipulating the input data
X_train, X_test = X_train/255.0, X_test/255.0
# In case CNN model is utilized another dimension should be considered as the number of channels.
if kind == "CNN":
    X_train = np.expand_dims(X_train, (-1))
    X_test = np.expand_dims(X_test, (-1))

# Building the model
# In this function, a Fully Connected model is created.
def create_FC_model():
    
    i = Input(shape = (T,D))
    x = Flatten()(i)
    x = Dense(units = 784, activation = relu)(x)
    x = Dropout(rate = 0.2)(x)
    x = Dense(units = 196, activation = relu)(x)
    x = Dropout(rate = 0.2)(x)
    x = Dense(10, activation = softmax)(x)
    model = Model(i,x)
    return model

# In this function, a CNN model is created.
def create_CNN_model():
    
    i = Input(shape = X_train[0].shape)
    x = Conv2D(16, (7,7), activation = relu, padding = 'same')(i)
    x = MaxPool2D(pool_size = (2,2), strides = 2)(x)
    x = Conv2D(32, (5,5), activation = relu, padding = 'same')(x)
    x = MaxPool2D(pool_size = (2,2), strides = 2)(x)
    x = Conv2D(64, (5,5), activation = relu, padding = 'same')(x)
    x = MaxPool2D(pool_size = (2,2), strides = 2)(x)
    x = Flatten()(x)
    x = Dropout(rate = 0.2)(x)
    x = Dense(units = 512, activation = relu)(x)
    x = Dropout(rate = 0.2)(x)
    x = Dense(units = 128, activation = relu)(x)
    x = Dense(units = num_classes, activation = softmax)(x)
    model = Model(i,x)
    
    return model

# In this function, a LSTM model is created.
def create_LSTM_model():
    
    i = Input(shape = (T,D))
    x = LSTM(128, activation = relu)(i)
    x = Dense(10, activation = softmax)(x)
    model = Model(i,x)
    return model

# In this function, the model is compiled using Adam optimizer, spare_categorical_crossentropy loss function.
# Two modes of processing the data through CNN is presented namely "parallel" and "normal".
def compile_model(learning_rate, loss, metrics):
    
    if kind == "FC":
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = create_FC_model()
            model.compile(optimizer = Adam(learning_rate = learning_rate),
                          loss = loss,
                          metrics = metrics)

    elif kind == "CNN":
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = create_CNN_model()
            model.compile(optimizer = Adam(learning_rate = learning_rate),
                          loss = loss,
                          metrics = metrics)
            
    elif kind == "LSTM":
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = create_LSTM_model()
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
              epochs = 5,
              callbacks = [scheduler],
              batch_size = 128)

# Plotting the loss and accuracy convergence via loss_accuracy function in the Mnist class
mnist.loss_accuracy(model_dict, 5)

# Predicting the test data
y_pred = np.argmax(model.predict(X_test),axis = 1)

# Plotting the Confusion matrix using the confusion_matrixx function in the Mnist class
mnist.confusion_matrixx(y_pred,num_classes = num_classes, fontsize = 14,thresh = 50,cmap = plt.cm.Blues)

# Plotting Wrong predictions using wrong_pred class in the Mnist class
mnist.wrong_pred(y_pred)
