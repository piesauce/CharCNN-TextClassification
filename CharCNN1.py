# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 20:59:07 2018

@author: piesauce
"""

from keras.layers import Input, Embedding, Convolution1D, GlobalMaxPooling1D, Dense
from keras.layers import Concatenate, Dropout
from keras.models import Model

class CharCNN1(object):
    
    """
    Implements a character-level convolutional neural network
    """
    
    
    def __init__(self, input_size, vocab_size, embedding_size,
                 num_classes, conv_layers, fc_layers,
                 optim_alg, loss_fnc, drop_prob):
        """
        Initialization for the Character Level CNN model.
        Args:
            input_size (int): Length of input
            vocab_size (int): Number of characters in input vocabulary
            embedding_size (int): Size of embeddings
            num_of_classes (int): Number of output classes
            conv_layers (list[list[int]]): List of Convolution layers
            fc_layers (list[list[int]]): List of Fully Connected layers for model
            optim_alg(str): Optimization algorithm used for training
            loss_fnc(str): Loss function
            drop_prob(str): Dropout probability (percentage of units to drop)
        
        """
        self.input_size = input_size
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.conv_layers = conv_layers
        self.fc_layers = fc_layers
        self.optim_alg = optim_alg
        self.loss_fnc = loss_fnc
        self.drop_prob = drop_prob
        self.model()
    
    def model(self):
        
        """
        Definition of the character level CNN
        
        """
        # Input
        inputs = Input(shape=(self.input_size,), dtype='int64')
        
        # Embedding layer
        X = Embedding(input_dim=self.input_size, output_dim=self.embedding_size, input_size=self.input_size)(inputs)
        
        # Convolutional layers
        conv_layers_output = []
        for filter, filter_size in self.conv_layers:
            conv_output = Convolution1D(filters=filter, kernel_size=filter_size, activation='tanh')(X)
            maxpool_output = GlobalMaxPooling1D()(conv_output)
            conv_layers_output.append(maxpool_output)
        X = Concatenate()(conv_layers_output)
        
        # Fully connected layers
        for layer_units in self.fc_layers:
            X = Dense(units=layer_units, activation='relu', kernel_initializer='glorot_uniform')(X)
            X = Dropout(self.drop_prob)
        
        # Softmax layer    
        pred = Dense(self.num_classes, activation='softmax')(X)
        
        m = Model(inputs=inputs, outputs=pred)
        m.compile(optimizer=self.optim_alg, loss=self.loss_fnc)
        self.model = m
        self.model.summary()
        
    def train(self, X_train, y_train, X_dev, y_dev, epochs, batch_size):
        
          """
          Train the model
        Args:
            X_train (numpy.ndarray): Training set inputs
            y_train (numpy.ndarray): Training set output labels
            X_dev (numpy.ndarray): Dev set inputs
            y_dev (numpy.ndarray): Dev set output labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            
        """
        self.model.fit(X_train, y_train, validation_data=(X_dev, y_dev), epochs=epochs, batch_size=batch_size)
        
    def test(self, X_test, y_test, batch_size):
        
         """
        Test the model
        Args:
            X_test (numpy.ndarray): Test set inputs
            y_test (numpy.ndarray): Test set output labels
            batch_size (int): Batch size
        """
        
        self.model.evalauate(X_test, y_test, batch_size=batch_size)
        
        
   
        
        
        
        
        
        
        
    
