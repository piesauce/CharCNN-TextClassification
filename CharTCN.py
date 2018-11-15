# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 22:33:09 2018

@author: piesauce
"""
from keras.layers import Input, Embedding, Convolution1D, Dense
from keras.layers import BatchNormalization, Activation
from keras.layers import Flatten, Add, Dropout, SpatialDropout1D
from keras.models import Model

class CharTCN(object):
    """
    Implements a character-level TCN (Temporal Convolutional Network). Includes dilated convolutations and residual connections.
    
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
        # Input layer
        inputs = Input(shape=(self.input_size,), dtype='int64')
        
        # Embedding layer
        X = Embedding(input_dim=self.input_size, output_dim=self.embedding_size, input_size=self.input_size)(inputs)
        
        # Temporal Convolution layers
        dilation_rate = 1
        for filter, filter_size in self.conv_layers:
            res_conn = X
            for _ in range(2):
                X = Convolution1D(filters=filter, kernel_size=filter_size, 
                                  padding='causal', dilation_rate=dilation_rate, 
                                  activation='linear')(X)
                X = BatchNormalization()(X)
                X = Activation('relu')(X)
                X = SpatialDropout1D(self.drop_prob)(X)
                dilation_rate *= 2
            res_conn = Convolution1D(filters=filter, kernel_size=1, 
                                  padding='causal', dilation_rate=dilation_rate, 
                                  activation='linear')(res_conn)
            X = Add()([res_conn, X])
        
        # Fully connected layers
        X = Flatten(X)
        for layer_dim in self.fc_layers:
            X = Dense(units=layer_dim, activation='relu', kernel_initializer='glorot_uniform')(X)
            X = Dropout(self.drop_prob)(X)
                
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
        
        



      
