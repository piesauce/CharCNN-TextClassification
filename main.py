# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 14:24:16 2018

@author: piesauce
"""

import tensorflow as tf
import json

from utils import Data
from CharCNN1 import CharCNN1
from CharCNN2 import CharCNN2
from CharTCN import CharTCN

tf.flags.DEFINE_string("m", "CharCNN1", "Select between models CharCNN1, CharCNN2, and CharTCN")
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

if __name__ == "__main__":
    config = json.load(open("config.json"))
    train_data = Data(path=config["data"]["train_path"],
                      input_size=config["data"]["input_size"],
                      vocab=config["data"]["vocab"],
                      num_classes=config["data"]["num_classes"])
    train_data.load()
    X_train, y_train = train_data.load()
    
    dev_data = Data(path=config["data"]["dev_path"],
                      input_size=config["data"]["input_size"],
                      vocab=config["data"]["vocab"],
                      num_classes=config["data"]["num_classes"])
    dev_data.load()
    X_dev, y_dev = dev_data.load()
    
    if FLAGS.m == "CharCNN1":
        m = CharCNN1(input_size=config["data"]["input_size"],
                 vocab_size=config["data"]["vocab_size"],
                 embedding_size=config["data"]["embedding_size"],
                 num_classes=config["data"]["num_classes"],
                 conv_layers=config["cnn1"]["conv_layers"],
                 fc_layers=config["cnn1"]["fc_layers"],
                 optim_alg=config["cnn1"]["optim_alg"],
                 loss_fnc=config["cnn1"]["loss_fnc"],
                 drop_prob=config["cnn1"]["drop_prob"])
    elif FLAGS.m == "cnn2":
        m = CharCNN2(input_size=config["data"]["input_size"],
                 vocab_size=config["data"]["vocab_size"],
                 embedding_size=config["data"]["embedding_size"],
                 num_classes=config["data"]["num_classes"],
                 conv_layers=config["cnn2"]["conv_layers"],
                 fc_layers=config["cnn2"]["fc_layers"],
                 optim_alg=config["cnn2"]["optim_alg"],
                 loss_fnc=config["cnn2"]["loss_fnc"],
                 drop_prob=config["cnn2"]["drop_prob"])
    else:
        m = CharTCN(input_size=config["data"]["input_size"],
                 vocab_size=config["data"]["vocab_size"],
                 embedding_size=config["data"]["embedding_size"],
                 num_classes=config["data"]["num_classes"],
                 conv_layers=config["cnn3"]["conv_layers"],
                 fc_layers=config["cnn3"]["fc_layers"],
                 optim_alg=config["cnn3"]["optim_alg"],
                 loss_fnc=config["cnn3"]["loss_fnc"],
                 drop_prob=config["cnn3"]["drop_prob"],
                 
        
    m.train(X_train=X_train, y_train=y_train, X_dev=X_dev, y_dev=y_dev,
            epochs=config["params"]["epochs"], batch_size=config["params"]["batch_size"])
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
                                  