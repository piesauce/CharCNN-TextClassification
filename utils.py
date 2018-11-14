# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 14:47:53 2018

@author: piesauce
"""

import numpy as np
import csv
import re


class Data(object):
    
    """
    Class to load and process training and test data
    """
    
    def __init__(self, path,input_size,vocab,num_classes):
        
        """
        Initialization of a Data object.
        Args:
            path (str): Path to data source
            input_size (int): Length of input
            vocab (str): Character alphabet 
            num_of_classes (int): Number of output classes
        """
        self.path=path
        self.input_size=input_size
        self.vocab=vocab
        self.num_classes=num_classes
        self.vocab_size=len(self.vocab)
        self.char_dict = {}
        for idx, ch in enumerate(self.vocab):
            self.char_dict[ch] = idx + 1
    
    def load(self):
        
        """
        Load training/test data 
        Returns:
            (np.ndarray) Data encoded in 'one-hot' character representation
        """
        
        preprocess()
        data_len = len(self.data)
        start_idx = 0
        end_idx = data_len
        
        batch_texts = self.data[start_idx:end_idx]
        batch_indices=[]
        one_hot = np.eye(self.num_classes, dtype='int64')
        classes = []
        for sent, clas in batch_texts:
            batch_indices.append(self.str_to_idx(sent))
            clas = int(clas) - 1
            classes.append(one_hot[clas])
        return np.asarray(batch_indices, dtype='int64'), np.asarray(classes)
    
    def preprocess(self):
        
        """
        Read and preprocess raw data by removing whitespace etc
        
        """
        
        data = []
        with open(self.path,'r', encoding='utf-8') as f:
            rdr = csv.reader(f, delimiter=',', quotechar='"')
            for row in rdr:
                txt = ""
                for s in row[1:]:
                    txt = txt + " " + re.sub("^\s*(.-)\s*","%1", s).replace("\\n", "\n")
                data.append(txt, int(row[0]))
        self.data=np.array(data)
    
    def str_to_idx(self, s):
        
        """
        Convert a string to character index
        
        Args:
            s (str): String to be converted to indices
        Returns:
            str2idx (np.ndarray): Indices of characters in s
        """
        
        s = s.lower()
        max_len = min(len(s), self.input_size)
        str2idx = np.zeros(self.input_size, dtype='int64')
        for i in range(1,max_len +1):
            char = s[-i]
            if char in self.char_dict:
                str2idx[i-1] = self.char_dict[char]
        return str2idx
            
    
        
        
        
        
        
  
