# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 15:35:44 2019

@author: andy3
"""
####################################
#
#
#
#
#
#
#
#
#
#
#
#
####################################
from keras.datasets import mnist

from keras.utils import to_categorical

#we input the data from MNIST datasets.

(train_images,train_labels),(test_images,test_labels)=mnist.load_data()

#Next,we need to deal with the datas as suitable array (match the CNN as inputing data)
#(60000->the number of graph,28->length,28->width,1->RGB number(because our data is black and ))

train_images=train_images.reshape((60000,
                                   28,
                                   28,
                                   1))

train_images=train_images.astype('float32')/255

test_images=test_images.reshape((10000,
                                 28,
                                 28,
                                 1))

test_images=test_images.astype('float32')/255

train_labels=to_categorical(train_labels)

test_labels=to_categorical(test_labels)