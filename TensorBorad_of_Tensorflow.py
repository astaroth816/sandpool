# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 12:53:08 2019

@author: andy3
"""

###################################
#
#TensorBoard introduction:Tensorflow of visible frame
#
#
#TensorBoard is the module of Tensorflow package
#
###################################

###Use TensorBoard to classic the word,and we will create the model.###

import keras
from keras import layers
from keras.datasets import imdb
from keras.preprocessing import sequence

#the feature of the words number
max_features=2000

#We only concern 500 words in each article
max_len=500

#Load the data from imdb datasets
import numpy as np
# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# call load_data with allow_pickle implicitly set to true
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# restore np.load for future normal usage
np.load = np_load_old

x_train=sequence.pad_sequences(x_train,maxlen=max_len)

x_test=sequence.pad_sequences(x_test,maxlen=max_len)

######The sturcture of model######
model=keras.models.Sequential()

#The first layers we use the Embedding layer to deal with the data
model.add(layers.Embedding(max_features,
                           128,
                           input_length=max_len,
                           name="embed"))

#The second layers we use the Conv1d layer
model.add(layers.Conv1D(32,
                        7,
                        activation='relu'))

#Next,we add the MaxPooling layer to extract the feature
model.add(layers.MaxPool1D(5))

#Next,we use the Conv1d layer
model.add(layers.Conv1D(32,
                        7,
                        activation='relu'))

#Next,use the GlobalMaxPooling layer 
model.add(layers.GlobalMaxPooling1D())

#Final, we use the dense layer as our output layers
model.add(layers.Dense(1))

#Create the compiler of model
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])

#The sutrucute bulided model
model.summary()

