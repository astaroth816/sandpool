# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 10:23:01 2019

@author: andy3
"""

###################################
#
#Twins LSTM (Siamese LSTM)
#
#Another more important feature of api is Integration Services . 
#The same weight layer object is reused for each call. 
#Means that you can construct a model of shared branches, 
#that is, they share the same notation and learn these representations for different input materials at the same time.
#
#
#
###################################
#############Twins LSTM############

from keras import layers,Input
from keras.models import Model

#Create the LSTM layer with 32 unions.(The key steps of Twins LSTM)
lstm=layers.LSTM(32)

####Create the "left" inputing branch: the vector which size is 128.###
left_input=Input(shape=(None,128))

#Print the shape of left_input:
print("The shape of left_input:",left_input.shape)

#The output of left_input through the lstm layer
left_output=lstm(left_input)

#Print the shape of left_output:
print("The shape of left_output:",left_output.shape)

###Create the "right" inputing branch: the vector which size is also 128###
right_input=Input(shape=(None,128))

#Print the shape of right_input:
print("The shape of right_input:",right_input.shape)

#The output of right_input through the lstm layer
right_output=lstm(right_input)

#Print the shape of right_output:
print("The shape of right_output:",right_output.shape)

#Print the shape of right_output:
print("The shape of right_output:",right_output.shape)

###Concatenate the left LSTM and right LSTM
merged=layers.concatenate([left_output,right_output],axis=-1)

#Print the shape of merged LSTM:
print( "The shape of merged LSTM:",merged.shape)

#after concatenating the LSTM, we get the final output through the dense layer
predictions=layers.Dense(1,activation='sigmoid')(merged)

#The sturcture of Twins_API
Twins_API=Model([left_input,right_input],predictions)

Twins_API.summary()

