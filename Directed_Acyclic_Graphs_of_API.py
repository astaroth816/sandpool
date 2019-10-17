# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 14:01:12 2019

@author: andy3
"""

#################################
#
#
#API with directed acyclic graphs
#
#The purpose:Inception V3
#################################

#########Use API to create the Inception module##########

from keras import layers,Input,Model

#Define the input_tensor as x with 4D tensor
x=Input(batch_shape=(1000,28,28,256))

#Use the Conv2D to be a branch_a
branch_a=layers.Conv2D(64,
                       1,
                       activation='relu',
                       strides=2
                       )(x)

print('The shape of branch_a:',branch_a.shape)

#Use the Conv2D and Conv2D to be a branch_b
branch_b=layers.Conv2D(128,
                       1,
                       activation='relu'
                       )(x)

branch_b=layers.Conv2D(128,
                       3,
                       activation='relu',
                       strides=2,
                       padding='same'
                       )(branch_b)

print('The shape of branch_b:',branch_b.shape)

#Use the AveragePooling2D and Conv2d to be a branch_c
branch_c=layers.AveragePooling2D(3,
                                 strides=2,
                                 padding='same'
                                 )(x)

branch_c=layers.Conv2D(128,
                       3,
                       activation='relu',
                       padding='same'
                       )(branch_c)

print('The shape of branch_c:',branch_c.shape)

#Use the triple Conv2D to be a branch_d
branch_d=layers.Conv2D(128,
                       1,
                       activation='relu'
                       )(x)

branch_d=layers.Conv2D(128,
                       3,
                       activation='relu',
                       padding='same'
                       )(branch_d)

branch_d=layers.Conv2D(128,
                       3,
                       activation='relu',
                       strides=2,
                       padding='same'
                       )(branch_d)

print('The shape of branch_d:',branch_d.shape)

#Concatenate the total branchs and get the output of module
output=layers.concatenate([branch_a,branch_b,branch_c,branch_d],axis=-1)

print('The shape of output:',output.shape)

API_model=Model(x,output)

API_model.summary()

from keras.utils import plot_model

plot_model(API_model,show_shapes=True,to_file='model.png')