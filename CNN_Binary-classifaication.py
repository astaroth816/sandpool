# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 20:12:49 2019

@author: andy3
"""

#######################################
#
#CNN for MNIST 
#
#
#
#
######################################
#
#
#_________________________________________________________________
#
#_________________________________________________________________
#Layer (type)                 Output Shape              Param #   
#=================================================================
#conv2d_1 (Conv2D)           (None, 26, 26, 32)        320       
#_________________________________________________________________
#max_pooling2d_1 (MaxPooling (None, 13, 13, 32)        0         
#_________________________________________________________________
#conv2d_2 (Conv2D)           (None, 11, 11, 64)        18496     
#_________________________________________________________________
#max_pooling2d_2 (MaxPooling (None, 5, 5, 64)          0         
#_________________________________________________________________
#conv2d_3 (Conv2D)           (None, 3, 3, 64)          36928     
#_________________________________________________________________
#flatten_1 (Flatten)         (None, 576)               0         
#_________________________________________________________________
#dense_1 (Dense)             (None, 64)                18464     
#_________________________________________________________________
#dense_2 (Dense)             (None, 10)                330       
#=================================================================
#Total params: 74,538
#Trainable params: 74,538
#Non-trainable params: 0
#_________________________________________________________________
#      
#      
#        
######################################

from keras.datasets import mnist

from keras.utils import to_categorical

#we input the data from MNIST datasets.

(train_images,train_labels),(test_images,test_labels)=mnist.load_data()

#Next,we need to deal with the datas as suitable array (match the CNN as inputing data)
#For fitting the CNN, we need the reshape the origin data(ex:train_images.shape=(60000,28,28)->(60000,28,28,1))
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

##Next, we can build our CNN(conv2D and MaxPooling)
from keras import layers

from keras import models

#Create our NN sturcture.

CNN=models.Sequential()

#1 Conv2D layer as input layer with 32 neural network union,(3,3) means the size of "window"

CNN.add(layers.Conv2D(32,
                        (3,3),
                        activation='relu',
                        input_shape=(28,28,1)))

#1 MaxPooling2D layer, this layey can cut the dimension to half of inputing dimension.MaxPooling2D can
#concentrate the feature of inputing data.

CNN.add(layers.MaxPool2D((2,2)))

#2 Conv2D layer with 64 neural network union and window (3,3)

CNN.add(layers.Conv2D(64,
                        (3,3),
                        activation='relu'))

#2 MaxPooling2D layer

CNN.add(layers.MaxPool2D((2,2)))

#3 Conv2D layer with 64 neural network union,window (3,3)

CNN.add(layers.Conv2D(64,
                        (3,3),
                        activation='relu'))

CNN.add(layers.MaxPool2D((2,2)))
\

#The last Conv2D output 3D tensor(shape=(3,3,64)).Next, we need to trans it to next NN.

#We flat the (28,28)->(784,).Because we need to put in DNN layers,so we need to flat the inputing tensor.

CNN.add(layers.Flatten())

#1D layers with 64 neural network union.

CNN.add(layers.Dense(64,
                       activation='relu'))

#the output layers(softmax) with 10 NN union.

CNN.add(layers.Dense(10,
                       activation='softmax'))

#the strusture of CNN layers.

CNN.summary()

#After we create the CNN,we take it to train our MNIST datasets with CNN.

#We use optimizer as rmsprop and loss fun as categorical_crossentropy.

CNN.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


#Train our CNN with MNIST datasets, with 20 times learning.

history=CNN.fit(train_images,
                  train_labels,
                  epochs=20,
                  batch_size=64)

#After training our CNN, we use the test_images to test CNN.

test_loss,test_acc=CNN.evaluate(test_images,
                                  test_labels)

#Print the accuracy and loss quantity of training

import matplotlib.pyplot as plt

acc=history.history['acc']

loss=history.history['loss']

epochs=range(1,
             len(acc)+1)

plt.plot(epochs,acc,
         'bo',
         label='Traning accuracy')

plt.title('The accuracy of training ')

plt.legend()

plt.figure()

plt.plot(epochs,loss,
         'bo',
         label='Training loss quantity')

plt.title('Training loss')

plt.legend()

plt.show()

print('The accuracy of test_images on CNN:')

print(test_acc)

print('The loss of test_images on CNN:')

print(test_loss)
