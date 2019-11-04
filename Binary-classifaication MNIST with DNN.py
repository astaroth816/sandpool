# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 09:57:37 2019

@author: andy3
"""

###################################################
#
#
#The first nerual netwrok- DNN_for_MNIST
#
#
###################################################


#Download the MNIST datasets from keras

from keras.datasets import mnist

#use the 'mnist.load_data()'to get the mnist datasets and save it as tuple.

#(train_images,train_labels) we call it 'training sets' and (test_images,test_labels) 'testing sets'

(train_images,train_labels),(test_images,test_labels)=mnist.load_data()

#Prepare the inputing data,and let them put in tensor for nerual network

#reshape from uint8(3-D tensor) to float32(2-D tensor) and /255 [0,255]->[0,1]

train_images=train_images.reshape((60000,
                                   28*28))

train_images=train_images.astype('float32')/255

test_images=test_images.reshape((10000,
                                 28*28))

test_images=test_images.astype('float32')/255


#The sturctures of our neural network:

#input: Dense layers with relu and 512 NN unions.

#Output:Dense layers with softmax and 10 NN unions.

from keras import models

from keras import layers

Neural_network=models.Sequential()

Neural_network.add(layers.Dense(512,
                                activation='relu',
                                input_shape=(28*28,)))
Neural_network.add(layers.Dense(10,
                                activation='softmax'))

#The compiler are including optimizer ,loss function, metrics.

Neural_network.compile(optimizer='rmsprop',
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

print("Our DNN can be described by: ")
Neural_network.summary()
#We also need to label the input_images with categorical compiler.
from keras.utils import to_categorical

train_labels=to_categorical(train_labels)
test_labels=to_categorical(test_labels)

#Start to train our NN with 20 times.
history=Neural_network.fit(train_images,
                           train_labels,
                           epochs=20,
                           batch_size=128)

# Evaluate
_, accuracy = Neural_network.evaluate(test_images, test_labels)
print('Loss: %.6f' % _, ', Accuracy: %.2f' % (accuracy*100))

#Show the accuracy rate with each training.
import matplotlib.pyplot as plt

acc=history.history['acc']
loss=history.history['loss']
epochs=range(1,len(acc)+1)

plt.plot(epochs,
         acc,
         'bo',
         label='Traning accuracy')
plt.title('The accuracy of training ')
plt.legend()

#Show the loss rate with each training.
plt.plot(epochs,
         loss,
         'g',
         label='Training loss quantity')
plt.title('Training loss')
plt.legend()

plt.show()
