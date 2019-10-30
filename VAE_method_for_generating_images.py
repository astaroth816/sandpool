# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 13:19:57 2019

@author: andy3
"""

###################################
#
#
#Variatioonal Autoencoders,VAE
#
#
#
#
##################################

###Let the input_graph put in 2 vector###
#We use the MNIST datasets as our input_graph
import keras
from keras import layers
from keras import backend as K
from keras.models import Model
import numpy as np

img_shape=(28,28,1)

batch_size=16

#latent space
latent_dim=2

###Setting the inputting
input_img=keras.Input(shape=img_shape)
print(input_img.shape)

#Use API method to constructure our NN
x=layers.Conv2D(32,
                3,
                padding='same',
                activation='relu'
                )(input_img)

x=layers.Conv2D(64,
                3,
                padding='same',
                activation='relu',
                strides=(2,2)
                )(x)

x=layers.Conv2D(64,
                3,
                padding='same',
                activation='relu'
                )(x)

x=layers.Conv2D(64,
                3,
                padding='same',
                activation='relu'
                )(x)

#After convolutional input_img of shape
print('The shape of input_img:',input_img.shape)

#Use tuple to feedback the size of shape
shape_before_flattening=K.int_shape(x)

#Print the shape of shape_before_fattening
print('The shape of shape_before_flattening:',shape_before_flattening)

x=layers.Flatten()(x)

x=layers.Dense(32,activation='relu')(x)


print('The shape of x :',x.shape)

#encode the inputing graph to 2 type parameters
z_mean=layers.Dense(2)(x)

z_var=layers.Dense(2)(x)




print(z_mean.shape,z_var.shape)

#Use z_mean,z_var to get the code ,the z_mean and z_var are the probability distribution of generating input_img
def sampling(args):
    z_mean,z_var=args
    epsilon=K.random_normal(shape=(K.shape(z_mean)[0],latent_dim),mean=0.,stddev=1.)
    return z_mean+K.exp(z_var)*epsilon

z=layers.Lambda(sampling)([z_mean,z_mean])

#reshape the size of z,and use some CNN layers to get the new_img with same size of input_img
docoder_input=layers.Input(K.int_shape(z)[1:])#Let the input_tensor z of (,2) as our input

x=layers.Dense(np.prod(shape_before_flattening[1:]),
               activation='relu')(docoder_input)

print('The shape of x:',x.shape)

#Reshape the x to pre-flatren layers of shape
x=layers.Reshape(shape_before_flattening[1:])(x)

print('The shape of x:',x.shape)

#CNN layer
x=layers.Conv2DTranspose(32,
                         3,
                         padding='same',
                         activation='relu',
                         strides=(2,2))(x)

print('The shape of x though Conv2DTranspose layers:',x.shape)

x=layers.Conv2D(1,
                3,
                padding='same',
                activation='sigmoid')(x)

print('The shape of x though Conv2D layer:',x.shape)

model=Model(docoder_input,x)
model.summary()

z_decoded=model(z)

##Define the loss function with VAE!
class VAElayer(keras.layers.Layer):
    
    #Calculate the VAE loss quantity
    def vae_loss(self,x,z_decoded):
        x=K.flatten(x)
        z_decoded=K.flatten(z_decoded)
        xent_loss=keras.metrics.binary_crossentropy(x,z_decoded)
        kl_loss=-5e-4*K.mean(1+z_mean-K.square(z_var)-K.exp(z_var),axis=-1)
        return K.mean(xent_loss+kl_loss)
    
    #Define the call method
    def call(self,inputs):
        x=inputs[0] #inputing img
        z_decoded=inputs[1] #encodeing img
        loss=self.vae_loss(x,z_decoded) #use the x and z_decoded to calcualte the VAE loss
        self.add_loss(loss,inputs=inputs)#add the "loss" method to our define
        
        return x
#Use origin input(input_img) and decode (z_deocded) as base,call the VAElayer() to get the valuse of calcuation.
y=VAElayer()([input_img,z_decoded])

print(y.shape)
vaaae=Model(input_img,y)
from keras.utils import plot_model

plot_model(vaaae,show_shapes=True,to_file='model.png')

#Traning the VAE loss function , we use the MNIST as our input_img
from keras.datasets import mnist

vae=Model(input_img,y)

vae.compile(optimizer='rmsprop',loss=None)

vae.summary()

(x_train,_),(x_test,y_test)=mnist.load_data()

x_train=x_train.astype('float64')/255

x_train=x_train.reshape(x_train.shape+(1,))
    
x_test=x_test.astype('float64')/255

x_test=x_test.reshape(x_test.shape+(1,))

vae.fit(x=x_train,y=None,
        shuffle=True,
        epochs=2,
        batch_size=batch_size,
        validation_data=(x_test,None))


##Create the graph of generating graph from MNIST
vaaae=Model(input_img,z_decoded)
from keras.utils import plot_model

plot_model(vaaae,show_shapes=True,to_file='model.png')

import matplotlib.pyplot as plt
from scipy.stats import norm

n=10
digit_size=28

img=np.zeros((digit_size*n,digit_size*n))

grid_x=norm.ppf(np.linspace(0.05,0.95,n))
grid_y=norm.ppf(np.linspace(0.05,0.95,n))

for i,j in enumerate(grid_x):
    for w,z in enumerate(grid_y):
        z_sample=np.array([[z,j]])
        z_sample=np.tile(z_sample,batch_size).reshape(batch_size,2)
        x_decoded=model.predict(z_sample,batch_size=batch_size)
        digit=x_decoded[0].reshape(digit_size,digit_size)
        img[i*digit_size:(i+1)*digit_size,
            w*digit_size:(w+1)*digit_size]=digit
            
plt.figure(figsize=(10,10))
plt.imshow(img,cmap='Greys_r')
plt.show

