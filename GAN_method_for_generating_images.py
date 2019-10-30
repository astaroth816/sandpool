# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 11:22:56 2019

@author: andy3
"""

###################################
#
#
#The generative adersarial network
#
#
#
#Generator NN vs discriminator NN
#
#
##################################

######Generator######
import keras
from keras import layers
import numpy as np

#The latent space of dimension
latent_dim=32

#The size of graph(with 3 channel(RGB)) of generator
height=32

width=32

channels=3

####Create the neural network of generator####

#Createh the input of generator with shape=(?,latent_dim)
Generator_input=keras.Input(shape=(latent_dim,))

#Use API method to construct the Neural netwrok
x=layers.Dense(128*16*16)(Generator_input)

#For the mission, we use lEAKYRelu instead the Relu activation
x=layers.LeakyReLU()(x)
 
x=layers.Reshape((16,
                  16,
                  128))(x)

#Add the Convalution layers
x=layers.Conv2D(256,
                5,
                padding='same'
                )(x)

x=layers.LeakyReLU()(x)

#Add the anti-Convalution layers
x=layers.Conv2DTranspose(256,
                         4,
                         strides=2,
                         padding='same'
                         )(x)

x=layers.LeakyReLU()(x)

#Add the Convalution layers
x=layers.Conv2D(256,
                5,
                padding='same'
                )(x)

x=layers.LeakyReLU()(x)

x=layers.Conv2D(channels,
                7,
                activation='tanh',
                padding='same'
                )(x)

Generator=keras.models.Model(Generator_input,
                             x
                             )

from keras.utils import plot_model

plot_model(Generator,
           show_shapes=True,
           to_file='Generator.png')

######Discriminator neural network######

#Create the discriminator neural network
Discriminator_input=layers.Input(shape=(height,
                                        width,
                                        channels))

x=layers.Conv2D(128,
                3)(Discriminator_input)

x=layers.LeakyReLU()(x)

x=layers.Conv2D(128,
                4,
                strides=2)(x)

x=layers.LeakyReLU()(x)

x=layers.Conv2D(128,
                4,
                strides=2)(x)

x=layers.LeakyReLU()(x)

x=layers.Conv2D(128,
                4,
                strides=2)(x)

x=layers.LeakyReLU()(x)

x=layers.Flatten()(x)

x=layers.Dropout(0.5)(x)

x=layers.Dense(1,
               activation='sigmoid')(x)

Discriminator=keras.models.Model(Discriminator_input,x)

Discriminator_optimizer=keras.optimizers.RMSprop(lr=0.0008,
                                                 clipvalue=1.0,
                                                 decay=1e-8)#For the steady learning, we use decay learning
                                                             
Discriminator.compile(optimizer=Discriminator_optimizer,
                      loss='binary_crossentropy')

from keras.utils import plot_model

plot_model(Discriminator,
           show_shapes=True,
           to_file='Discriminator.png')

###gan neural network###
Discriminator.trainable=False

gan_input=keras.Input(shape=(latent_dim,))

gan_output=Discriminator(Generator(gan_input))

Gan=keras.models.Model(gan_input,gan_output)

Gan_optimizer=keras.optimizers.RMSprop(lr=0.0004,
                                       clipvalue=1.0,
                                       decay=1e-8)

Gan.compile(optimizer=Gan_optimizer,loss='binary_crossentropy')
Gan.summary()
from keras.utils import plot_model

plot_model(Gan,
           show_shapes=True,
           to_file='Gan.png')

#Implement the train of GAN
import os
from keras.preprocessing import image

(x_train,y_train),(_,_)=keras.datasets.cifar10.load_data()

x_train=x_train[y_train.flatten()==5]

x_train=x_train.reshape((x_train.shape[0],)+(height,width,channels)).astype('float32')/255

iterations=10000
batch_size=20

save_dir='C:/Users/andy3/'

#-------Start traning------------#

print('----------Start training--------------------')

start=0

for step in range(iterations):
    
    random_latent_vectors=np.random.normal(size=(batch_size,
                                                 latent_dim))
    
    #Generate the false images(generated_images)
    generated_images=Generator.predict(random_latent_vectors)
    
    #Combination the generated_images and real_images
    stop=start+batch_size
    
    real_images=x_train[start:stop]
    
    combined_imagees=np.concatenate([generated_images,real_images])
    
    labels=np.concatenate([np.ones((batch_size,1)),
                           np.zeros((batch_size,1))
                           ]
                          )
    
    #Add the arbitrary noise
    labels += 0.05*np.random.random(labels.shape)
    
    #Training the Discriminator
    d_loss=Discriminator.train_on_batch(combined_imagees,labels)
    
    
    random_latent_vectors=np.random.normal(size=(batch_size,
                                                 latent_dim))
    
    #Tell the Generator wichis ture of false
    misleading_targets=np.zeros((batch_size,1))
    
    #Training the Generator and freeze the weight of Discriminator
    a_loss=Gan.train_on_batch(random_latent_vectors,
                              misleading_targets)
    
    start += batch_size
    if start > len(x_train) -batch_size:
        start=0
    
    if step % 100 ==0:
        Gan.save_weights('Gan.h5')
        
        print('Discriminator loss at NO. %s:%s'%(step,d_loss))
        print('Adversarial loss at NO. %s:%s'%(step,a_loss))
        
        #Save the generative images
        img=image.array_to_img(generated_images[0]*255.,scale=False)
    
        img.save(os.path.join(save_dir,'generated_images'+str(step)+'.png'))
        
        #Save the Real images to compare the geenerative images
        img=image.array_to_img(real_images[0]*255.,scale=False)
        
        img.save(os.path.join(save_dir,'real_images'+str(step)+'.png'))