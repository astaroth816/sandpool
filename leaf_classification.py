# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 10:34:38 2019

@author: andy3
"""

import numpy as np
import pandas as pd
from keras import layers
from keras import models
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout
from keras.optimizers import SGD
from keras.utils import np_utils

train=pd.read_csv('C:/Users/andy3/leaf-classification/train.csv')
test=pd.read_csv('C:/Users/andy3/leaf-classification/test.csv')

print(train)

print(test)
ssss=np.zeros((10,3,64))

print(ssss)
#Refalsh the train,test,labels,classes
def spilt(train,test):
    
    label_encoder = LabelEncoder().fit(train.species)
    
    labels = label_encoder.transform(train.species)
    
    classes = list(label_encoder.classes_)
    
    train = train.drop(['species', 'id'], axis=1)
  
    test = test.drop('id', axis=1)

    return train, labels, test, classes

train, labels, test, classes = spilt(train, test)

print("The classes:",classes)

print("The labels:",labels)

print("The train :",train)

print("The test :",test)

# standardize train  and test features
scaler = StandardScaler().fit(train.values)

test_scaler=StandardScaler().fit(test.values)

scaled_train = scaler.transform(train.values)

scaled_test  = test_scaler.transform(test.values)

print("The scaled_train  of shape:",scaled_train.shape)

print("The scaled_test of shape:",scaled_test.shape)

# split train data into train and validatio
spilt= StratifiedShuffleSplit(test_size=0.2, random_state=23)

for train_index, valid_index in spilt.split(scaled_train,labels):
    
    X_train, X_val = scaled_train[train_index], scaled_train[valid_index]
    
    X_train_label, X_val_label = labels[train_index], labels[valid_index]
    
print("The X_train of shape:",X_train.shape)

print("The X_val of shape:",X_val.shape)

print("The X_train_label of shape:", X_train_label.shape)

print("The X_val_label of shape:",X_val_label.shape)

number_features = 64 # number of features per features type (shape, texture, margin)   
nb_class = len(classes)# number of species of leaf 

#### reshape train data for inputing the NN ####

#Define the zero array for put in data
X_train_r = np.zeros((len(X_train),number_features, 3))

print("The X_train_r of shape:",X_train_r.shape)

X_train_r[:, :, 0] = X_train[:, :number_features]

X_train_r[:, :, 1] = X_train[:, number_features:128]

X_train_r[:, :, 2] = X_train[:, 128:]
print(X_train_r)
print(X_train_r[:, :, 0])
print("The X_train_r[0] of shape:",X_train_r[:, :, 0].shape)

print("The X_train_r[1] of shape:",X_train_r[:, :, 1].shape)

print("The X_train_r[2] of shape:",X_train_r[:, :, 2].shape)

# reshape validation data
X_val_r = np.zeros((len(X_val), number_features, 3))

X_val_r[:, :, 0] = X_val[:, :number_features]

X_val_r[:, :, 1] = X_val[:, number_features:128]

X_val_r[:, :, 2] = X_val[:, 128:]

print("The X_val_r of shape:",X_val_r.shape)

print("The X_val_r[0] of shape:",X_val_r[:, :, 0].shape)

print("The X_val_r[1] of shape:",X_val_r[:, :, 1].shape)

print("The X_val_r[2] of shape:",X_val_r[:, :, 2].shape)
print(X_val_r )
model = Sequential()

model.add(layers.Conv1D(512,
                        1,
                        activation='relu',
                        input_shape=(64, 3)))



model.add(Flatten())

model.add(layers.Dense(500,
                       activation='relu'
                       )
    )
    
model.add(layers.Dense(250,
                       activation='relu'
                       )
    )
    
#The multi-classifical problem,so we use softmax with 46 NN union to achieve the purpose.
x=model.add(layers.Dense(99,
                       activation='softmax'
                       )
    )

print(x)
#Compile the model
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc'])

X_train_label = np_utils.to_categorical(X_train_label, nb_class)

X_val_label = np_utils.to_categorical(X_val_label, nb_class)

from keras.utils import plot_model

plot_model(model,show_shapes=True,to_file='leaf_classification.png')    
    

rmsprop = SGD(lr=0.01, nesterov=True, decay=1e-6, momentum=0.9)

model.compile(loss='categorical_crossentropy',optimizer=rmsprop,metrics=['acc'])

history=model.fit(X_train_r,
                  X_train_label,
                  epochs=15,
                  validation_data=(X_val_r, X_val_label),
                  batch_size=16)

from keras.utils import plot_model

plot_model(model,show_shapes=True,to_file='leaf_classification.png')    
    

import matplotlib.pyplot as plt

acc=history.history['acc']

val_acc=history.history['val_acc']

loss=history.history['loss']

val_loss=history.history['val_loss']

epochs=range(1,len(acc)+1)


plt.plot(epochs,
         acc,
         'bo',
         label='4:1 Traning accuracy')

plt.plot(epochs,
         val_acc,
         'b',
         label='4:1 Validation accuracy')

plt.title('The accuracy of training and validation')

plt.legend()

plt.figure()

plt.plot(epochs,
         loss,
         'bo',
         label="4:1 Training loss quantity")

plt.plot(epochs,
         val_loss,
         'b',
         label='4:1 Validation loss quantity')

plt.title('Training and validation loss')

plt.legend()


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    