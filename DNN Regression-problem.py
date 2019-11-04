# -*- coding: utf-8 -*-
"""
@author: andy3

"""
###################################################
#
###########The regression problem##########
#
#Use Dense layers to predict the price of house in Boston though the NN model after training boston_husing datasets.
#
#Boston_husing datasets includes the 404 train_data and 102 test_data,and each data has different(13) features.
#
#The 13 features include:1.crim : per capita crime rate by town.
#                        2.zn : proportion of residential land zoned for lots over 25,000 sq.ft.
#                        3.indus : proportion of non-retail business acres per town.
#                          .
#                          .  
#                          .
#                        13.medv : median value of owner-occupied homes in $1000s.
###################################################

###Prepare the train_data and test_data for fitting the NN model.###

#Download the boston_housing from keras's package.
from keras.datasets import boston_housing

(train_data,train_labels),(test_data,test_labels)=boston_housing.load_data()

print("The shape of train_data which include:(number of data , 13 features)",train_data.shape)

print("The shape of test_data which include:(number of data , 13 features)",test_data.shape)

print("The shape of train_labels which include:(number of data , None)",train_labels.shape)

print("The shape of test_labels which include:(number of data , None)",test_labels.shape)

#Normalizaion method: the train_data, normalization includes the 1.train_data - mean as axis=0 and 2.train_data / std as axis=0

#mean= average(train_data) and train_data - mean as axis=0
mean=train_data.mean(axis=0)

train_data-=mean

#std=standard deviation and train_data / std as axis=0
std=train_data.std(axis=0)

train_data /=std

test_data-=mean

test_data /=std

###Bulid the NN models with sequenctial method###
from keras import models

from keras import layers

def sturcture_model():
    
    model=models.Sequential()
    
    model.add(layers.Dense(64,
                           activation='relu',
                           input_shape=(train_data.shape[1], )
                           )
    )
    
    model.add(layers.Dense(64,
                           activation='relu')
    )
    
    model.add(layers.Dense(1))
    
    model.compile(optimizer='rmsprop',
                  loss='mse',#Because we use the normalization method to deal with the train_data/test_data.
                  metrics=['mae'])
    return model

#K-fold cross validation method.Because the number of train_data and test_data are too little.

#In this reason, we can use K-fold cross validation method to increase the number of train_data and test_data.

import numpy as np

#4-fold
k=4

num_val_samples=len(train_data)//k

#The epochs=100
num_epochs=100

all_scores=[]

for i in range(k):
    
    print('processing fold #',i)
    
    val_data=train_data[i*num_val_samples:(i+1)*num_val_samples]
    
    val_labels=train_labels[i*num_val_samples:(i+1)*num_val_samples]
    
    partial_train_data=np.concatenate([train_data[:i*num_val_samples],train_data[(i+1)*num_val_samples:]],axis=0)
    
    partial_train_targets=np.concatenate([train_labels[:i*num_val_samples],train_labels[(i+1)*num_val_samples:]],axis=0)
    
    model=sturcture_model()
    
    history=model.fit(partial_train_data ,
                      partial_train_targets ,
                      epochs=num_epochs,
                      batch_size=1,
                      verbose=0)
    
    val_mse,val_mae=model.evaluate(val_data,val_labels,verbose=0)
    
    all_scores.append(val_mae)

print(all_scores)

print(np.mean(all_scores))

#K-fold cross validation method for the epochs=500.
num_epochs=500

all_mae_histories=[]

for i in range(k):
    
    print('processing fold #',i)
    
    val_data=train_data[i*num_val_samples:(i+1)*num_val_samples]
    
    val_targets=train_labels[i*num_val_samples:(i+1)*num_val_samples]
    
    partial_train_data=np.concatenate([train_data[:i*num_val_samples],
                                       train_data[(i+1)*num_val_samples:]],
                                       axis=0)
    partial_train_targets=np.concatenate([train_labels[:i*num_val_samples],
                                          train_labels[(i+1)*num_val_samples:]],
                                          axis=0)
    model=sturcture_model()
    
    history=model.fit(partial_train_data,
                      partial_train_targets,
                      validation_data=(val_data,val_targets),
                      epochs=num_epochs,
                      batch_size=1,
                      verbose=0)
    
    mae_history=history.history['val_mean_absolute_error']
    
    all_mae_histories.append(mae_history)
    
average_mae_history=[np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

import matplotlib.pyplot as plt

plt.plot(range(1,len(average_mae_history)+1),
         average_mae_history)

plt.xlabel('Epochs')

plt.ylabel('Validation MAE')

plt.show()

plt.legend()


