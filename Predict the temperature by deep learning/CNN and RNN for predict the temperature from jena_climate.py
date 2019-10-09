# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 15:39:47 2019

@author: andy3
"""

##############################################
#
#
# the CNN  to predict the temperature
#
#
#
#
#
#
###############################################
#Take data from jena_climate,and encode it to the list.
import os

data_dir='C:/Users/andy3/jena_climate'

fname=os.path.join(data_dir,'jena_climate_2009_2016.csv')

f=open(fname)

data=f.read()

f.close()

#Fisrt, we split the data, the header is outline of data, and the "lines" is the total data without the time(the first raw 'Date Time')

lines=data.split('\n')

header=lines[0].split(',')

#the header can be described by:

print("The outline of data can be written as: ",header)

#We split out the "Date Time" and its data.

lines=lines[1:]

#The data quantity:

print("The quantity of data are given by:",len(lines))

#Create the zeros tensor.

import numpy as np

#The dimension of tensor can be given by:(data quantity , the quantity of outline without Date Time)

float_data=np.zeros((len(lines),len(header)-1))

#Use the ennmerate cycle to put in the data into the tensor.(one by one via the for loop)

for i ,line in enumerate(lines):
    
    values=[float(x) for x in line.split(',')[1:]]
    
    float_data[i, :]=values

#Exactly,the dimension of "float_data" are (420511,15-1).
    
print("The dimension of data tensor :",float_data.shape)

#####show relation of temperature and time with graph.

from matplotlib import pyplot as plt

#We just need the temperature,so we split the "float_data" tensor.

temp=float_data[:,1]

plt.title('The relation of temp and time in total data(420551) ')

plt.plot(range(len(temp)),temp)

plt.show()

plt.figure()

plt.title('The relation of temp and time in 10 days ')

plt.plot(range(1440),temp[:1440])

plt.show()

#先減去每個時間序列資料的平均值並除以標準差來預先處理資料(標準化)。因為訓練集(Training set)只訓練200000個。

#標準化是一種資料處理的方式，我們需要其平均值(mean)和標準差(std)並取平均值為0的地方當原點。

mean=float_data[:200000].mean(axis=0)

float_data-=mean

std=float_data[:200000].std(axis=0)

float_data/=std

#define the generator(data,lookback,delay,min index,max index,shuffle=False),the purpose is generating the sample of time.

#data=origin data in array,we deal it above.

#lookback:the time point should feedback the input data.

#delay:what time is it in future

#min_index,max_index:main and max of data,including the time point,we can use them to validation and test.

#shuffle=we break the sample in time.

# generator function used to feed the training, validation and test data.
def generator(data, lookback, delay, min_index, max_index,
                shuffle=False, batch_size=128, step=6):
    
    if max_index is None:
        
        max_index = len(data) - delay - 1
    
    i = min_index + lookback
    
    while 1:
        
        if shuffle:
            
            rows = np.random.randint(
                    
                    min_index + lookback, max_index, size=batch_size)
        else:
            
            if i + batch_size >= max_index:
                
                i = min_index + lookback
            
            rows = np.arange(i, min(i + batch_size, max_index))
            
            i += len(rows)

        samples = np.zeros((len(rows),
                            lookback // step,
                            data.shape[-1]))
        
        targets = np.zeros((len(rows),))
        
        for j, row in enumerate(rows):
            
            indices = range(rows[j] - lookback, rows[j], step)
            
            samples[j] = data[indices]
            
            targets[j] = data[rows[j] + delay][1]
        
        yield samples, targets    
        
#After we created the generator function,  we also need to create 3 generators:training sets,validation sets,testing sets.

#Create the training sets,validation sets,testing sets

lookback=1440

step=6

delay=144

batch_size=128

#Training sets generator:

train_gen = generator(float_data,
                        lookback=lookback,
                        delay=delay,
                        min_index=0,
                        max_index=200000,
                        shuffle=True,
                        step=step,
                        batch_size=batch_size)

#validation sets generator:

val_gen = generator(float_data,
                        lookback=lookback,
                        delay=delay,
                        min_index=200001,
                        max_index=300000,
                        step=step,
                        batch_size=batch_size)

#testing sets generator:

test_gen = generator(float_data,
                        lookback=lookback,
                        delay=delay,
                        min_index=300001,
                        max_index=None,
                        step=step,
                        batch_size=batch_size)
#the number of validating rounds:

val_steps = (300000 - 200001 - lookback)  // batch_size # How many steps to draw from
            # val_gen in order to see the entire validation set

#the number of testing rounds:

test_steps = (len(float_data) - 300001 - lookback)  // batch_size # How many steps to draw
        # from test_gen in order to see the entire test set 

#######Use Dense layers to evulate the temperature accuracy########
        
from keras.models import Sequential

from keras import layers

from keras.optimizers import RMSprop

Neural_network=Sequential()

#Use the conv1d layers as our forst hidden layers
Neural_network.add(layers.Conv1D(32,
                                 5,
                                 activation='relu',
                                 input_shape=(None,float_data.shape[-1])))

#Use the Maxpooling layers as our second hidden layers
Neural_network.add(layers.MaxPool1D(3))


#Use the Maxpooling layers as our second hidden layers
Neural_network.add(layers.Conv1D(32,
                                 5,
                                 activation='relu'))

Neural_network.add(layers.GRU(32,dropout=0.1,recurrent_dropout=0.5))

#Use the Dense layers as our output layses
Neural_network.add(layers.Dense(1))
#The optimizer we use RMSprop ,the loss function we use mean square average.

Neural_network.compile(optimizer=RMSprop(),loss='mae')

Neural_network.summary()

history = Neural_network.fit_generator(train_gen,
                            steps_per_epoch=500,
                            epochs=20,
                            validation_data=val_gen,
                            validation_steps=val_steps)

#We plot the relation of trainimg loss and validation loss.

import matplotlib.pyplot as plt

loss=history.history['loss']

val_loss=history.history['val_loss']

epochs=range(1,len(loss)+1)

plt.figure()

plt.plot(epochs,loss,'bo',label='Traning loss')

plt.plot(epochs,val_loss,'b',label='Validation loss')

plt.title('The accuracy of training and validation')

plt.legend()

plt.show()

