# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 13:42:47 2019

@author: andy3
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt  

from timeit import default_timer as timer
from sklearn import preprocessing


from ultimate.mlp import MLP 

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint

import gc, sys
gc.enable()
def state(message,start = True, time = 0):
    if(start):
        print(f'Working on {message} ... ')
    else :
        print(f'Working on {message} took ({round(time , 3)}) Sec \n')
        
INPUT_DIR = "../input/"
INPUT_DIR = 'C:/Users/andy3/'

def feature_engineering(is_train):
    # When this function is used for the training data, load train_V2.csv :
    if is_train: 
        print("processing train_V2.csv")
        df = pd.read_csv(INPUT_DIR + 'train_V2.csv')
        
        # Only take the samples with matches that have more than 1 player 
        # there are matches with no players or just one player ( those samples could affect our model badly) 
        df = df[df['maxPlace'] > 1]
    
    # When this function is used for the test data, load test_V2.csv :
    else:
        print("processing test_V2.csv")
        df = pd.read_csv(INPUT_DIR + 'test_V2.csv')
        
    # Make a new feature indecating the total distance a player cut :
    state('totalDistance')
    s = timer()
    df['totalDistance'] = df['rideDistance'] + df["walkDistance"] + df["swimDistance"]
    e = timer()
    state('totalDistance', False, e - s)
          

    state('rankPoints')
    s = timer()
    # Process the 'rankPoints' feature by replacing any value of (-1) to be (0) :
    df['rankPoints'] = np.where(df['rankPoints'] <= 0 ,0 , df['rankPoints'])
    e = timer()                                  
    state('rankPoints', False, e-s)
    

    target = 'winPlacePerc'
    # Get a list of the features to be used
    features = list(df.columns)
    
    # Remove some features from the features list :
    features.remove("Id")
    features.remove("matchId")
    features.remove("groupId")
    features.remove("matchDuration")
    features.remove("matchType")
    
    y = None
    
    # If we are processing the training data, process the target
    # (group the data by the match and the group then take the mean of the target) 
    if is_train: 
        y = np.array(df.groupby(['matchId','groupId'])[target].agg('mean'), dtype=np.float64)
        # Remove the target from the features list :
        features.remove(target)
    
    # Make new features indicating the mean of the features ( grouped by match and group ) :
    print("get group mean feature")
    agg = df.groupby(['matchId','groupId'])[features].agg('mean')
    # Put the new features into a rank form ( max value will have the highest rank)
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    
    
    # If we are processing the training data let df_out = the grouped  'matchId' and 'groupId'
    if is_train: df_out = agg.reset_index()[['matchId','groupId']]
    # If we are processing the test data let df_out = 'matchId' and 'groupId' without grouping 
    else: df_out = df[['matchId','groupId']]
    
    # Merge agg and agg_rank (that we got before) with df_out :
    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_mean", "_mean_rank"], how='left', on=['matchId', 'groupId'])
    
    # Make new features indicating the max value of the features for each group ( grouped by match )
    print("get group max feature")
    agg = df.groupby(['matchId','groupId'])[features].agg('max')
    # Put the new features into a rank form ( max value will have the highest rank)
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    
    # Merge the new (agg and agg_rank) with df_out :
    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_max", "_max_rank"], how='left', on=['matchId', 'groupId'])
    
    # Make new features indicating the minimum value of the features for each group ( grouped by match )
    print("get group min feature")
    agg = df.groupby(['matchId','groupId'])[features].agg('min')
    # Put the new features into a rank form ( max value will have the highest rank)
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    
    # Merge the new (agg and agg_rank) with df_out :
    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_min", "_min_rank"], how='left', on=['matchId', 'groupId'])
    
    # Make new features indicating the number of players in each group ( grouped by match )
    print("get group size feature")
    agg = df.groupby(['matchId','groupId']).size().reset_index(name='group_size')
     
    # Merge the group_size feature with df_out :
    df_out = df_out.merge(agg, how='left', on=['matchId', 'groupId'])
    
    # Make new features indicating the mean value of each features for each match :
    print("get match mean feature")
    agg = df.groupby(['matchId'])[features].agg('mean').reset_index()
    
    # Merge the new agg with df_out :
    df_out = df_out.merge(agg, suffixes=["", "_match_mean"], how='left', on=['matchId'])
    
    # Make new features indicating the number of groups in each match :
    print("get match size feature")
    agg = df.groupby(['matchId']).size().reset_index(name='match_size')
    
    # Merge the match_size feature with df_out :
    df_out = df_out.merge(agg, how='left', on=['matchId'])
    
    # Drop matchId and groupId
    df_out.drop(["matchId", "groupId"], axis=1, inplace=True)
    
    # X is the output dataset (without the target) and y is the target :
    X = np.array(df_out, dtype=np.float64)
    
    
    del df, df_out, agg, agg_rank
    gc.collect()

    return X, y

# Process the training data :
x_train, y = feature_engineering(True)

# Scale the data to be in the range (-1 , 1)
scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=False).fit(x_train)

print("x_train", x_train.shape, x_train.max(), x_train.min())
scaler.transform(x_train)
print("x_train", x_train.shape, x_train.max(), x_train.min())

y = y * 2 - 1
print("y", y.shape, y.max(), y.min())

# create NN_model
NN_model = Sequential()
NN_model.add(Dense(x_train.shape[1],  input_dim = x_train.shape[1], activation='relu'))
NN_model.add(Dense(250, activation='relu'))
NN_model.add(Dense(125, activation='relu'))
NN_model.add(Dense(75, activation='relu'))
# output Layer
NN_model.add(Dense(1, activation='linear'))

# Compile the network :
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
NN_model.summary()

checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]

history=NN_model.fit(x_train,y, batch_size=1000,
             epochs=20, verbose=1, callbacks=callbacks_list,
             validation_split=0.15, validation_data=None, shuffle=True,
             class_weight=None, sample_weight=None, initial_epoch=0,
             steps_per_epoch=None, validation_steps=None)

acc=history.history['mean_absolute_error']

val_acc=history.history['val_mean_absolute_error']

loss=history.history['loss']

val_loss=history.history['val_loss']

epochs=range(1,len(acc)+1)
def averagenum(num):
    nsum = 0
    for i in range(len(num)):
        nsum += num[i]
    return nsum / len(num)

print("The average of model_3_250 validation_accuracy:",averagenum(val_acc))


plt.plot(epochs,
         acc,
         'red',
         label="model_3_250_acc")

plt.plot(epochs,
         val_acc,
         'b',
         label="model_3_250_val_acc")

plt.title('Model_3_250_acc')

plt.legend()

plt.figure()