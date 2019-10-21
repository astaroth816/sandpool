# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:59:53 2019

@author: andy3
"""

#######################################
#
#Generator sequence with RNN(LSTM)
#
#
#
#
#
#######################################
###Download the analysze the training article.###
import keras
import numpy as np

path=keras.utils.get_file(
        'generatorsequence.txt',
        origin='http://s3.amazonaws.com/text-datasets/nietzsche.txt')
text=open(path,encoding="utf-8").read().lower()
print('The number of words:',len(text))
#######################################
#
#
#
#
#
#
#
#
#######################################

###Vectorize the word's sequence.###

max_word_union=60

step=3

sentences=[]

next_chars=[]

for i in range(0,len(text)-max_word_union,step):
    
    sentences.append(text[i:i+max_word_union])
    
    next_chars.append(text[i+max_word_union])
    
print('The number of senquence:',len(sentences))

chars=sorted(list(set(text)))

print('Unique_word_union:',len(chars))

###Put the unique word_union into {} dict type means:{a,b,c...}

char_indices=dict((char,chars.index(char))for char in chars)

print('Encodeing to vectorization words(one-hot encoding')

x=np.zeros((len(sentences),max_word_union,len(chars)),dtype=np.bool)

y=np.zeros((len(sentences),len(chars)),dtype=np.bool)

print("The shape of x:",x.shape)

print('The shape of y:',y.shape)

for i,sentences in enumerate(sentences):
    
    for t,char in enumerate(sentences):
        
        x[i,t,char_indices[char]]=1
        
    y[i,char_indices[next_chars[i]]]=1

###Create the Neural Network###

from keras import layers

from keras import models


model=keras.models.Sequential()

model.add(layers.LSTM(128,input_shape=(max_word_union,len(chars))))

model.add(layers.Dense(len(chars),activation='softmax'))

optimizer = keras.optimizers.RMSprop(lr=0.01)

model.compile(optimizer=optimizer,loss='categorical_crossentropy')

def sample(predicts,temperature=1.0):
    
    predicts=np.asarray(predicts).astype('float64')
    
    predicts=np.log(predicts)/temperature
    
    exp_predicts=np.exp(predicts)
    
    predicts=exp_predicts/np.sum(exp_predicts)
    
    probas=np.random.multinomial(1,predicts,1)
    
    return np.argmax(probas)

import random
import sys

for epoch in range(1,60):
    print('epochs:',epoch)
    model.fit(x,
              y,
              batch_size=128,
              epochs=1)
   #Random choose the certern 60 word_union of article
    start_index=random.randint(0,len(text)-max_word_union-1)
    generated_text=text[start_index:start_index+max_word_union]
    print('---The random of initial word_sentence:"'+generated_text+'"')
    for temperature in [0.5]:
        sys.stdout.write(generated_text)
        #each temperature of generator 5oo word_union
        for i in range(400):
            sampled=np.zeros((1,max_word_union,len(chars)))
            
            for t,char in enumerate(generated_text):
               
                sampled[0,t,char_indices[char]]
            
            #Generate the word_union of probability
            predicts=model.predict(sampled,verbose=0)[0]
            next_index=sample(predicts,temperature)
            next_char=chars[next_index]
            generated_text+=next_char
            generated_text=generated_text[1:]
            sys.stdout.write(next_char)
            

