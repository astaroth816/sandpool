# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 14:52:48 2019

@author: andy3
"""

######################################
#API func
#Mutil-input data
#
#
#
#
#
######################################
#Use API to pratice the binary-input of question-answer model
from keras import Model

from keras import layers

from keras import Input

#Create the size of question and answer
text_word_size=10000

#question's word of size:
question_word_size=10000

#answer's word of size:
answer_word_size=500


#################The first input_data(text_input)#######################

#We don't limit the inputubg the shape of input (shape=(None,)).
text_input=Input(shape=(None,),dtype='int32',name="text")

#Let the input_data(text_input) input into the Embedding layers(embedding layer) 
embedded_text=layers.Embedding(text_word_size,64)(text_input)

#print the output of embedding layer(enbedded_text):
print('embedded_text of shape:',embedded_text.shape)

#Through the RNN(LSTM) layers,We encode the embedded_text to single vector:
encoded_text=layers.LSTM(32)(embedded_text)

#print the output of encode_text:
print('encoded_text of shape:',encoded_text.shape)

#################The second input_data(question_input)#######################

#The setup of dealing with the input_data(question):
question_input=Input(shape=(None,),dtype="int32",name="question")

# Let the input_data(question_input) input into the Embedding layers(embedding layer) 
embedded_question=layers.Embedding(question_word_size,32)(question_input)

#prnt the output of embedding layer(enbedded_question): 
print('embedded_text of shape:',embedded_question.shape)

#Through the RNN(LSTM) layers,We encode the embedded_question to single vector:
encoded_question=layers.LSTM(16)(embedded_question)

#print the output of encode_question:
print('encoded_question of shape:',encoded_question.shape)

#combine the "question" and "text" and concatenate one layer(concatenated layer)
concatenated=layers.concatenate([encoded_question,encoded_text],axis=-1)

#print the concatenated.shape:
print("concatenated of shape:",concatenated.shape)

#The output layers we use softmax with dense layer
answer=layers.Dense(answer_word_size,activation='softmax')(concatenated)

#print the answer.shape
print('answer of shape:',answer.shape)

API_model=Model([text_input,question_input],answer)

API_model.compile(optimizer="rmsprop",
                  loss="categorical_crossentropy",
                  metrics=['acc'])
API_model.summary()

##########Create the training data#############################
import numpy as np

num_samples=1000

max_length=100

###Create the "text" training data:1000 piece of data and each data has 100word.###
text=np.random.randint(1,text_word_size,size=(num_samples,max_length))

#[[2,23,54,......totaly 100 word],[...],[...],...total 1000 piece of data]
print("The shape of text:",text.shape)

###Create the "question" training data###
question=np.random.randint(1,question_word_size,size=(num_samples,max_length))

#print the question shape:
print('The shape of question:',question.shape)

###Create the "answer",we need one-hot encode ,total 1000 correct answer
answers=np.zeros(shape=(num_samples,
                        answer_word_size),dtype='int32')

#let arbitary(random) one set 1
for answer in answers:
    answer[np.random.randint(answer_word_size)]=1

#[[0,1,0,.....total 500 word],[...],[...],[...],..total 1000 list]

#print the shape of answer:
print('The shape of answer:',answers.shape)

#Training method 1:use list method put in training sets
history_list=API_model.fit([text,question],answers,epochs=10,batch_size=128)

#Training method 2:use dict method put in training sets
history_dict=API_model.fit({'text':text,'question':question},answers,epochs=10,batch_size=128)

#Plot the history_list with acc and loss
