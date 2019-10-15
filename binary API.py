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
