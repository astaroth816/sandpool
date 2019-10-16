# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 12:31:36 2019

@author: andy3
"""

##########################
#API
#Mutil-output
#
#從名人發表的文章來預測該名人的年齡，性別，收入
#
##########################
from keras import layers,Input
from keras.models import Model

#The size of article
word_size=50000

#The gorup of people of income
num_income_groups=10

#The input of our API func
posts_input=Input(shape=(None,),dtype='int32',name='posts')

#Use API func,let the input vector put in the Embedding layers,and get the embedding vector with 256 dim
embedding_posts=layers.Embedding(word_size,256)(posts_input)

#print the shape of embedding_posts
print('The shape of embedding posts :',embedding_posts)

#Use CNN layers to deal with the embedding_vector.

#First, let the ebmedding_posts input the Conv1D layer
x=layers.Conv1D(128,
                5,
                activation='relu')(embedding_posts)

#Next, put in the Maxpooling1D layer
x_2=layers.MaxPool1D(5)(x)

#Next, put in the Conv1D layer
x_3=layers.Conv1D(256,5,activation='relu')(x_2)

#Next, put in the Conv1D layer
x_4=layers.Conv1D(256,5,activation='relu')(x_3)

#Next, put in the Maxpooling1D layer
x_5=layers.MaxPool1D(5)(x_4)

#Next, put in the Conv1D layer
x_6=layers.Conv1D(256,5,activation='relu')(x_5)

#Next, put in the Conv1D layer
x_7=layers.Conv1D(256,5,activation='relu')(x_6)

#Next, put in the GlobalMaxPool1D layer
x_8=layers.GlobalMaxPool1D()(x_7)

#Next, put in the Dense layer
x_9=layers.Dense(128,activation='relu')(x_8)

print('Final,we concern the shape of output:',x_9.shape,'Before branch')

###################Next,We will put x_9 tensor into 3 output_layer##########################

###1.output:   Predict the age of output_layer:
age_prediction=layers.Dense(1,name='age')(x_9)

###2.output:   Predict the income of groups of out_layer:
income_prediction=layers.Dense(num_income_groups,
                               activation='softmax',
                               name='income')(x_9)

###3.output:   Predict the male/female of output_layer:
gender_prediction=layers.Dense(1,
                               activation='sigmoid',
                               name='gender')(x_9)

#The structure of API func
API_model=Model(posts_input,
                [age_prediction,income_prediction,gender_prediction])

API_model.summary()

#Because the different output will deal with different mission,so we need different loss function to compile the each output

#Use the loss list method: The each loss function correspond the each output layer
API_model.compile(optimizer='rmsprop',
                  loss=['mse',
                        'categorical_crossentropy',
                        'binary_crossentropy'],
                        loss_weights=[0.25,1.,10.])

#Training our models
API_model.fit(posts_input,[age_targets,income_targets,gender_targers],
              epochs=10,
              batch_size=64)




























