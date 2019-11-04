# -*- coding: utf-8 -*-
"""

@author: andy3

"""



###################################################
#
###########The binary-classifical problem##########
#
#Use Dense layers to comment the movie of review from IMDB,we will achieve to divide the comment though NN in.
#
#IMDB includes the 25000 comment with 50 percent of positive comment and 50 percent of negative comment.
#
#
#
#
#
###################################################

###Prepare the train_data and test_data for fitting the NN model.###

#Download the imdb from keras's package.
from keras.datasets import imdb

(train_data,train_labels),(test_data,test_labels)=imdb.load_data(num_words=10000)

#Use the one-hot encoding to encode the words.
import numpy as np

def vectorize_sequences(sequences,
                        dimension=10000):

    results=np.zeros((len(sequences),
                      dimension))

    for i, sequence in enumerate(sequences):

        results[i,sequence]=1.

    return results

train_data=vectorize_sequences(train_data)

test_data=vectorize_sequences(test_data)

train_labels=np.asarray(train_labels).astype('float32')

test_labels=np.asarray(test_labels).astype('float32')

###Bulid the NN models with sequenctial method###
from keras import models
from keras import layers

#The Sequential method.
model=models.Sequential()

model.add(layers.Dense(16, 
                       activation='relu',
                       input_shape=(10000,)
                       )
        )

model.add(layers.Dense(16,
                       activation='relu',
                       )
        )

#The binary classifical problem,so we use sigmoid with 1 NN union to achieve the purpose.
model.add(layers.Dense(1,
                       activation='sigmoid'
                       )
        )

#Compile the model
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

#Make the first 1000 train_data as validation_data.
validation_data=train_data[:1000]

#Except for the first 1000 validation_data, the rest are partial_train_data.
partial_train_data=train_data[1000:]

#Make the first 1000 train_labels as validation_labels.
validation_labels=train_labels[:1000]

#Except for the first 1000 validation_labels, the rest are partial_train_labels.
partial_train_labels=train_labels[1000:]

#Start training the model.
history=model.fit(partial_train_data,
                  partial_train_labels,
                  epochs=20,
                  batch_size=512,
                  validation_data=(validation_data,validation_labels))

#Analyize the training result with ploting the training progress.
import matplotlib.pyplot as plt

history_dict=history.history

history_dict.keys()

loss_values=history_dict['loss']

val_loss_values=history_dict['val_loss']

epochs=range(1,len(loss_values)+1)

plt.plot(epochs,
         loss_values,
         'bo',
         label='Training loss')

plt.plot(epochs,
         val_loss_values,
         'b',
         label='Valadation loss'
         )

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()

plt.clf()

acc=history_dict['acc']

val_acc=history_dict['val_acc']

plt.plot(epochs,
         acc,
         'bo',
         label='Training accuracy'
         )

plt.plot(epochs,
         val_acc,
         'b',
         label='Valadation accccuracy'
         )

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()

#Use test_data and test_labels to evaluate the accuracy rate though the model.
result=model.evaluate(test_data,test_labels)

print("the accuracy rate though the model:",result[1])