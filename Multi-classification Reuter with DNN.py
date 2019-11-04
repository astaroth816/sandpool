###################################################
#
###########The multi-classifical problem##########
#
#Use Dense layers to classify the subject of news from Reuter's datasets,we will classify to divide the news though NN.
#
#Reuter's datasets includes the 8982 train_data and 2246 test_data.
#
#
#
#
#
###################################################



###Prepare the train_data and test_data for fitting the NN model.###

#Download the reuters datasets from keras's package.
from keras.datasets import reuters

(train_data,train_labels),(test_data,test_labels)=reuters.load_data(num_words=10000)

#Use the one-hot encoding to encode the train_data and test_data.
import numpy as np

def vectorize_sequences(sequences,dimension=10000):

    results=np.zeros((len(sequences),dimension))

    for i, sequence in enumerate(sequences):

        results[i,sequence]=1.

    return results

train_data=vectorize_sequences(train_data)

test_data=vectorize_sequences(test_data)

#Use the one-hot encoding to encode the train_labels and test_labels.
def vectorize_sequences(labels,dimension=46):

    results=np.zeros((len(labels),dimension))

    for i, label in enumerate(labels):

        results[i,label]=1.

    return results

train_labels=vectorize_sequences(train_labels)

test_labels=vectorize_sequences(test_labels)

###Bulid the NN models with sequenctial method###
from keras import models

from keras import layers

model=models.Sequential()

model.add(layers.Dense(64,
                       activation='relu',
                       input_shape=(10000,)
                       )
    )
    
model.add(layers.Dense(64,
                       activation='relu'
                       )
    )
    
#The multi-classifical problem,so we use softmax with 46 NN union to achieve the purpose.
model.add(layers.Dense(46,
                       activation='softmax'
                       )
    )
    
#Compile the model
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
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
         label='Valadation loss')

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
         label='Training accuracy')

plt.plot(epochs,
         val_acc,
         'b',
         label='Valadation accccuracy')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()

#Use test_data and test_labels to evaluate the accuracy rate though the model.
result=model.evaluate(test_data,test_labels)

print("the accuracy rate though the model:",result[1])
