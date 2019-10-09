# -*- coding: utf-8 -*-

#################################################
#此二元分類為將電影評論分類為正評或負評。利用Keras中的IMDB資料庫的資料來當作訓練資料。總共有25000個評論。
#
#
#
#
#B
#
#
#
##################################################
#從keras.datasets的package中匯入imdb。


from keras.datasets import imdb

#利用(train_data，train_labels)，(test_data，test_labels)的方式從imdb中讀取資料並放入這些位置。

(train_data,train_labels),(test_data,test_labels)=imdb.load_data(num_words=10000)

#匯入python中的numpy package並命名為np。

import numpy as np

#將函數'sequences'傳入維度為10000的雙層list中。

def vectorize_sequences(sequences,
                        dimension=10000):

#建立全為0的矩陣，其矩陣的形式為(len(sequences),dimension)

    results=np.zeros((len(sequences),
                      dimension))

#用enumerate()為每個子串列編號，編號會存在i，子串列存到sequence。

    for i, sequence in enumerate(sequences):

#接著將result[i]中的多個元素(以sequence串列的每個元素值為索引)設為1.0

        results[i,sequence]=1.

    return results

#將訓練資料放入我們上面假設有10000個維度的向量中。

x_train=vectorize_sequences(train_data)

#同樣邏輯，將測試資料放入向量中。

x_test=vectorize_sequences(test_data)

#當然，處理完訓練的資料後，緊接著也要把標籤的資料丟入向量中。將訓練標籤像量化。

y_train=np.asarray(train_labels).astype('float32')

#將測試標籤向量化。

y_test=np.asarray(train_labels).astype('float32')

from keras import models
from keras import regularizers
from keras import layers

#使用models的package中的sequential type，建立一個物件讓新增的神經網路層可以進行堆疊。

model=models.Sequential()

#輸入層是隱藏層

model.add(layers.Dense(64, 
                       activation='relu',
                       input_shape=(10000,)))

#64位元隱藏層

model.add(layers.Dense(64,
                       activation='relu',
                       kernel_regularizer=regularizers.l1_l2(l1=0.001,l2=0.001)))

#dropout64位元中的三個神經元

#輸出層
#輸出層

model.add(layers.Dense(1,activation='sigmoid'))

#指定損失函數，並進行compile。

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#取輸入資料的前10000個做為驗證資料集。

x_val=x_train[:10000]

#輸入資料的第10000個開始才是訓練資料。

partial_x_train=x_train[10000:]

#對應的，取標籤的前10000個做為驗證標籤。

y_val=y_train[:10000]

#從標籤的第10000個開始才是訓練資料的標籤。

partial_y_train=y_train[10000:]

#呼叫fit()開始訓練(使用partial_x_train輸入資料，partial_y_train標籤，20個訓練週期，一次batch使用512筆資料)，同時傳入validation set的資料即標籤。

history=model.fit(partial_x_train,
                  partial_y_train,
                  epochs=20,
                  batch_size=512,
                  validation_data=(x_val,y_val))

#匯入matplotlib.pyplot模組並命名為plt。

import matplotlib.pyplot as plt

#取得每次訓練的loss訓練損失分數並存成loss_values變數。

history_dict=history.history

history_dict.keys()

loss_values=history_dict['loss']

#取得每次驗證的val_loss驗證損失分數並指定給val_loss_values變數。

val_loss_values=history_dict['val_loss']

#len(loss_values)項目個數為20，範圍從1到21的週期。

epochs=range(1,len(loss_values)+1)

#以'b'指定用藍色線條畫出x軸為訓練週期，y軸為驗證損失分數的圖表。

plt.plot(epochs,
         loss_values,
         'bo',
         label='Training loss')

#以'bo'指定用藍色點畫出x軸為訓練週期，y軸為訓練損失分數的圖表。

plt.plot(epochs,
         val_loss_values,
         'b',
         label='Valadation loss')

#顯示圖表的標題。

plt.title('Training and validation loss')

#將X軸命名為epochs

plt.xlabel('Epochs')

#將Y軸命名為loss

plt.ylabel('Loss')

#追加每個施出圖表的圖像名稱

plt.legend()

#顯示圖表

plt.show()

#繪製訓練和驗證的的準確定，plt.clf()是把上述作圖的變數給清除重新設定一個圖表。

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
