#從keras.datasets的package中匯入reuters的資料集#從keras.datasets的package中匯入reuters的資料集

from keras.datasets import reuters

#從reuters資料集中讀取訓練資料，訓練標籤，測試資料，測試標籤

(train_data,train_labels),(test_data,test_labels)=reuters.load_data(num_words=10000)

#將函數'sequences'傳入維度為10000的雙層list中。
import numpy as np

def vectorize_sequences(sequences,dimension=10000):

#建立全為0的矩陣，其矩陣的形式為(len(sequences),dimension)

    results=np.zeros((len(sequences),dimension))

#用enumerate()為每個子串列編號，編號會存在i，子串列存到sequence。

    for i, sequence in enumerate(sequences):

#接著將result[i]中的多個元素(以sequence串列的每個元素值為索引)設為1.0

        results[i,sequence]=1.

    return results

x_train=vectorize_sequences(train_data)

#同樣邏輯，將測試資料放入向量中。

x_test=vectorize_sequences(test_data)

def to_one_hot(labels,dimension=46):
    results=np.zeros((len(labels),dimension))
    for i, label in enumerate(labels):
        results[i,label]=1.
    return results

one_hot_train_labels=to_one_hot(train_labels)

one_hot_test_labels=to_one_hot(test_labels)

from keras import models

from keras import layers

#使用models的package中的sequential type，建立一個物件讓新增的神經網路層可以進行堆疊。

model=models.Sequential()

#輸入層是隱藏層

model.add(layers.Dense(32,
                       activation='relu',
                       input_shape=(10000,)))

#隱藏層

model.add(layers.Dense(32,
                       activation='relu'))

#輸出層

model.add(layers.Dense(46,
                       activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

x_val=x_train[:1000]

#輸入資料的第10000個開始才是訓練資料。

partial_x_train=x_train[1000:]

#對應的，取標籤的前10000個做為驗證標籤。

y_val=one_hot_train_labels[:1000]

#從標籤的第10000個開始才是訓練資料的標籤。

partial_y_train=one_hot_train_labels[1000:]

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

