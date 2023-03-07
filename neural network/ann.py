# This code is for paper Non-Convex Optimization via Non-Reversible Stochastic Gradient Langevin Dynamics
# should be run in the certain environment
from keras.datasets import imdb

from keras import models
from keras import layers
from keras import optimizers
import numpy as np
(train_data,train_labels),(test_data,test_labels) = imdb.load_data(num_words=10000)


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1 # one-hot
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

model = models.Sequential()
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]
model.add(layers.Dense(16, activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

sgsd=optimizers.SGLD(lr=0.1,momentum=0)
model.compile(loss='binary_crossentropy',
              optimizer=sgld,
              metrics=['accuracy'])

history = model.fit(partial_x_train,partial_y_train,epochs=50,batch_size=5000,validation_data=(x_val, y_val))