# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 17:26:54 2020

@author: rowe1
"""

import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

train = pd.read_csv('training_data.csv', header = None)
test = pd.read_csv('test_data.csv', header = None)
valid = pd.read_csv('validation_data.csv', header = None)

#Sample Size Reduction
#train = train.iloc[:len(train)//2]

y_train = train[1860]
y_test = test[1860]
y_valid = valid[1860]

X_train = train.drop(1860, axis = 1)
X_test = test.drop(1860, axis = 1)
X_valid = valid.drop(1860, axis = 1)

# =============================================================================
# CUSTOM ACTIVATION
# =============================================================================

def half_sigmoid(z):
    return 0.5*(1+(1/(1+tf.math.exp(-z))))
                
# =============================================================================
# BUILD MODEL
# =============================================================================
model = keras.Sequential()
model.add(keras.layers.Conv2D(filters=200, kernel_size=9, strides=1, padding='same', 
                              activation='relu', input_shape=[30, 62, 1]))
model.add(keras.layers.Conv2D(filters=200, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Conv2D(filters=200, kernel_size=3, strides=1, padding='same'))
model.add(keras.layers.Conv2D(filters=200, kernel_size=3, strides=1, padding='same'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.GlobalAveragePooling2D())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(1, activation = half_sigmoid))
    

#keras.applications.resnet50.ResNet50(weights='imagenet')

# =============================================================================
# COMPILE MODEL
# =============================================================================
optimizer = keras.optimizers.Nadam(lr=1E-6, decay=0.001)
loss = keras.losses.mae
model.compile(optimizer=optimizer, loss=loss, metrics=['mae'])

# =============================================================================
# FIT MODEL
# =============================================================================
patience=5 #how many epochs to wait without improvement before stopping
cb_earlystop = keras.callbacks.EarlyStopping(patience=patience)
cb_checkpoint = keras.callbacks.ModelCheckpoint('./models/cnn_greed_model_3200.h5', save_best_only=True)
callbacks=[cb_earlystop, cb_checkpoint]

X_train = np.array(X_train)
X_valid = np.array(X_valid)
X_test = np.array(X_test)

X_train = np.reshape(X_train, [-1, 30, 62, 1])
X_valid = np.reshape(X_valid, [-1, 30, 62, 1])
X_test = np.reshape(X_test, [-1, 30, 62, 1])
#or equivalently X_train = X_train[...,np.newaxis]
print('passed reshape')
h = model.fit(X_train, y_train, batch_size = 64, epochs = 100, 
              callbacks=callbacks, validation_data = (X_valid, y_valid))

plt.close('all')
epochs = h.epoch
plt.figure(dpi=300)
for m in h.history:
    if 'mae' in m:
        plt.plot(epochs,h.history[m], label=m)
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.legend()





