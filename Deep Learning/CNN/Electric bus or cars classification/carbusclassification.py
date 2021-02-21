# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 12:35:08 2021

@author: lefti
"""

#importing the libraries

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

# Preprocessing the data
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

training_set = train_datagen.flow_from_directory( 'dataset/Training_set',
                                                    target_size=(256,256),
                                                    batch_size =32,
                                                    class_mode='binary')
test_datagen =ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory('dataset/test',
                                                  target_size= (256,256),
                                                  batch_size =32,
                                                  class_mode = 'binary'
                                                  )

#building the Convolutional neural network
#initialise the cnn
cnn = tf.keras.models.Sequential()
#adding the first and input layer
cnn.add(tf.keras.layers.Conv2D(filters=32,
                               kernel_size=4,
                               activation='relu',
                               input_shape =[256,256,3]))
#adding the pooling to first layer
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

#add layer 2
cnn.add(tf.keras.layers.Conv2D(filters= 32,
                               kernel_size=4,
                               activation='relu'))
#add pooling to layer2
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
        
#flattening the layers
cnn.add(tf.keras.layers.Flatten())

#creating full connection
cnn.add(tf.keras.layers.Dense(units=150,activation='relu'))

#output layer
cnn.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))

#compiling the cnn
cnn.compile(optimizer = 'adam', loss='binary_crossentropy',metrics=['accuracy'])

#training the cnn
cnn.fit(x=training_set,validation_data = test_set,epochs=32)
