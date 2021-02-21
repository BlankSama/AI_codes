#importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf

#Part 1 - Data Preprocessing

# Importing the dataset
#old code :  dataset = pd.read_excel('E:/AI/Deep Learning/ANN/Regression/project 1/Dataset/energy_output_data.xlsx')
dataset = pd.read_excel('Dataset/Folds5x2_pp.xlsx', engine='openpyxl')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Part 2 - Building the ANN

# Initializing the ANN
ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the second hidden layer

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

#Adding the output layer
#note: output layer doesnt have an activation function in output layer
ann.add(tf.keras.layers.Dense(units=1))

#Training the ANN

# Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Training the ANN model on the Training set
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

# Predicting the results of the Test set
y_pred = ann.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1),y_pred.reshape(len(y_pred),1) - y_test.reshape(len(y_test),1)),1))
# difference_vector = (y_pred.reshape(len(y_pred),1) - y_test.reshape(len(y_test),1))
# print(difference_vector)
#making confusion isnt suitable for regression task and error will be thrown

