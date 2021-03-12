# Mnist dataset

#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib 
from sklearn.metrics import f1_score, roc_auc_score, log_loss
from sklearn.model_selection import cross_val_score, cross_validate
#importing the dataset
from mlxtend.data import loadlocal_mnist

#reading the training images
X_train, y_train = loadlocal_mnist(
            images_path='train-images.idx3-ubyte', 
            labels_path='train-labels.idx1-ubyte')
#reading the test set
X_test, y_test = loadlocal_mnist(
            images_path='t10k-images.idx3-ubyte', 
            labels_path='t10k-labels.idx1-ubyte')

image_data = X_train[1:100,:]
a=np.unique(y_train)

def displayImage(image_data):
    plt.imshow(image_data)
    plt.axis("off")


# for image in range(0,99):
#     some_digit_image = image_data[image].reshape(28,28)
#     displayImage(some_digit_image);

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

Classifier = Sequential()

Classifier.add(Dense(units=80,kernel_initializer='uniform',activation='relu'))
Classifier.add(Dense(units=100,kernel_initializer='uniform',activation='relu'))
Classifier.add(Dense(units=10,kernel_initializer='uniform',activation='softmax'))

##compiling the ann
Classifier.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

Classifier.fit(X_train,y_train,epochs=60,batch_size=1000)

y_pred = Classifier.predict(X_test)

#evaluating the model
#viewing the predicted and real values
test_loss=Classifier.evaluate(X_test,y_test,verbose=0)

f1_e25 = f1_score(y_test, Classifier.predict_classes(X_test), average='micro')
roc_e25 = roc_auc_score(y_test, Classifier.predict_proba(X_test), multi_class='ovo')
#create evaluation dataframe
stats_e25 = pd.DataFrame({'Test accuracy' :  round(test_loss[1]*100,3),
                      'F1 score'      : round(f1_e25,3),
                      'ROC AUC score' : round(roc_e25,3),
                      'Total Loss'    : round(test_loss[0],3)}, index=[0])
#print evaluation dataframe
print(stats_e25)

from joblib import dump
dump(Classifier,'digitIdentifier.joblib')



