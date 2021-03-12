#immporting the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

#importing the dataset
iris_dataset = datasets.load_iris()
X= iris_dataset.data
y= iris_dataset.target

##no need of categorical encoding and feature scaling

## splitting the dataset into test and train
from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test = train_test_split(X,y, random_state = 0, test_size =0.2)

from sklearn.neighbors import KNeighborsClassifier
clf=KNeighborsClassifier()
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)

#viewing the result
print(np.concatenate((y_test.reshape(len(y_test),1), y_pred.reshape(len(y_pred),1)),1))

#making the confusion matrix and accuracy score
from sklearn.metrics import confusion_matrix , accuracy_score
cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)


