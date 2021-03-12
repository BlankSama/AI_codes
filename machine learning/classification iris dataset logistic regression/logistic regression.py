# importing the libraries
import numpy as np
import pandas as pd
from sklearn import datasets

#importing the dataset
iris_dataset = datasets.load_iris()
X = iris_dataset.data
y = iris_dataset.target

#splittting into dataset into test set and training set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2 , random_state=1)

#training the model  on training set
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train, y_train)

#testing the model on test set
y_pred = clf.predict(X_test)
y_porb = clf.predict_proba(X_test)

#printing the results
np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1)

#creating the confusion matrix and accuracy score
from sklearn.metrics import confusion_matrix, accuracy_score

cm= confusion_matrix(y_test,y_pred)
accuracy_score(y_test,y_pred)
print(cm)

#visualising the result
import matplotlib.pyplot as plt
plt.plot()