## Machine learning 

# importing the libraries
import numpy as np
import pandas as pd
from sklearn import datasets,linear_model
import matplotlib.pyplot as plt
#importing the dataset
dataset = datasets.load_diabetes()
datset_data = dataset.data
target_set = dataset.target
X = datset_data[:,:]
y = target_set
x_shape = np.shape(X)

#splitting the datset into test and trainset
from sklearn.model_selection import train_test_split
X_train,X_test, y_train , y_test = train_test_split(X,y,test_size =0.2, random_state=0)

#no need to perform feature scaling and categorial encoding




#using various models on the dataset

linear_regressor = linear_model.LinearRegression()
linear_regressor.fit(X_train,y_train)

y_pred = linear_regressor.predict(X_test)

#calculating root mean square error
from sklearn.metrics import mean_squared_error
print("root mean square error is : \n" + str(mean_squared_error(y_test, y_pred)))

print(np.concatenate((y_pred.reshape(len(y_pred),1),
                y_test.reshape(len(y_test),1)),1))

#visualising the result
plt.scatter(X_test[:,0],y_test)
plt.plot(X_test[:,0],y_pred)
plt.title('diabetes')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
