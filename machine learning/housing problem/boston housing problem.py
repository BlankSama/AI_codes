#housing problem

## importing the libraries
import numpy as np
import pandas as pd
# pd.__version__

#importing the dataset
dataset = pd.read_excel('hosuing data.xlsx',engine='openpyxl')


# making histogram
dataset['CHAS'].value_counts()
# image = dataset.hist(bins=30, figsize=(10,10))

#viewing corelation matrix
corelation_matrics = dataset.corr()
print(corelation_matrics['CHAS'])

#feature scaling
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train,y_test = train_test_split(X,y, random_state= 1, test_size=0.2)

#since the number of values of feature CHAR is low and there might be a possibility that the value binary value might be present in the test or train set and will not equally distributed we will use 

from sklearn.model_selection import StratifiedShuffleSplit
shufflesplit = StratifiedShuffleSplit(n_splits = 1,test_size=0.2, random_state=1)
for train_index, test_index in shufflesplit.split(dataset,dataset['CHAS']):
    strartified_train = dataset.loc[train_index]
    strartified_test = dataset.loc[test_index]

#taking out the dependent and indepenedent variable
X_train= strartified_train.iloc[:,:-1].values
y_train=strartified_train.iloc[:,-1].values
X_test= strartified_test.iloc[:,:-1].values
y_test=strartified_test.iloc[:,-1].values


#creating a pipeline of data that do feature scaling and simple imputing
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

pipeline_dummy = Pipeline([
                        ('imputer',SimpleImputer(strategy='median')),
                        ('featureScaling',StandardScaler()),
                        ])

ready_train_data = pipeline_dummy.fit_transform(X_train)
ready_test_data = pipeline_dummy.fit_transform(X_test)
y_train = np.array(y_train)

#training the model on training data using various regression algorithm
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

clf1 = LinearRegression()
clf1.fit(ready_train_data,y_train)
y_pred_linear = clf1.predict(ready_test_data)

clf2 = DecisionTreeRegressor()
clf2.fit(ready_train_data,y_train)
y_pred_decision = clf2.predict(ready_test_data)

clf3=RandomForestRegressor()
clf3.fit(ready_train_data,y_train)
y_pred_random = clf3.predict(ready_test_data)


#checking for the accuracy of the regression algorithm using root mean squared error
from sklearn.metrics import mean_squared_error
mse_lr = mean_squared_error(y_test, y_pred_linear)
mse_dt = mean_squared_error(y_test, y_pred_decision)
mse_rf = mean_squared_error(y_test, y_pred_random)
print('rmse_lr: ', np.sqrt(mse_lr))
print('rmse_dt: ', np.sqrt(mse_dt))
print('rmse_rf: ', np.sqrt(mse_rf))

#using advance evaluation technqiues cross validation
from sklearn.model_selection import cross_val_score
cvs_lr = cross_val_score(clf1,ready_train_data,y_train,scoring = 'neg_mean_squared_error' , cv=15)
rmse_score_cross_val_lr = np.sqrt(-cvs_lr)
cvs_dt = cross_val_score(clf2,ready_train_data,y_train,scoring = 'neg_mean_squared_error' , cv=15)
rmse_score_cross_val_dt = np.sqrt(-cvs_dt)
cvs_rf = cross_val_score(clf3,ready_train_data,y_train,scoring = 'neg_mean_squared_error' , cv=15)
rmse_score_cross_val_rf = np.sqrt(-cvs_rf)


def rmsescore(scores):
    print('score :', str(scores))
    print('mean :' ,scores.mean())
    print('standard deviation :' , scores.std())

print(rmsescore(rmse_score_cross_val_lr))
print(rmsescore(rmse_score_cross_val_dt))
print(rmsescore(rmse_score_cross_val_rf))

#saving the model using joblib
from joblib import dump, load
dump(clf1,'linearRegressor.joblib')
dump(clf2,'DecisionTreeRegressor.joblib')
dump(clf3,'RandomForest.joblib')


