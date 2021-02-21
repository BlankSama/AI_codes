#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
X = sc.fit_transform(X)

#Training the Self orgainsing maps
from minisom import MiniSom
som = MiniSom(x=10, y=10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)
 
#visualising the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers=['x','o']
colors=['r','g']

for i,x in enumerate(X):
    winner = som.winner(x)
    plot(winner[0] +0.5,
         winner[1]+0.5,
         markers[Y[i]],
         markeredgecolor = colors[Y[i]],
         markerfacecolor = 'none',
         markersize = 10,
         markeredgewidth = 2)
    
show()

# finding and printing the frauds
mappings = som.win_map(X)
frauds= np.concatenate((mappings[(9,2)], mappings[(5,1)]),axis=0)
frauds = sc.inverse_transform(frauds)

# Printing the frauds

print('frauds customer IDs are :')
for i in frauds[:,0]:
    print(int(i))
 