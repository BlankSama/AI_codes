#importing the libraries
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.parallel
from torch.autograd import variable

#import the dataset
movies = pd.read_csv('ml-1m/movies.dat',sep='::' , header=None , engine='python' , encoding='latin-1')
users = pd.read_csv('ml-1m/users.dat',sep='::' , header=None , engine='python' , encoding='latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat',sep='::' , header=None , engine='python' , encoding='latin-1')

#importing the training set and test set
training_set = pd.read_csv('ml-100k/u1.base', sep='\t')
training_set = np.array(training_set, dtype='int')

test_set = pd.read_csv('ml-100k/u1.test', delimiter='\t')
test_set = np.array(test_set, dtype='int')

#getting the number the users and movies
num_users = int(max(max(training_set[:,0]),max(test_set[:,0])))
num_movies = int(max(max(training_set[:,1]),max(test_set[:,1])))

#coverting the data into an 2d array with users as row and movies as column
#creating function for data merging
def convert(data):
    #empty array creation
    new_data =[]
    for id_users in range(1,num_users+1):
        #getting all movies users id_user has rated
        id_movies = data[:,1][data[:,0]== id_users]
        #getting all the ratings user user_id has given
        id_ratings = data[:,2][data[:,0]== id_users]
        #creating an array to hold to ratings, intialised as array of zeros
        ratings = np.zeros(num_movies)
        #replacing ratings in user_id rating array by actual ratings
        ratings[id_movies-1]=id_ratings
        #merging individual rating array into a 2d array
        new_data.append(list(ratings))
    #returning list of list
    return new_data

#converting training set and test set as 2d array using convert function
training_set=convert(training_set)
test_set=convert(test_set)

#converting test set and training set in pytorch tensors
training_set= torch.FloatTensor(training_set)
test_set= torch.FloatTensor(test_set)

#converting ratings into binary ratings liked(1) or not liked(0)
training_set[training_set[:,1] == 0] = -1
training_set[training_set[:,1] == 1] = 0
training_set[training_set[:,1] == 2] = 0
training_set[training_set[:,1] >= 3] = 1

test_set[test_set[:,1] == 0] = -1
test_set[test_set[:,1] == 1] = 0
test_set[test_set[:,1] == 2] = 0
test_set[test_set[:,1] >= 3] = 1

#Creating the archetecture of neural network
class RBM():
    def __init__(self, nv, nh):
        self.W = torch.randn(nh,nv)
        self.a = torch.randn(1,nh)
        self.b = torch.randn(1,nv)
    
    def sample_h(self,x): 
        wx = torch.mm(x,self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v,torch.bernoulli(p_h_given_v)
    
    def sample_v(self,y):
        wy = torch.mm(y,self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    
    def train(self, v0, vk, ph0, phk):
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b += torch.sum((v0 - vk ),0)
        self.a += torch.sum((ph0-phk),0)
        
nv= len(training_set[0]);
nh= 100
batch_size = 100
rbm= RBM(nv,nh)
        
#Training the RBM
nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
  train_loss = 0
  s = 0.
  for id_user in range(0, num_users - batch_size, batch_size):
    vk = training_set[id_user : id_user + batch_size]
    v0 = training_set[id_user : id_user + batch_size]
    ph0,_ = rbm.sample_h(v0)
    for k in range(10):
      _,hk = rbm.sample_h(vk)
      _,vk = rbm.sample_v(hk)
      vk[v0 < 0] = v0[v0 < 0]
    phk,_ = rbm.sample_h(vk)
    rbm.train(v0, vk, ph0, phk)
    train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0]))
    s += 1.
  print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))
  
#Testing the RBM        
test_loss = 0
s = 0.
for id_user in range(num_users):
    v = training_set[id_user : id_user + 1]
    vt = test_set[id_user : id_user + 1]
    if len((vt[vt>=0])> 0):    
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)         
        test_loss += torch.mean(torch.abs(vt[vt >= 0] - v[vt >= 0]))
        s += 1.  
print('test_loss: '+ str(test_loss/s))
        
        
        