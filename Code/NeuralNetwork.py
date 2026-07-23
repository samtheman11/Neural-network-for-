
from turtle import back

import numpy as np
np.random.seed(10)
class NeuralNetwork:
    def __init__(self,input_size=3,hidden_layers=8  ,output_size=1,learning_rate=.01):
        self.input_size = input_size
        self.hidden_layers=hidden_layers
        self.output_size=output_size
        self.learning_rate = learning_rate 
        #wih = weights input to hidden | who = weights hidden to output 
        #bh = hidden layer bias | bo = output layer bias
        self.wih = np.random.randn(input_size,hidden_layers)*.1 #random small numbers for initialization
        self.who = np.random.randn(hidden_layers,output_size)*.1 #random small numbers for initialization
        self.bh = np.zeros((1,hidden_layers))#set to zero for initialization
        self.bo = np.zeros((1,output_size))#set to zero for initialization
    
    def ReLu(self,x): #using x bc ReLu is function
        return np.maximum(0,x)
    
    def frwd(self,X):
        z1 = X @ self.wih + self.bh #dot products
        hidden = self.ReLu(z1)
        z2 = hidden @ self.who + self.bo#dot products 
        return z1,hidden,z2#z2 is the prediction
    
    def loss(self,prediction, actual):#bigger number is bad
        return np.mean((prediction-actual)**2)#takes mean of squared error and makes our loss function to measure how wrong the network is
    
    def backpropagation(self,input,z1,hidden,z2,actual):
        #derivative of loss function / batch number
        dL_dP = 2 * (z2 - actual) / actual.shape[0] 
        #derivative of hidden weights to output
        dWho = hidden.T @ dL_dP
        #derivative of output bias
        dBo = dL_dP.sum(axis=0).reshape(1,-1)
        #derivative of hidden
        dHidden = dL_dP @ self.who.T
        #derivative of Relu
        d_Relu = np.where(z1>0,1,0)
        #derivative of dz1
        dz1 = dHidden * d_Relu
        #reshapes the input to become 2Darray and the -1 tells it to calc num of columns based
        #off original array elements
        R_input = input.reshape(1,-1)
        #derivative of wih need to transpose because the second number of first shape must be same to first number of second shape
        dWih = R_input.T @ dz1
        #derivative of hidden bias
        dBh = dz1.sum(axis=0).reshape(1,-1)
        self.who -= self.learning_rate * dWho
        self.wih -= self.learning_rate * dWih
        self.bo -= self.learning_rate * dBo
        self.bh -= self.learning_rate * dBh
  
    def train(self,epoch,X_data,Y_data):
        while epoch > 0 :
            #shuffles the order of rows index
            index = np.random.permutation(X_data.shape[0])
            #can use same index because both are same size
            X_data_shuffle = X_data[index]
            Y_data_shuffle = Y_data[index]
            epoch_losses = np.array([])
            for row in range(X_data_shuffle.shape[0]):
                z1,hidden,z2 = self.frwd(X_data_shuffle[row])
                self.backpropagation(X_data_shuffle[row],z1,hidden,z2,Y_data_shuffle[row])
                epoch_losses = np.append(epoch_losses, self.loss(z2, Y_data_shuffle[row]))
                #prints the number epoch and the avg loss
            print(f"epoch {epoch}: avg loss = {epoch_losses.mean():.4f}")
            #counts down till epoch gets to zero
            epoch = epoch - 1

    