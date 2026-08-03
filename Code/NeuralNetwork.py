

import numpy as np
np.random.seed(1)
class NeuralNetwork:
    def __init__(self,input_size=7,hidden_layers=8,output_size=1,learning_rate=.01):
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
        #derivative of loss function 
       
        dL_dP = 2 * (z2 - actual) 
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
            # can use same index because both are same size
            X_data_shuffle = X_data[index]
            Y_data_shuffle = Y_data[index]
            epoch_losses = np.array([])
            for row in range(X_data_shuffle.shape[0]):
                z1,hidden,z2 = self.frwd(X_data_shuffle[row])
                self.backpropagation(X_data_shuffle[row],z1,hidden,z2,Y_data_shuffle[row])
                epoch_losses = np.append(epoch_losses, self.loss(z2, Y_data_shuffle[row]))
            #counts down till epoch gets to zero
            epoch = epoch - 1
    #converts all the unnormilized data into a 2d array of normilized data 
    def convert_input(self,hour,day,day_of_year,temp,Y):
        #Network can't sense how close days 0 and 6 and... so you need to make the numbers circular 
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        day_sin  = np.sin(2 * np.pi * day / 7)
        day_cos  = np.cos(2 * np.pi * day / 7)
        year_sin = np.sin(2 * np.pi * day_of_year / 365)
        year_cos = np.cos(2 * np.pi * day_of_year / 365)
        norm_temp_c = (temp-np.mean(temp)) / np.std(temp)
        X = np.column_stack([hour_sin, hour_cos, day_sin, day_cos, year_sin, year_cos,norm_temp_c])
        #splits the data into training and non training
        split_index = int(len(X)*.7)
        X_train, X_test = X[:split_index], X[split_index:]
        Y_train_raw, Y_test_raw = Y[:split_index], Y[split_index:]
        # need to normalize data using Z score normalization to help the network run better
        Y_train = (Y_train_raw - Y_train_raw.mean()) / Y_train_raw.std()
        Y_test  = (Y_test_raw  - Y_train_raw.mean()) / Y_train_raw.std()
        return X_train, Y_train, X_test, Y_test, Y_train_raw, Y_test_raw