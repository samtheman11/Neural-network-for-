
from numpy import std
from Dataset import Data
from NeuralNetwork import NeuralNetwork
d=Data()
X,Y=d.dataset(24*365)
#makes Y 2D
Y = Y.reshape(-1,1)
normalX =(X-X.mean(axis=0))/X.std(axis=0)
normalY =(Y-Y.mean(axis=0))/Y.std(axis=0)
n=NeuralNetwork()
print()
n.train(100,normalX,normalY)
print()
n.train(100,X,Y)





