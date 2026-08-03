import numpy as np
from NeuralNetwork import NeuralNetwork
import matplotlib.pyplot as plt
import json
data = np.genfromtxt("merged_data.csv", delimiter=",", skip_header=1)
hour = data[:, 0]
day = data[:, 1]
day_of_year = data[:, 2]
temp = data[:, 3]
Y = data[:, 4].reshape(-1,1)
n = NeuralNetwork(input_size=7,learning_rate=0.01)
X_train, Y_train, X_test, Y_test, Y_train_raw, Y_test_raw = n.convert_input(hour, day, day_of_year,temp, Y)
n.train(5000,X_train,Y_train)
z1,hidden,z2 = n.frwd(X_test)
real = (z2*(np.std(Y_train_raw)))+ np.mean(Y_train_raw)
print(n.loss(z2,Y_test))
rmse_real = np.sqrt(n.loss(real,Y_test_raw))
percent_error = rmse_real / np.mean(Y_test_raw) * 100
print(percent_error)
weights = {
    "wih": n.wih.tolist(),
    "who": n.who.tolist(),
    "bh": n.bh.tolist(),
    "bo": n.bo.tolist(),
    "temp_mean": float(np.mean(temp)),
    "temp_std": float(np.std(temp)),
    "Y_mean": float(Y_train_raw.mean()),
    "Y_std": float(Y_train_raw.std()),
}
 
with open("weights.js", "w") as f:
    f.write("const WEIGHTS = ")
    json.dump(weights, f)
    f.write(";\n")
 
print("Saved weights.js -- place it in the same folder as predictor.html")