import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def denormalize_data(normalized_data, original_min, original_max):
    return normalized_data * (original_max - original_min) + original_min

def scale_to_minus_one_one(x):
    return 2 * ((x - x.min()) / (x.max() - x.min())) - 1


data = pd.read_csv('US_City_Temp_Data.csv')
nb_rows, nb_columns = data.shape
T = np.zeros((nb_columns-1, nb_rows))           ## Array where each element is an array with the temperatures of a city over the years
t = np.array(data.iloc[:, 0])                   ## Array with the dates of temperature measurement
cities = np.array(data.columns[1:])             ## Array with the names of the cities
i = 0
for city in cities:
     T[i] = np.array(data[city])
     i += 1

data_prediction = pd.read_csv('array_predictions_new.csv')
nb_rows, nb_columns = data_prediction.shape
T_predictions = np.zeros((nb_rows, nb_columns))           
print(T_predictions.shape)
for i in range(len(cities)):
    T_predictions[i, :] = np.array(data_prediction.iloc[i, :])

T_predictions = np.delete(T_predictions, 1, axis=0)
T_predictions = np.delete(T_predictions, 12, axis = 0)
months_predictions = np.arange(1, 91)                    ## we have 899 dates in the data, each one for a month

T_time = T_predictions.T

# Define the neural network architecture
class MyNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(MyNeuralNetwork, self).__init__()
        self.layers = nn.ModuleList() # initialize the layers list as an empty list using nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size)) # Add the first input layer. The layer takes as input <input_size> neurons and gets as output <hidden_size> neurons
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size)) # Add hidden layers
        self.layers.append(nn.Linear(hidden_size, output_size)) # add output layer

    def forward(self, x):    # Function to perform forward propagation
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        x = self.layers[-1](x)
        return x


input_size = 2
num_layers = 18
hidden_size = 41
output_size = 1

y = [35.0844, 33.7490, 43.6150, 42.3601, 42.8864, 35.2271, 41.8781, 32.7767, 39.7392,
                 42.3314, 46.5891, 39.7684, 30.3322, 39.0997, 36.1699, 34.0522, 35.1495, 25.7617,
                 44.9778, 29.9511, 40.7128, 35.4676, 33.4484, 45.5051, 44.0805, 39.5296, 37.5407, 38.5816,
                 40.7608, 29.4241, 37.7749, 47.6062, 27.9506]

x = [-106.6504, -84.3880, -116.2023, -71.0589, -78.8784, -80.8431, -87.6298, -96.7970, -104.9903,
                  -83.0458, -112.0391, -86.1581, -81.6557, -94.5786, -115.1398, -118.2437, -90.0490, -80.1918,
                  -93.2650, -90.0715, -74.0060, -97.5164, -112.0740, -122.6750, -103.2310, -119.8138, -77.4360, -121.4944,
                  -111.8910, -98.4936, -122.4194, -122.3321,-82.4572]

x = normalize_data(x)
y = normalize_data(y)

# Convert numpy arrays to PyTorch tensors
x_tensor = torch.tensor(x, dtype=torch.float32).reshape(-1, 1)
y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
x_plot, y_plot = np.meshgrid(x, y)

# Combine x and y tensors into a single input tensor
input_tensor_first = torch.cat((x_tensor, y_tensor), dim=1)
model = MyNeuralNetwork(input_size, hidden_size, output_size, num_layers)

from sklearn.model_selection import train_test_split
# Split the training data into a new training set and a validation set
X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(
    x_tensor, y_tensor, T_time[0], test_size=0.1, random_state=42)

# Convert the variables to tensors so that they can be used in pytorch
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.float32)
Z_train = torch.tensor(Z_train, dtype=torch.float32)
Z_test = torch.tensor(Z_test, dtype=torch.float32)


####################################################""

#step 2
#Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001) # here we are using the Adam optimizer, to optimize model.parameters, but what is there inside this attribute?

###########################################################

# step 3
# Lists to store training loss and testing loss
train_loss_list = []
test_loss_list = []

# Training loop
input_train = torch.cat((X_train, Y_train), dim=1)
input_test = torch.cat((X_test, Y_test), dim=1)
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    output = model(input_train)
    loss = criterion(output, Z_train)

    # Compute loss on testing data. NOTE: we aren't gonna use the test loss for optimization!!!
    output_test = model(input_test)
    loss_test = criterion(output_test, Z_test)

    # Backprop and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Test loss: {loss_test.item():.4f}')


# Plot the NN performance:
with torch.no_grad():
    Z_pred = model(input_tensor_first)


fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection='3d')
ax3.scatter(X_test, Y_test, Z_test.numpy(), color='r', label='Testing')
ax3.scatter(X_train, Y_train, Z_train.numpy(), color='b', label='Training')
ax3.plot_surface(x_plot, y_plot, Z_pred, color = 'g', label = 'Predicted surface')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')
plt.show()

