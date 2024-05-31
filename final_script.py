import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt





import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.interpolate import griddata

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def denormalize_data(normalized_data, original_min, original_max):
    return normalized_data * (original_max - original_min) + original_min

def scale_to_minus_one_one(x):
    return 2 * ((x - x.min()) / (x.max() - x.min())) - 1

def create_grid(x, y): 
    nx = len(x)
    ny = len(y)
    grid = np.array([[(x[i], y[j]) for j in range(ny)] for i in range(nx)])
    return grid

def find_temp(x,y, xs, ys): 
    closest_x_index = np.argmin(np.abs(xs) - abs(x))
    closest_y_index = np.argmin(np.abs(ys) - abs(y))
    print()
    return closest_x_index, closest_y_index

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


y = np.array([35.0844, 33.7490, 43.6150, 42.3601, 42.8864, 35.2271, 41.8781, 32.7767, 39.7392,
                 42.3314, 46.5891, 39.7684, 30.3322, 39.0997, 36.1699, 34.0522, 35.1495, 25.7617,
                 44.9778, 29.9511, 40.7128, 35.4676, 33.4484, 45.5051, 44.0805, 39.5296, 37.5407, 38.5816,
                 40.7608, 29.4241, 37.7749, 47.6062, 27.9506])

x = np.array([-106.6504, -84.3880, -116.2023, -71.0589, -78.8784, -80.8431, -87.6298, -96.7970, -104.9903,
                  -83.0458, -112.0391, -86.1581, -81.6557, -94.5786, -115.1398, -118.2437, -90.0490, -80.1918,
                  -93.2650, -90.0715, -74.0060, -97.5164, -112.0740, -122.6750, -103.2310, -119.8138, -77.4360, -121.4944,
                  -111.8910, -98.4936, -122.4194, -122.3321,-82.4572])

ti = 7
T_ti = T_time[ti]
# x_range = np.linspace(int(x.max()), int(x.min()),  int(x.max())-int(x.min())+1)
# y_range = np.linspace(int(y.min()), int(y.max()), int(y.max())-int(y.min())+1)

x_range = np.sort(x)
y_range = np.sort(y)
print(x_range)

# T_map = np.zeros((len(x_range),  len(y_range)))
# for k in range(len(T_ti)): 
#     i  = np.where(x_range == x[k])[0]
#     j  = np.where(y_range == y[k])[0]
#     T_map[i, j] = T_ti[k] 

# grid = create_grid(x_range, y_range)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for i in range(len(x_range)): 
#     for j in range(len(y_range)): 
#         if (T_map[i][j] != 0):
#           ax.scatter(x_range[i], y_range[j], T_map[i, j])

# plt.show()


# This is the Gaussian RBF
def Gauss_RBF(x, y, x_r=0, y_r=0, c_r=0.1):
    dx = x - x_r  # Get distance
    dy = y - y_r
    phi_r = np.exp(-c_r**2 * ( (dx**2)/15 + (dy**2)/7))
    return phi_r

# Define the 3D identity matrix function
def identity3D(n):
    return np.eye(n).reshape((n, n))

# Number of bases
n_b = 50

# Define the basis locations
x_b = np.linspace(x.min(), x.max(), n_b)
y_b = np.linspace(y.min(), y.max(), n_b)

def PHI_Gauss_XY(x_in, y_in, x_b, y_b, c_r=0.8):
    n_x = np.size(x_in)
    n_b = len(x_b)
    Phi_XY = np.zeros((n_x, (n_b + 1) * (n_b + 1)))  # Initialize Basis Matrix
    # Add a constant term
    Phi_XY[:, 0] = 1  # The first basis is a constant term
    # Loop to prepare the basis matrices (inefficient)
    for i in range(n_b):
        for j in range(n_b):
            # Prepare all the terms in the basis
            Phi_XY[:, i * (n_b + 1) + j + 1] = Gauss_RBF(x_in, y_in, x_r=x_b[i], y_r=y_b[j], c_r=c_r)
    return Phi_XY



def Boot_strap_RBF_Train(x, y, z, x_b, y_b, c_r=0.6, alpha=0.001, n_e=500, tp=0.2):
    '''
    x, y, z training data
    x_b, y_b where to put gaussian
    Bootstrap function that will train n_e RBF models using Gaussian RBFs
    located at x_b, y_b, having shape factor c_r, using Ridge regression with alpha
    regularization and keeping a tp fraction for testing.
    '''
    J_i = np.zeros(n_e)  # in-sample error of the population
    J_o = np.zeros(n_e)  # out-of-sample error of the population
    n_b = len(x_b)
    w_e = np.zeros(((n_b + 1) * (n_b + 1), n_e))  # Distribution of weights

    # Loop over the ensemble
    for j in range(n_e):
        # Split the data
        xs, xss, ys, yss, zs, zss = train_test_split(x, y, z, test_size=tp)
        # Construct phi_x_s
        Phi_x_s = PHI_Gauss_XY(xs, ys, x_b, y_b, c_r=c_r)
        Phi_x_ss = PHI_Gauss_XY(xss, yss, x_b, y_b, c_r=c_r)

        # Compute w
        H = Phi_x_s.T @ Phi_x_s + alpha * np.eye((n_b + 1) * (n_b + 1))
        # Train Model using np.linalg.solve to solve the linear system H * w_s = Phi_x_s.T @ zs
        w_s = np.linalg.solve(H, Phi_x_s.T @ zs)
        # Assign vectors to the distributions
        w_e[:, j] = w_s
        # Make in-sample prediction
        z_p_s = Phi_x_s @ w_s
        # In-sample error
        J_i[j] = 1 / len(xs) * np.linalg.norm(z_p_s - zs)**2
        # Make out-of-sample prediction (and errors)
        z_p_ss = Phi_x_ss @ w_s
        # Out-of-sample error
        J_o[j] = 1 / len(xss) * np.linalg.norm(z_p_ss - zss)**2
        print(j)
    return J_i, J_o, w_e

def Ensamble_RBF(xg, yg, x_b, y_b, c_r, w_e, sigma_z):
    '''
    Make ensemble prediction of models of RBF models using Gaussian RBFs
    located at x_b, y_b, having shape factor c_r from an ensemble of weights w_e
    on data that is estimated to have a random noise component sigma_z
    '''
    n_e = np.shape(w_e)[1]  # get the n_e from the weight population
    n_p = len(xg)  # number of points where predictions are required
    z_pop = np.zeros((n_p, n_e))  # Prepare the population of predictions

    # Prepare the Phi matrix on xg:
    Phi_XY_ss = PHI_Gauss_XY(xg, yg, x_b, y_b, c_r=c_r)

    for e in range(n_e):  # could have been just matrix multiplication
        z_pop[:, e] = Phi_XY_ss @ w_e[:, e]

    # Get statistics over the ensemble
    # Mean prediction
    z_e = np.mean(z_pop, axis=1)
    # The ensemble variance:
    Var_Z_model = np.std(z_pop, axis=1)**2
    # So the global uncertainty, considering the aleatoric is:
    Unc_z = np.sqrt(Var_Z_model + sigma_z)
    return z_e, Unc_z

# Assume x, y, and T_ti are your input data arrays
# Example:
# x = np.random.rand(35)
# y = np.random.rand(35)
# T_ti = np.random.rand(35)

# Training the model
J_i, J_o, w_e = Boot_strap_RBF_Train(x, y, T_ti, x_b, y_b, c_r=1, alpha=0.00001, n_e=20, tp=0.2)
sigma_z_estimate = np.sqrt(np.mean(J_i))  # Estimate the data variance

# Define the grid for predictions
x_ss = np.linspace(x.min(), x.max(), 200)
y_ss = np.linspace(y.min(), y.max(), 200)
x_grid, y_grid = np.meshgrid(x_ss, y_ss)
x_flat = x_grid.flatten()
y_flat = y_grid.flatten()

# Make predictions
z_e, Unc_z = Ensamble_RBF(x_flat, y_flat, x_b, y_b, c_r=1, w_e=w_e, sigma_z=sigma_z_estimate)
# z_e2, Unc_z2 = Ensamble_RBF(y_flat, x_flat, x_b, y_b, c_r=1, w_e=w_e, sigma_z=sigma_z_estimate)

# z_e3 = (z_e+z_e2)/2
# Reshape the predictions to the grid shape
z_e_grid = z_e.reshape(y_grid.shape)
Unc_z_grid = Unc_z.reshape(y_grid.shape)
coord = find_temp(-81.22, 28.32, x_ss, y_ss)
print(z_e[coord[0] * (n_b + 1) + coord[1] + 1])


# Phi_XY_s = PHI_Gauss_XY(x_flat, y_flat, x_b, y_b, c_r=0.1)
# fig3 = plt.figure()
# ax3 = fig3.add_subplot(111, projection='3d')
# ax3.scatter(x_flat, y_flat, Phi_XY_s[:, 500])

# Plotting the results
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.plot_surface(x_grid, y_grid, z_e_grid, cmap='viridis')
plt.xlabel('Longitute')
plt.ylabel('Latitude')
ax2.scatter(x, y, T_ti, color = 'r')
plt.show()

# Optionally, to see the uncertainty:
plt.imshow(Unc_z_grid, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='viridis')
plt.colorbar(label='Uncertainty')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
