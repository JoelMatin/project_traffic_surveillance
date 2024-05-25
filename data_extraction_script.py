import pandas as pd 
import numpy as np


data = pd.read_csv('US_City_Temp_Data.csv')
l = [1, 1, 1]
nb_rows, nb_columns = data.shape
T = np.zeros((nb_columns-1, nb_rows))
t = np.array(data.iloc[:, 0])
cities = np.array(data.columns[1:])
i = 0
for city in cities:
     T[i] = np.array(data[city])
     i += 1
print(cities)
print(T)
print(t)

