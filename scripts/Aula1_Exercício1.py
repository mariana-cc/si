main = "../src/si"
from si.io.csv_file import read_csv

# Aula 1 - não foi possível realizar os exercícios em Jupyter Notebook

# Exercise 1: NumPy array Indexing/Slicing

# (1) Read csv file - iris.csv
iris_file = "../datasets/iris.csv"
iris = read_csv(filename=iris_file, sep=",", features=True, label=True)

# (2) Selects the first independent variable and checks the size of the resulting array
first_variable = iris.X[:,0].shape
print("1.2.")
print(first_variable)

# (3) Selects the last 5 samples from the iris dataset. What is the average of the last 5 samples for each
# independent variable/feature?
last_5 = iris.X[-5:]

mean_last_5 = iris.X[-5:,].mean(axis=0)
# axis = 0 because its the mean by column/feature
print("1.3.")
print(mean_last_5)

# (4) Selects all samples from the dataset with value greater or equal to 1. Note that the resulting array must have
# only samples with values equal or greater than 1 for all features.
import numpy as np

array = iris.X
sup_eq_1 = np.all(array[:,] >= 1, axis = 1)
final_array = array[sup_eq_1]
print("1.4.")
print(final_array)

# (5) Select all samples with the class/label equal to 'Irissetosa'. How many samples do you get?
array = iris.X
irissetosa = (iris.y =="Iris-setosa")
array_2 = array[irissetosa]
print("1.5. Irissetosa")
print(array_2)
print(len(array_2))