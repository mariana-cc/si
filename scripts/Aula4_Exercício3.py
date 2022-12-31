import matplotlib.pyplot as plt
import numpy as np
import sys
main = "../src/si"
from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split
from si.neighbors.knn_regressor import KNNRegressor
from si.statistics.euclidean_distance import euclidean_distance

# Aula 4 - não foi possível realizar os exercícios em Jupyter Notebook

# Test the KNNRegressor object using the dataset cpu.csv (regression)

# Read csv file - cpu.csv
cpu_file = "../datasets/cpu.csv"
cpu = read_csv(filename=cpu_file, sep=",", features=True, label=True)

# Split dataset in training and testing sets
cpu_train, cpu_test = train_test_split(cpu)

# KNN Regressor
knn = KNNRegressor(k = 2, distance=euclidean_distance)
knn.fit(cpu_train)
predictions = knn.predict(cpu_test)
predictions
print("The predictions:")
print(predictions)

print("The score:")
score = knn.score(cpu_test)
print(score)




