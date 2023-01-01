main = "../src/si"
import numpy as np
from si.io.csv_file import read_csv
from si.model_selection.cross_validate import cross_validate
from si.linear_model.logistic_regression import LogisticRegression
from si.model_selection.randomized_search_cv import randomized_search_cv
from sklearn.preprocessing import StandardScaler

# Aula 7 - não foi possível realizar os exercícios em Jupyter Notebook

# - Exercise 1: Test the cross_validate using the breast-bin.csv dataset

# (1) Read csv file - breast-bin.csv
breast_file = "../datasets/breast-bin.csv"
breast = read_csv(filename=breast_file, sep=",", features=True, label=True)

# (2) Standardize features by removing the mean and scaling to unit variance.
breast.X = StandardScaler().fit_transform(breast.X)

# (3) Create Logistic Regression model
lr = LogisticRegression(l2_penalty= 1, alpha= 0.001, max_iter=2000)

# (4) Performs a cross validation with 5 folds
scores = cross_validate(lr, breast, cv=5)

# (5) What is the score obtained?
print(f"The cross validation scores: {scores}")

# - Exercise 2: Test the grid_search using the breast-bin.csv dataset

# (1) Read csv file - breast-bin.csv
breast_file = "../datasets/breast-bin.csv"
breast = read_csv(filename=breast_file, sep=",", features=True, label=True)

# (2) Standardize features by removing the mean and scaling to unit variance.
breast.X = StandardScaler().fit_transform(breast.X)

# (3) Create Logistic Regression model
lr2 = LogisticRegression(l2_penalty= 1, alpha= 0.001, max_iter=2000)

# (4) Perform a grid search with the following parameters
lr2_param = {'l2_penalty': [1, 10], 'alpha': [0.001, 0.0001], 'max_iter': [1000, 2000]}

# (5) cross validate the model with 3 folds
scores = cross_validate(lr2, breast, cv=3)

# (6) What is the score obtained?
print(f"The scores: {scores}")

# - Exercise 8: Test the randomized_search_cv using the breast-bin.csv dataset

# (1) Read csv file - breast-bin.csv
breast_file = "../datasets/breast-bin.csv"
breast = read_csv(filename=breast_file, sep=",", features=True, label=True)

# (2) Standardize features by removing the mean and scaling to unit variance.
breast.X = StandardScaler().fit_transform(breast.X)

# (3) Create Logistic Regression model
lr3 = LogisticRegression(l2_penalty= 1, alpha= 0.001, max_iter=2000)

# (4) Perform a randomized search with the following parameters
lr3_param = {'l2_penalty': np.linspace(start=1, stop=10, num=10),
             'alpha': np.linspace(start=0.001, stop=0.0001, num=100),
             'max_iter': np.linspace(start=1000, stop=2000, num=200)}

# (5.1) cross validate the model with 3 folds
scores_3 = randomized_search_cv(lr3, breast, lr3_param, cv=3)

# (5.2) cross validate the model with 10 folds
scores_10 = randomized_search_cv(lr3, breast, lr3_param, cv=10)

# (6) What is the score obtained?
print(f"The scores with 3 folds: {scores_3}")
print(f"The scores with 10 folds: {scores_10}")
