main = "../src/si"
from si.io.csv_file import read_csv
from si.decomposition.pca import PCA

# Aula 3 - não foi possível realizar os exercícios em Jupyter Notebook

# Test the object PCA using the iris.csv dataset (classification)

# Read csv file - iris.csv
iris_file = "../datasets/iris.csv"
iris = read_csv(filename=iris_file, sep=",", features=True, label=True)

# PCA
pca_iris = PCA(n_components=2)
pca_iris.fit(iris)
pca_iris.transform(iris)
print(pca_iris.explained_variance)
iris_reduced = pca_iris.fit_transform(iris)
print(iris_reduced)
