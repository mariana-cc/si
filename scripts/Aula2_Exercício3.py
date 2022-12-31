main = "../src/si"
from si.io.csv_file import read_csv
from si.statistics.f_classification import f_classification
from si.feature_seletion.select_percentile import SelectPercentile

# Aula 2 - não foi possível realizar os exercícios em Jupyter Notebook

# Test the class SelectPercentile using the dataset iris.csv (classification)

# Read csv file - iris.csv
iris_file = "../datasets/iris.csv"
iris = read_csv(filename=iris_file, sep=",", features=True, label=True)
iris_df = iris.print_dataframe()

# SelectPercentile
selector = SelectPercentile(percentile = 20, score_func = f_classification)
selector.fit_transform(iris_df)
selector.f_value
selector.p_value
transformed_df = selector.fit_transform(iris_df)
transformed_df.shape()