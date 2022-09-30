import numpy as np
import pandas as pd
from si.data.dataset import Dataset
def read_csv(filename: str, sep: str=";", features: bool= True, label: bool=True):
    dataframe = pd.read_csv('../Exercises/iris.csv', delimiter= sep)
    return dataframe

def write_csv_file(filename: str, dataset: Dataset, sep: str=";", features: bool= True, label: bool=True):
    #...
