import numpy as np
import pandas as pd
from si.data.dataset import Dataset

def read_data_file(filename: str, sep: str=";", label: bool=True):
    rdata = np.genfromtxt('../Exercises/iris.csv', delimiter= sep)
    return rdata

def write_data_file (filename: str, sep: str=";", label: bool=True):
    wdata = np.savetxt('iris.csv', delimiter= sep)
    return wdata

