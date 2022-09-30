#Aula 1 - 26/09
##Exercício 1

import pandas as pd
#from si.io import iris.csv


def read_csv(filename: str, sep: str=",", features: bool= True, label: bool=True):
    dataset = pd.read_csv('iris.csv', delimiter= sep)
    return dataset

def shape_1stvariable(dataset):
    if dataset.columns[0] is not None:
        return column[0].shape
    else:
        return None


def teste():
    print (read_csv('iris.csv'))
    print (shape_1stvariable(dataset))

teste()