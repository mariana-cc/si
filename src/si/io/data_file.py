import numpy as np
from si.data.dataset_module import Dataset

def read_data_file(filename: str, sep: str = None, label: bool = False):
    """
    It reads a file into a Dataset object.
    :param filename: str, path to file to read
    :param sep: str, separator used in the file
    :param label: bool, if the file has a label
    :return: Dataset, dataset object
    """
    r_data = np.genfromtxt(filename, delimiter= sep)
    if label is True: #se o ficheiro tem label definida
        X = r_data[:,:-1] #todas as colunas, exceto a última
        y = r_data[:,-1] #última coluna
    else: #se o ficheito não tem label definida
        X = r_data
        y = None #não tem labels
    return Dataset(X,y) #cria dataset com os dados

def write_data_file (filename: str, dataset: Dataset, sep: str = None, label: bool = False):
    """
    It writes a Dataset object to a data file.
    :param filename: str, path to file to read
    :param dataset: Dataset, dataset object
    :param sep: str, separator used in the file
    :param label: bool, if it write a file with a label
    :return: None
    """
    if label is True: #se temos label definida
        data = np.hstack(dataset.X, dataset.y.reshape(-1,1)) #hstack - reune os arrays na horizontal
        #reshape(-1,1) - passa o array para uma só coluna
    else:
        data = dataset.X
    return np.savetxt(filename, data, delimiter= sep) #guarda o ficheiro

