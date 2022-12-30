import pandas as pd
from si.data.dataset_module import Dataset

def read_csv(filename: str, sep: str = ",", features: bool = False, label: bool = False) -> Dataset:
    """
    It reads a csv file.
    :param filename: str, name of the csv file to read
    :param sep: str, separator used in the file, by default ','
    :param features: boolean, value that indicates if the file has defined features
    :param label: boolean, value that indicates if the file has defined labels
    :return: Dataset, dataset object
    """
    data = pd.read_csv(filename, sep= sep)
    if features and label: #se temos as features e labels
        features = data.columns[:-1] #todas as colunas exceto a última
        label = data.columns[-1]
        X = data.iloc[:, :-1].to_numpy() #queremos começar na primeira linha #passa para numpy
        y = data.iloc[:, -1].to_numpy()

    elif features and not label: #se temos as features, mas não as labels
        features = data.columns
        X = data.to_numpy()
        y = None

    elif not features and label: #se não temos as fetaures, mas temos as labels
        X = data.iloc[:, :-1].to_numpy()
        y = data.iloc[:, -1].to_numpy()
        features = None
        label = None

    else: #quando não temos nem as features nem as labels
        X = data.to_numpy()
        y = None
        features = None
        label = None

    return Dataset(X, y, features = features, label = label)

def write_csv_file(filename: str, dataset: Dataset, sep: str = ",", features: bool = False, label: bool = False):
    """
    It writes a dataset to a csv file.
    :param filename: str, name of the csv file that is going to be written
    :param dataset: Dataset, the dataset that is going to be written
    :param sep: separator value that is used to separate the data
    :param features: boolean, value that indicates if the file has defined features
    :param label: boolean, value that indicates if the file has defined labels
    :return: a csv file with the dataset
    """
    data = pd.DataFrame(dataset.X) #construir de novo o dataframe e para isso basta passar o x
    if features:
        data.columns = dataset.features
    if label:
        data[dataset.label] = dataset.y

    data.to_csv(filename, sep=sep, index=False) #passa os dados do dataset escrito para um ficheito csv


