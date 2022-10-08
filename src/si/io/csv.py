import pandas as pd
from si.data.dataset import Dataset
def read_csv(filename: str, sep: str = ",", features: bool = False, label: bool = False) -> Dataset:
    '''
    Método que lê um ficheiro csv
    :param filename: str, nome do  csv a ler
    :param sep: str, separador usado para separar os dados no ficheiro, ","
    :param features: boolean, valor que indica se o dataset possui features definidas
    :param label: boolean, valor que indica se o dataset possui labels definidas

    :return: Dataset
    '''
    dataframe = pd.read_csv(filename, delimiter= sep)
    if features and label: #se temos as features e labels
        features = data.colums[:-1]
        x = data.iloc[1:, 0:, -1].to_numpy() #queremos começar na primeira linha #passa para numpy
        y = data.iloc[:, -1].to_numpy()
        features = None
        label = None
    elif features and not label: #se temos as features, mas não as labels
        features = data.columns
        y = None
    elif not features and label: #se não temos as fetaures, mas temos as labels
        label = data.columns[-1]
        y = data.iloc[:, -1]
        data = data.iloc[:, :-1]
    else: #quando não temos nem as features nem as labels
        y = None
    return Dataset(data, y, features, label)

def write_csv_file(filename: str, dataset: Dataset, sep: str = ",", features: bool = False, label: bool = None) -> None:
    '''
    Método que escreve um ficheiro csv
    :param filename: str, nome do ficheiro csv
    :param dataset: o dataset que vai ser escrito
    :param sep: str, separador usado para separar os dados no ficheiro, ","
    :param features: boolean, valor que indica se o dataset possui features definidas
    :param label: boolean, valor que indica se o dataset possui labels definidas

    :return: None
    '''
    data = pd.DataFrame(dataset.x) #construir de novo o dataframe e para isso basta passar o x
    if features:
        data.colums = dataset.features
    if label:
        data[dataset.label] = dataset.y

    data.to_csv(filename, sep=sep, index=False) #passa os dados do dataset escrito para um ficheito csv


