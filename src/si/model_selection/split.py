from typing import Tuple

import numpy as np

from si.io.csv_file import read_csv

from si.data.dataset import Dataset


def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into training and testing sets

    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float
        The proportion of the dataset to include in the test split
    random_state: int
        The seed of the random number generator

    Returns
    -------
    train: Dataset
        The training dataset
    test: Dataset
        The testing dataset
    """
    # set random state
    np.random.seed(random_state)
    # get dataset size
    n_samples = dataset.shape()[0]
    # get number of samples in the test set
    n_test = int(n_samples * test_size)
    # get the dataset permutations
    permutations = np.random.permutation(n_samples)
    # get samples in the test set
    test_idxs = permutations[:n_test]
    # get samples in the training set
    train_idxs = permutations[n_test:]
    # get the training and testing datasets
    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)
    return train, test


def stratified_train_test_split(dataset: Dataset, test_Size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Performs a stratified split of the dataset into training and testing sets

    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float
        The proportion of the dataset to include in the test split
    random_state: int
        The seed of the random number generator

    Returns
    -------
    train: Dataset
        The training dataset
    test: Dataset
        The testing dataset
    """
    # Get unique class labels and their counts;
    classes_with_count = dataset.get_classes_with_count()   #tupla, com array de labels unicas e um array das respectivas contagens
                                                            # (['Iris-setosa' 'Iris-versicolor' 'Iris-virginica'], [50, 50, 50])
    
    #  Initialize empty lists for train and test indices;
    i_train = []
    i_test = []

    # number of test samples for each unique label
    nts = classes_with_count[1] * test_Size           # [50 50 50] * 0.2 = [10.0, 10.0, 10.0] = nts
    
    # Loop through unique labels;
    for lb in range(np.size(classes_with_count, axis = 1)):      # np.size(classes_with_count, axis = 1) = 3

        # array for samples of a particular unique label
        aux = []
        i_train_lb = []
        i_test_lb = []
        
        # percorre o array de todas as amostras e coloca em aux as amostras que
        #     correspondem a label com o indice lb
        for x in range(len(dataset.y)):          # len(dataset.y) = 150  - notar que dataset.y tem todas as labels, com repeticoes
           # compara a label de cada amostra com a label que tem indice lb 
           if dataset.y[x] == classes_with_count[0][lb]:      # classes_with_count[0] = ['Iris-setosa' 'Iris-versicolor' 'Iris-virginica']
              aux.append(x)
           # neste ponto, aux esta preenchido com os indices de todos os samples com label de indice lb   

        # para cada label:
        permutations = np.random.permutation(aux)   # baralha os indices do lb
        i_test_lb = permutations[:int(nts[lb])]     # retira o numero pretendido de amostras para o dataset de teste
        i_train_lb = permutations[int(nts[lb]):]    # as restantes amostras sao retiradas para o dataset de treino

        # para a totalidade das amostras:
        i_test.extend(i_test_lb)                    # acrescenta ao dataset de teste as amostras de treino de lb  
        i_train.extend(i_train_lb)                  # acrescenta ao dataset de treino as amostras de treino de lb        

    train = Dataset(dataset.X[i_train], dataset.y[i_train], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[i_test], dataset.y[i_test], features=dataset.features, label=dataset.label)
    return train, test


'''
• algorithm:
- Calculate the number of test samples for the current class;
- Shuffle and select indices for the current class and add them to the test
indices;
- Add the remaining indicesto the train indices;
- After the loop, create training and testing datasets;
- Return the training and testing datasets.
'''

caminho = 'C:\\Percurso Académico\\Mestrado em Bioinformática - UMinho\\2023-2024\\02 Disciplinas\\1º Semestre\\03 Sistemas Inteligentes para a Bioinformática\\0Base de Dados\\TesteExercicio6.csv'
#caminho = 'C:\\Percurso Académico\\Mestrado em Bioinformática - UMinho\\2023-2024\\02 Disciplinas\\1º Semestre\\03 Sistemas Inteligentes para a Bioinformática\\0Base de Dados\\iris.csv'
dados = read_csv(caminho, ',', True, True)
(train, teste) = stratified_train_test_split(dados)
print("Train:")
print(train.X)
print("Teste:")
print(teste.y)

(tr, tt) = train_test_split(dados)
print("Train:")
print(tr.X)
print("Teste:")
print(tt.y)
