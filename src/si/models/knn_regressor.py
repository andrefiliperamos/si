from typing import Callable, Union

import numpy as np

from si.data.dataset import Dataset
#from si.metrics.accuracy import accuracy
from si.statistics.euclidean_distance import euclidean_distance
from si.io.csv_file import read_csv
from si.metrics.rmse import rmse

class KNNRegressor:

    def __init__(self, k: int = 1, distance: Callable = euclidean_distance):

        self.k = k
        self.distance = distance

        # attributes
        self.dataset = None
    
    def fit(self, dataset: Dataset) -> 'KNNRegressor':

        self.dataset = dataset
        return self
    
    def _get_closest_label(self, sample: np.ndarray) -> Union[int, str]:
        """
        It returns the closest label of the given sample

        Parameters
        ----------
        sample: np.ndarray
            The sample to get the closest label of

        Returns
        -------
        label: str or int
            The closest label
        """
        # compute the distance between the sample and the dataset
        distances = self.distance(sample, self.dataset.X)
        
        # get the k nearest neighbors
        k_nearest_neighbors = np.argsort(distances)[:self.k]
        print(k_nearest_neighbors)
        # get the labels of the k nearest neighbors
        k_nearest_neighbors_labels = self.dataset.y[k_nearest_neighbors]
        print(k_nearest_neighbors_labels)
        # get the most common label
        labels = np.mean(k_nearest_neighbors_labels)
        
        return labels

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        It predicts the classes of the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the classes of

        Returns
        -------
        predictions: np.ndarray
            The predictions of the model
        """
        return np.apply_along_axis(self._get_closest_label, axis=1, arr=dataset.X)
    
    def score(self, dataset: Dataset) -> float:
        """
        It returns the accuracy of the model on the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to evaluate the model on

        Returns
        -------
        accuracy: float
            The accuracy of the model
        """
        predictions = self.predict(dataset)
        return rmse(dataset.y, predictions)

caminho = 'C:\\Percurso Académico\Mestrado em Bioinformática - UMinho\\2023-2024\\02 Disciplinas\\1º Semestre\\03 Sistemas Inteligentes para a Bioinformática\\si\\datasets\\cpu\\cpu.csv'

data = read_csv(caminho, ',', True, True)
knnr = KNNRegressor(10)
knnr.fit(data)
print(knnr.predict(data))
print(knnr.score(data))