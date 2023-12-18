
import numpy as np
from typing import Callable
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification

class SelectPercentile:

    def __init__(self, percentile, score_func: Callable = f_classification) -> None:

        self.score_func = score_func
        self.percentile = percentile
        self.F = None
        self.p = None
    

    def fit(self, dataset: Dataset):

        self.F, self.p = self.score_func(dataset)
        
        return self
    
    def transform(self, dataset: Dataset):
        
        idxs = np.argsort(self.F)[-int (np.size(self.F)*self.percentile):]
        features = np.array(dataset.features)[idxs]
        return Dataset(X=dataset.X[:, idxs], y=dataset.y, features=list(features), label=dataset.label)

    def fit_transform(self, dataset: Dataset):
         
         self.fit(dataset)
         return self.transform(dataset)
    
    