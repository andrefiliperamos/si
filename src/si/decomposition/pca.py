import numpy as np
from si.data.dataset import Dataset

class PCA:

    def __init__(self, n_components):

        # Parameters
        self.n_components = n_components
        # atributtes
        self.mean = 0
        self.explained_variance = None
        self.components = None

    '''
    fit:
    Step 1. Start by centering the data:
        - Infer the mean of the samples.
        - Subtractthemean fromthe dataset (X - mean).
    Step 2. Calculate the SVD:
        - SVD of X can be calculated using the following formula: X = U*S*VT
        - The function numpy.linalg.svd(X, full_matrices=False) gives us U, S, VT
        - U: unitarymatrix of eigenvectors; S: diagonal matrix of eigenvalues; VT: unitary matrix
        of rightsingular vectors
    Step 3. Infer the Principal Components:
        - The principal components(components) correspondto the first n_components of V^T.
    Step 4. Infer the Explained Variance:
        - The explained variance can be calculated using the following formula:
            EV = S^2/(n-1) - where n correspondsto the number of samples, and S is obtained from the SVD.
        - The explained variance (explained_variance) correspondsto the first n_components of EV
    '''
    def fit(self, dataset: Dataset):
        # Step 1
        self.mean = dataset.get_mean()
        X = dataset.X - self.mean   # Confirmar
        
        # Step 2
        (U, S, VT) = np.linalg.svd(X, full_matrices=False)
        
        # Step 3
        self.components = VT[:self.n_components]

        # Step 4
        print(S**2)
        EV = S**2/(len(dataset.X)-1)   # Confirmar
        self.explained_variance = EV[:self.n_components]

        '''
        transform:
       Step 1. Start by Centering the Data:
        - Subtract the mean from the dataset (X - mean).
        - Use the mean inferred in the "fit" method.
        Step 2. Calculate the reduced X:
        - The reduction of X can be calculated using the following formula:
                X_reduced = X*V
        - The function numpy.dot(X, V) - which performs matrix multiplication - gives us the reduction of X to the principal components.
        - NOTE: V corresponds to the transpose matrix of V^T
        '''


    def transform(self, dataset: Dataset):

        # Step 1:

        X = dataset.X - self.mean 

        # Step 2:

        (U, S, VT) = np.linalg.svd(X, full_matrices=False)

        X_reduced = np.dot(X, np.transpose(VT))

        return X_reduced
    
    def fit_transform(self, dataset: Dataset):

        self.fit(dataset)
        return self.transform(dataset)
    
if __name__ == '__main__':

    from si.io.csv_file import read_csv

    ficheiroIris = read_csv('C:\\Percurso Académico\\Mestrado em Bioinformática - UMinho\\2023-2024\\02 Disciplinas\\1º Semestre\\03 Sistemas Inteligentes para a Bioinformática\\si\\datasets\\iris\\iris.csv', features=True, label=True)
    p = PCA(5)
    print(p.fit_transform(ficheiroIris))