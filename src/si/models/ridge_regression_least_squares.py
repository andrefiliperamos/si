import numpy as np

from si.data.dataset import Dataset
from si.metrics.mse import mse


class RidgeRegressionLeastSquares:

    def __init__(self, l2_penalty, scale):

        '''
        parameters:
        - l2_penalty - L2 regularization parameter 
        - scale - wheter to scale the data or not 
        • estimated parameters:
        - theta - the coefficients of the model for every feature
        - theta_zero - the zero coefficient (y intercept) 
        - mean - mean of the dataset (for every feature) 
        - std - standard deviation of the dataset (for every feature) 
        • methods:
        - fit - estimates the theta and theta_zero coefficients, mean and std
        - predict - predicts the dependent variable (y) using the estimated theta coefficients
        - score - calculates the error between the real and predicted y values
        '''
        #Atributos
        self.l2_penalty = l2_penalty
        self.scale =scale
        #Parámetros Estimados
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None

    def fit(self, dataset: Dataset):

        # fit - estimates the theta and theta_zero coefficients, mean and std
        # 1. Scale the data if required

        if self.scale:
            # compute mean and std
            self.mean = np.nanmean(dataset.X, axis=0)
            self.std = np.nanstd(dataset.X, axis=0)
            # scale the dataset
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        # 2. Add intercept term to X (hint: you can use np.c_ and np.ones to add a column of ones in the first column position)

        X = np.c_[np.ones(X.shape[0]), X]

        # 3. Compute the (penalty term l2_penalty * identity matrix) (hint: you can use np.eye for the identity matrix)
        
        penaltyMatrix = self.l2_penalty*np.eye(X.shape[1])

        # 4. Change the first position of the penalty matrix to 0 (this will make sure that the y intercept coefficient (theta_zero) is not penalized)
        penaltyMatrix[0][0] = 0

        

        '''
        5. Compute the model parameters (theta_zero (first element of the theta vector) and theta (remaining elements))
        Hint: you can use np.linalg.inv to calculate the inverse matrix
        matrix1.dot(matrix2) for matrix multiplication
        matrix.T for the tranpose
        '''
        
        teta = (np.linalg.inv(X.T.dot(X) + penaltyMatrix)).dot(X.T).dot(dataset.y)

        self.theta_zero = np.array(teta[0])

        self.theta = teta[1:]

    def predict(self, dataset: Dataset):

        '''
        1.Scale the data if required (Note: you should use the mean and std estimated in the fit method)
        2.Add intercept term to X (hint: you can use np.c_ and np.ones to add a column of ones in the first column position)
        3.Compute the predicted Y (X * thetas) (Hint: you can use np.r_ to concatenate theta_zero and theta and X.dot(thetas) for matrix multiplication)
        4.Return the predicted Y
        '''
        if self.scale:
            # scale the dataset
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        X = np.c_[np.ones(X.shape[0]), X]

        #print("teste " + np.r_(np.array([self.theta_zero]),self.theta))
        return X.dot(np.r_[self.theta_zero,self.theta])

    def score(self, dataset: Dataset):

        '''
        1. Get the predicted Y using the predict method
        2. Compute the mse score using the mse function
        '''

        y_pred = self.predict(dataset)
        return mse(dataset.y, y_pred)

# This is how you can test it against sklearn to check if everything is fine
if __name__ == '__main__':
    # make a linear dataset
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    dataset_ = Dataset(X=X, y=y)

    # fit the model
    model = RidgeRegressionLeastSquares(0.2, False) # alpha=2.0
    model.fit(dataset_)
    print(model.theta)
    print(model.theta_zero)

    # compute the score
    print(model.score(dataset_))

    # compare with sklearn
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=2.0)
    # scale data
    X = (dataset_.X - np.nanmean(dataset_.X, axis=0)) / np.nanstd(dataset_.X, axis=0)
    model.fit(X, dataset_.y)
    print(model.coef_) # should be the same as theta
    print(model.intercept_) # should be the same as theta_zero
    print(mse(dataset_.y, model.predict(X)))
